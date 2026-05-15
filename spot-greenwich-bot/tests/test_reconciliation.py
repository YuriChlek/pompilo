from __future__ import annotations

import asyncio
import unittest
from decimal import Decimal
from types import SimpleNamespace

from infrastructure.bybit_spot import derive_avg_entry_price_from_trades, split_symbol
from infrastructure.execution_service import BybitSpotExecutor
from domain.models import PositionState


class ReconciliationTests(unittest.TestCase):
    def test_symbol_is_split_into_base_and_quote_assets(self) -> None:
        self.assertEqual(split_symbol("ETHUSDT"), ("ETH", "USDT"))

    def test_avg_entry_price_is_derived_from_remaining_inventory(self) -> None:
        trades = [
            {"execId": "1", "execTime": 1, "side": "Buy", "execQty": "1", "execPrice": "100"},
            {"execId": "2", "execTime": 2, "side": "Buy", "execQty": "1", "execPrice": "120"},
            {"execId": "3", "execTime": 3, "side": "Sell", "execQty": "1", "execPrice": "140"},
        ]
        result = derive_avg_entry_price_from_trades("ETHUSDT", Decimal("1"), trades)
        self.assertEqual(result, Decimal("120"))

    def test_avg_entry_returns_zero_when_position_is_flat(self) -> None:
        trades = [
            {"execId": "1", "execTime": 1, "side": "Buy", "execQty": "1", "execPrice": "100"},
            {"execId": "2", "execTime": 2, "side": "Sell", "execQty": "1", "execPrice": "110"},
        ]
        result = derive_avg_entry_price_from_trades("ETHUSDT", Decimal("0"), trades)
        self.assertEqual(result, Decimal("0"))

    def test_entry_count_is_resolved_from_executed_buys_after_last_sell(self) -> None:
        class _Connection:
            async def fetchval(self, sql, *args):
                if "MAX(id)" in sql:
                    return 10
                return 2

        executor = BybitSpotExecutor(client=object())

        result = asyncio.run(executor._resolve_entry_count_from_order_ledger(_Connection(), "ETHUSDT", 0))

        self.assertEqual(result, 2)

    def test_entry_count_falls_back_to_one_for_open_position_without_ledger_history(self) -> None:
        class _Connection:
            async def fetchval(self, sql, *args):
                return 0

        executor = BybitSpotExecutor(client=object())

        result = asyncio.run(executor._resolve_entry_count_from_order_ledger(_Connection(), "ETHUSDT", 0))

        self.assertEqual(result, 1)

    def test_reconciliation_restores_first_take_profit_done_from_exit_state(self) -> None:
        class _Client:
            def fetch_asset_balance(self, asset: str):
                if asset == "ETH":
                    return SimpleNamespace(total=Decimal("1"))
                return SimpleNamespace(total=Decimal("0"), free=Decimal("0"))

            def fetch_my_trades(self, symbol: str):
                return [{"execId": "1", "execTime": 1, "side": "Buy", "execQty": "1", "execPrice": "100"}]

        class _Connection:
            async def execute(self, sql, *args):
                return None

        executor = BybitSpotExecutor(client=_Client())
        local_state = PositionState("ETHUSDT", Decimal("1"), Decimal("100"), Decimal("100"), entry_count=1, first_take_profit_done=False)

        async def _resolve_entry_count(conn, symbol: str, fallback_entry_count: int) -> int:
            return fallback_entry_count

        async def _load_position_exit_state(conn, symbol: str) -> dict | None:
            return {"first_take_profit_done": True}

        executor._resolve_entry_count_from_order_ledger = _resolve_entry_count
        executor._load_position_exit_state = _load_position_exit_state

        result = asyncio.run(executor._reconcile_position_state(_Connection(), "ETHUSDT", local_state))

        self.assertTrue(result.first_take_profit_done)
