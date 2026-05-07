from __future__ import annotations

import asyncio
import unittest
from decimal import Decimal
from unittest.mock import patch

from domain.models import ExecutionDecision, PositionState
from infrastructure.bybit_spot import BybitSpotFilters
from infrastructure.execution_service import BybitSpotExecutor


class DryRunExecutionTests(unittest.TestCase):
    def test_dry_run_returns_non_executed_result_without_touching_exchange(self) -> None:
        class _Client:
            def place_market_order(self, symbol, side, quantity):
                raise AssertionError("exchange call is not expected in dry-run")

        executor = BybitSpotExecutor(client=_Client())

        calls = []

        async def _record_ledger(decision, position_state, executed_price, exchange_order_id, status, **kwargs):
            calls.append((decision.action, status, executed_price, exchange_order_id))

        executor._record_ledger = _record_ledger  # type: ignore[attr-defined]
        decision = ExecutionDecision("buy", "ETHUSDT", Decimal("100"), Decimal("0.5"), Decimal("50"), "greenwich_accumulation_buy")
        state = PositionState("ETHUSDT", Decimal("1"), Decimal("120"), Decimal("120"))

        result = asyncio.run(executor.execute(decision, state, dry_run=True))

        self.assertFalse(result.executed)
        self.assertTrue(result.dry_run)
        self.assertEqual(result.executed_price, Decimal("100"))
        self.assertEqual(calls, [("buy", "dry_run", Decimal("100"), None)])

    def test_live_execution_reads_bybit_order_id_from_result_payload(self) -> None:
        class _Client:
            def get_symbol_filters(self, symbol: str) -> BybitSpotFilters:
                return BybitSpotFilters(
                    symbol=symbol,
                    min_qty=Decimal("0.001"),
                    max_qty=Decimal("1000"),
                    step_size=Decimal("0.001"),
                    min_notional=Decimal("5"),
                    tick_size=Decimal("0.01"),
                )

            def place_market_order(self, symbol: str, side: str, quantity: Decimal):
                return {"retCode": 0, "result": {"orderId": "abc-123", "avgPrice": "101"}}

            def extract_fill_price(self, order_payload, fallback_price: Decimal) -> Decimal:
                return Decimal(str(order_payload["result"]["avgPrice"]))

            def get_order_status(self, symbol: str, order_id: str):
                return {"status": "Filled"}

        executor = BybitSpotExecutor(client=_Client())
        calls = []

        async def _apply_position_update(decision, position_state, executed_price, exchange_order_id, order_payload, **kwargs):
            calls.append((decision.quantity, executed_price, exchange_order_id))

        executor._apply_position_update = _apply_position_update  # type: ignore[attr-defined]
        decision = ExecutionDecision("buy", "ETHUSDT", Decimal("100"), Decimal("0.123456"), Decimal("12.3456"), "test_buy")
        state = PositionState("ETHUSDT", Decimal("0"), Decimal("0"), Decimal("0"))

        result = asyncio.run(executor.execute(decision, state, dry_run=False))

        self.assertTrue(result.executed)
        self.assertEqual(result.exchange_order_id, "abc-123")
        self.assertEqual(calls, [(Decimal("0.123"), Decimal("101"), "abc-123")])

    def test_position_update_and_ledger_are_written_in_one_transaction(self) -> None:
        events: list[str] = []

        class _Transaction:
            async def __aenter__(self):
                events.append("transaction_enter")

            async def __aexit__(self, exc_type, exc, traceback):
                events.append("transaction_exit")

        class _Connection:
            def transaction(self):
                return _Transaction()

            async def execute(self, sql, *args):
                if "order_ledger" in sql:
                    events.append("ledger_insert")
                else:
                    events.append("position_upsert")

            async def close(self):
                events.append("close")

        async def _create_connection():
            return _Connection()

        executor = BybitSpotExecutor(client=object())
        decision = ExecutionDecision("buy", "ETHUSDT", Decimal("100"), Decimal("0.5"), Decimal("50"), "test_buy")
        state = PositionState("ETHUSDT", Decimal("1"), Decimal("90"), Decimal("90"))

        with patch("infrastructure.execution_service._create_connection", _create_connection):
            asyncio.run(executor._apply_position_update(decision, state, Decimal("100"), "abc-123", {"result": {"orderId": "abc-123"}}))

        self.assertEqual(
            events,
            [
                "transaction_enter",
                "position_upsert",
                "ledger_insert",
                "transaction_exit",
                "close",
            ],
        )
