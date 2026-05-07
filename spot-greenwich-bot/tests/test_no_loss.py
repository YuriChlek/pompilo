from __future__ import annotations

import asyncio
import importlib
import os
import unittest
from decimal import Decimal
from unittest.mock import patch

from domain.execution import decide_spot_execution
from domain.models import ExecutionDecision, PositionState, SpotSignal
from infrastructure.bybit_spot import BybitSpotFilters
from infrastructure.execution_service import BybitSpotExecutor


class NoLossTests(unittest.TestCase):
    def test_domain_blocks_sell_below_min_profit_ratio(self) -> None:
        signal = SpotSignal("ETHUSDT", "sell", Decimal("100"), "2026-01-01", "test")
        state = PositionState("ETHUSDT", Decimal("1"), Decimal("100"), Decimal("100"))

        decision = decide_spot_execution(signal, state, Decimal("500"))

        self.assertEqual(decision.action, "skip")
        self.assertEqual(decision.reason, "sell_price_not_profitable")

    def test_domain_allows_sell_at_min_profit_ratio(self) -> None:
        signal = SpotSignal("ETHUSDT", "sell", Decimal("101"), "2026-01-01", "test")
        state = PositionState("ETHUSDT", Decimal("1"), Decimal("100"), Decimal("100"))

        decision = decide_spot_execution(signal, state, Decimal("500"))

        self.assertEqual(decision.action, "sell")

    def test_no_loss_guard_blocks_sell_when_market_price_dropped(self) -> None:
        class _Client:
            def get_symbol_filters(self, symbol: str) -> BybitSpotFilters:
                return BybitSpotFilters(symbol, Decimal("0.001"), Decimal("1000"), Decimal("0.001"), Decimal("5"), Decimal("0.01"))

            def fetch_current_price(self, symbol: str) -> Decimal:
                return Decimal("99")

            def place_market_order(self, symbol: str, side: str, quantity: Decimal):
                raise AssertionError("order should be blocked before exchange placement")

        executor = BybitSpotExecutor(client=_Client())
        calls = []

        async def _record_ledger(decision, position_state, executed_price, exchange_order_id, status, **kwargs):
            calls.append((status, kwargs.get("no_loss_check_price")))

        async def _has_executed_signal(decision):
            return False

        executor._record_ledger = _record_ledger  # type: ignore[attr-defined]
        executor._has_executed_signal = _has_executed_signal  # type: ignore[method-assign]
        decision = ExecutionDecision("sell", "ETHUSDT", Decimal("101"), Decimal("1"), Decimal("101"), "greenwich_profitable_exit", "h4", "2026-01-01")
        state = PositionState("ETHUSDT", Decimal("1"), Decimal("100"), Decimal("100"))

        with self.assertLogs("infrastructure.execution_service", level="WARNING"):
            result = asyncio.run(executor.execute(decision, state, dry_run=False))

        self.assertFalse(result.executed)
        self.assertEqual(result.action, "skip")
        self.assertEqual(result.reason, "no_loss_guard_infrastructure")
        self.assertEqual(calls, [("blocked_no_loss", Decimal("99"))])

    def test_duplicate_signal_order_is_skipped(self) -> None:
        class _Client:
            def get_symbol_filters(self, symbol: str) -> BybitSpotFilters:
                return BybitSpotFilters(symbol, Decimal("0.001"), Decimal("1000"), Decimal("0.001"), Decimal("5"), Decimal("0.01"))

            def place_market_order(self, symbol: str, side: str, quantity: Decimal):
                raise AssertionError("duplicate signal should not place an order")

        executor = BybitSpotExecutor(client=_Client())
        calls = []

        async def _record_ledger(decision, position_state, executed_price, exchange_order_id, status, **kwargs):
            calls.append(status)

        async def _has_executed_signal(decision):
            return True

        executor._record_ledger = _record_ledger  # type: ignore[attr-defined]
        executor._has_executed_signal = _has_executed_signal  # type: ignore[method-assign]
        decision = ExecutionDecision("buy", "ETHUSDT", Decimal("100"), Decimal("0.1"), Decimal("10"), "test_buy", "h4", "2026-01-01")
        state = PositionState("ETHUSDT", Decimal("0"), Decimal("0"), Decimal("0"))

        result = asyncio.run(executor.execute(decision, state, dry_run=False))

        self.assertEqual(result.action, "skip")
        self.assertEqual(result.reason, "duplicate_signal_order")
        self.assertEqual(calls, ["skipped"])

    def test_order_status_must_be_filled_after_market_order(self) -> None:
        class _Client:
            def get_symbol_filters(self, symbol: str) -> BybitSpotFilters:
                return BybitSpotFilters(symbol, Decimal("0.001"), Decimal("1000"), Decimal("0.001"), Decimal("5"), Decimal("0.01"))

            def place_market_order(self, symbol: str, side: str, quantity: Decimal):
                return {"retCode": 0, "result": {"orderId": "abc-123", "avgPrice": "100"}}

            def extract_fill_price(self, order_payload, fallback_price: Decimal) -> Decimal:
                return Decimal("100")

            def get_order_status(self, symbol: str, order_id: str):
                return {"status": "Cancelled"}

        executor = BybitSpotExecutor(client=_Client())

        async def _has_executed_signal(decision):
            return False

        executor._has_executed_signal = _has_executed_signal  # type: ignore[method-assign]
        decision = ExecutionDecision("buy", "ETHUSDT", Decimal("100"), Decimal("0.1"), Decimal("10"), "test_buy", "h4", "2026-01-01")
        state = PositionState("ETHUSDT", Decimal("0"), Decimal("0"), Decimal("0"))

        with self.assertRaises(RuntimeError):
            asyncio.run(executor.execute(decision, state, dry_run=False))

    def test_min_profit_ratio_below_minimum_raises_on_startup(self) -> None:
        import utils.config as config

        with patch.dict(os.environ, {"MIN_PROFIT_RATIO": "0.001"}, clear=False):
            with self.assertRaises(ValueError):
                importlib.reload(config)
        with patch.dict(os.environ, {"MIN_PROFIT_RATIO": "0.01"}, clear=False):
            importlib.reload(config)
