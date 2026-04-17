from __future__ import annotations

import asyncio
import unittest
from decimal import Decimal

from trading.domain.models import ExecutionDecision, PositionState
from trading.infrastructure.execution_service import BinanceSpotExecutor


class DryRunExecutionTests(unittest.TestCase):
    def test_dry_run_returns_non_executed_result_without_touching_exchange(self) -> None:
        class _Client:
            def place_market_order(self, symbol, side, quantity):
                raise AssertionError("exchange call is not expected in dry-run")

        executor = BinanceSpotExecutor(client=_Client())

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
