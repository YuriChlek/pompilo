from __future__ import annotations

import asyncio
import unittest
from decimal import Decimal

from application.execution_service import TradingExecutionService
from domain.models import ExecutionDecision, ExecutionResult, PositionState, SpotSignal


class TradingExecutionServiceTests(unittest.TestCase):
    def test_execute_delegates_to_executor_and_notifier(self) -> None:
        calls: list[tuple[str, str, object]] = []

        class _Executor:
            async def execute(self, decision, position_state, *, dry_run: bool = False) -> ExecutionResult:
                calls.append(("execute", decision.action, dry_run))
                return ExecutionResult(
                    executed=False,
                    symbol=decision.symbol,
                    action=decision.action,
                    reason=decision.reason,
                    signal_price=decision.signal_price,
                    dry_run=dry_run,
                )

        class _Notifier:
            async def notify(self, signal: SpotSignal, result: ExecutionResult) -> None:
                calls.append(("notify", signal.signal_type, result.action))

        service = TradingExecutionService(_Executor(), _Notifier())
        signal = SpotSignal("ETHUSDT", "buy", Decimal("100"), "2026-01-01", "test_signal")
        decision = ExecutionDecision("buy", "ETHUSDT", Decimal("100"), Decimal("0.5"), Decimal("50"), "test_decision")
        state = PositionState("ETHUSDT", Decimal("0"), Decimal("0"), Decimal("0"))

        result = asyncio.run(service.execute(signal, decision, state, dry_run=True))

        self.assertTrue(result.dry_run)
        self.assertEqual(calls, [("execute", "buy", True), ("notify", "buy", "buy")])
