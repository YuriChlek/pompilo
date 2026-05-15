from __future__ import annotations

from domain.models import ExecutionDecision, ExecutionResult, PositionState, SpotSignal

from application.ports import PositionExecutor, SignalNotifier


class TradingExecutionService:
    """Own execution-side effects for one finalized trading decision."""

    def __init__(self, executor: PositionExecutor, notifier: SignalNotifier) -> None:
        self.executor = executor
        self.notifier = notifier

    async def execute(
        self,
        signal: SpotSignal,
        decision: ExecutionDecision,
        position_state: PositionState,
        *,
        dry_run: bool = False,
    ) -> ExecutionResult:
        """Execute one decision and publish the resulting notification."""

        result = await self.executor.execute(decision, position_state, dry_run=dry_run)
        await self.notifier.notify(signal, result)
        return result


__all__ = ["TradingExecutionService"]
