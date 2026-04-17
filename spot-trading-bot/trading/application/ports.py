from __future__ import annotations

from typing import Protocol

from trading.domain.models import ExecutionDecision, ExecutionResult, PositionState, SpotSignal


class MarketDataProvider(Protocol):
    def get_symbol_history(self, symbol: str):
        ...


class MarketDataSynchronizer(Protocol):
    async def synchronize(self) -> None:
        ...


class PositionExecutor(Protocol):
    async def get_position_state(self, symbol: str) -> PositionState:
        ...

    async def execute(
        self,
        decision: ExecutionDecision,
        position_state: PositionState,
        *,
        dry_run: bool = False,
    ) -> ExecutionResult:
        ...


class SignalNotifier(Protocol):
    async def notify(self, signal: SpotSignal, result: ExecutionResult) -> None:
        ...
