from __future__ import annotations

from typing import Any, Protocol

from decimal import Decimal

from domain.models import ExecutionDecision, ExecutionResult, PositionState, SpotSignal


class MarketDataProvider(Protocol):
    """Application port for loading market data required for one trading cycle."""

    def get_symbol_history(self, symbol: str) -> Any:
        """Return candle history for one symbol."""


class MarketDataSynchronizer(Protocol):
    """Application port for refreshing external market data before a cycle."""

    async def synchronize(self) -> None:
        """Refresh market data and return when the local store is up to date."""


class PositionExecutor(Protocol):
    """Application port for reading position state and executing one decision."""

    async def get_position_state(self, symbol: str) -> PositionState:
        """Return the current reconciled position state for one symbol."""

    async def get_quote_balance(self, symbol: str) -> Decimal:
        """Return the spendable quote-asset balance for one symbol."""

    async def execute(
        self,
        decision: ExecutionDecision,
        position_state: PositionState,
        *,
        dry_run: bool = False,
    ) -> ExecutionResult:
        """Execute one decision for one symbol and return the execution result."""


class SignalNotifier(Protocol):
    """Application port for outbound reporting of signal-processing results."""

    async def notify(self, signal: SpotSignal, result: ExecutionResult) -> None:
        """Publish a notification for one processed signal."""


class StateStore(Protocol):
    """Optional application port for persisting lightweight symbol runtime state."""

    async def initialize(self) -> None:
        """Prepare the backing store for later reads and writes."""

    async def load_symbol_state(self, symbol: str) -> object | None:
        """Load runtime state for one symbol if available."""

    async def save_symbol_state(self, symbol: str, state: object) -> None:
        """Persist runtime state for one symbol."""


__all__ = [
    "MarketDataProvider",
    "MarketDataSynchronizer",
    "PositionExecutor",
    "SignalNotifier",
    "StateStore",
]
