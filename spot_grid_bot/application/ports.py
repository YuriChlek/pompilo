from __future__ import annotations

from typing import Iterable, Protocol

from domain.models import MarketContext, StrategyDecision, SymbolRuntimeState, TargetOrder


class MarketDataProvider(Protocol):
    """Application port for loading market context before one trading cycle."""

    async def get_market_context(self, symbol: str, *, persisted_cost_basis: float | None = None) -> MarketContext:
        """Return candles, balances, and currently open orders for one symbol."""


class OrderExecutor(Protocol):
    """Application port for syncing target orders to the external venue."""

    async def reconcile_state(self, symbols: Iterable[str]) -> None:
        """Synchronize execution state before trading starts."""

    async def sync_orders(self, symbol: str, target_orders: list[TargetOrder]) -> bool:
        """Apply target orders to the execution venue and return whether sync happened."""


class SignalNotifier(Protocol):
    """Application port for outbound notifications about trading decisions."""

    async def notify_rebuild(self, decision: StrategyDecision) -> None:
        """Send a notification about a grid rebuild."""


class MarketDataSynchronizer(Protocol):
    """Application port for refreshing external market data on schedule."""

    async def synchronize(self) -> None:
        """Refresh external market data before the trading cycle starts."""


class StateStore(Protocol):
    """Application port for persisting and restoring per-symbol runtime state."""

    async def initialize(self) -> None:
        """Prepare the persistent runtime state storage."""

    async def load_symbol_state(self, symbol: str) -> SymbolRuntimeState | None:
        """Load runtime state for one symbol if available."""

    async def save_symbol_state(self, state: SymbolRuntimeState) -> None:
        """Persist runtime state for one symbol."""
