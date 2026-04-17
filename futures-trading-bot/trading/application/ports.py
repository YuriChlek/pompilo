from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Protocol, Tuple


class MarketDataProvider(Protocol):
    """Application port for retrieving market context before signal generation."""

    def get_market_context(self, symbol: str, is_test: bool) -> Tuple[Any, Any]:
        """Return the latest trend snapshot and indicator history for a symbol."""


class SignalNotifier(Protocol):
    """Application port for sending trade and position-management notifications."""

    async def notify_new_position(
        self,
        symbol: str,
        direction: str,
        price,
        take_profit,
        stop_loss,
        strategy_mode: str,
    ) -> None:
        """Send a notification about a newly created position payload."""

    async def notify_position_moved_to_breakeven(
        self,
        symbol: str,
        direction: str,
        entry_price,
        current_price,
        partial_close_qty=None,
    ) -> None:
        """Send a notification when an open position is moved to breakeven."""


class PositionExecutor(Protocol):
    """Application port for live position execution and management."""

    async def reconcile_state(self, symbols: Iterable[str]) -> None:
        """Synchronize live exchange state into the local persistence layer."""

    async def manage_open_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Manage an open position and return an event payload when state changes."""

    async def execute(self, position: Optional[Dict[str, Any]]) -> bool:
        """Apply the generated position to the external execution venue and return success status."""


class MarketDataSynchronizer(Protocol):
    """Application port for refreshing external market data on schedule."""

    async def synchronize(self) -> None:
        """Refresh external market data before signal generation starts."""
