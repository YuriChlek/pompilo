from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol

from trading.domain.models import ScalpSignal

__all__ = [
    "AccountPort",
    "EntryStatus",
    "ExecutionOrder",
    "ExecutionPort",
    "ExecutionStreamEvent",
    "FuturesTickerSnapshot",
    "LivePositionSnapshot",
    "MarketDataProvider",
    "OrderStatusSnapshot",
    "OrderStreamEvent",
    "OrderSubmissionResult",
    "PositionStatePort",
    "PositionStreamEvent",
    "RuntimeRepositoryPort",
    "SignalNotifier",
    "StopMoveResult",
]


@dataclass(frozen=True)
class ExecutionOrder:
    symbol: str
    direction: str
    order_type: str
    size: Decimal
    price: Decimal | None
    stop_loss: Decimal | None
    take_profit: Decimal | None


@dataclass(frozen=True)
class OrderSubmissionResult:
    status: str
    symbol: str
    order_id: str | None = None
    side: str | None = None
    size: Decimal | None = None
    reason: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderStatusSnapshot:
    status: str
    order_id: str
    symbol: str
    side: str | None = None
    price: Decimal | None = None
    qty: Decimal | None = None
    cum_exec_qty: Decimal | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LivePositionSnapshot:
    symbol: str
    direction: str
    size: Decimal
    avg_price: Decimal
    take_profit: Decimal | None = None
    stop_loss: Decimal | None = None


@dataclass(frozen=True)
class FuturesTickerSnapshot:
    bid: Decimal
    ask: Decimal
    mark: Decimal
    last: Decimal
    mid: Decimal


@dataclass(frozen=True)
class StopMoveResult:
    status: str
    symbol: str
    stop_price: Decimal
    reason: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderStreamEvent:
    order_id: str
    order_link_id: str | None
    symbol: str
    side: str | None
    status: str | None
    price: Decimal | None
    qty: Decimal | None
    cum_exec_qty: Decimal | None
    avg_price: Decimal | None
    take_profit: Decimal | None
    stop_loss: Decimal | None
    updated_time: int | None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionStreamEvent:
    order_id: str
    order_link_id: str | None
    symbol: str
    side: str | None
    exec_price: Decimal | None
    exec_qty: Decimal | None
    leaves_qty: Decimal | None
    exec_time: int | None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PositionStreamEvent:
    symbol: str
    side: str | None
    size: Decimal
    avg_price: Decimal | None
    take_profit: Decimal | None
    stop_loss: Decimal | None
    updated_time: int | None
    raw: dict[str, Any] = field(default_factory=dict)


class MarketDataProvider(Protocol):
    """Provides normalized market data required by the application layer."""

    def get_best_reference_book(self, symbol: str, now_ms: int | None = None) -> Any:
        """Return the best available reference order book for a symbol."""

    def get_best_reference_exchange(self, symbol: str, now_ms: int | None = None) -> str | None:
        """Return the selected reference exchange for a symbol."""


class AccountPort(Protocol):
    """Reads normalized account-level state from the execution venue."""

    async def fetch_account_equity(self) -> float:
        """Return current normalized account equity."""


class PositionStatePort(Protocol):
    """Reads normalized position and instrument state from the execution venue."""

    async def detect_live_position(self, symbol: str) -> LivePositionSnapshot | None:
        """Return the current live position for a symbol, if any."""

    async def fetch_futures_tick_size(self, symbol: str) -> float | None:
        """Return the normalized futures tick size for a symbol."""


class SignalNotifier(Protocol):
    """Sends external notifications for trading events."""

    async def notify_entry_submitted(self, symbol: str, payload: dict[str, Any]) -> None:
        """Notify that an entry order was submitted."""

    async def notify_stop_moved(self, symbol: str, payload: dict[str, Any]) -> None:
        """Notify that a stop was moved."""


class RuntimeRepositoryPort(Protocol):
    """Persists runtime, order, and position events."""

    async def insert_order_event(self, symbol: str, event_type: str, payload: dict[str, Any]) -> None:
        """Persist a normalized order event."""

    async def insert_position_event(self, **payload: Any) -> None:
        """Persist a normalized position event."""

    async def insert_runtime_transition(self, **payload: Any) -> None:
        """Persist a normalized runtime state transition."""


class ExecutionPort(AccountPort, PositionStatePort, Protocol):
    """Application-facing execution contract.

    The application layer depends on this protocol instead of concrete
    exchange clients or execution service implementations.
    """

    async def execute(self, signal: ScalpSignal, order_data: ExecutionOrder, dry_run: bool = True) -> OrderSubmissionResult:
        """Submit an entry order request."""

    async def poll_entry(self, symbol: str, order_id: str, dry_run: bool = True) -> OrderStatusSnapshot:
        """Read the current normalized status of an entry order."""

    async def cancel_entry(self, symbol: str, order_id: str, dry_run: bool = True, reason: str = "cancelled") -> OrderSubmissionResult:
        """Cancel an entry order."""

    async def exit_position(self, symbol: str, side: str, size: float, dry_run: bool = True, reason: str = "exit") -> OrderSubmissionResult:
        """Submit a position exit request."""

    async def move_stop_to_breakeven(
        self,
        symbol: str,
        side: str,
        stop_price: float,
        current_price: float,
        dry_run: bool = True,
        reason: str = "breakeven",
    ) -> StopMoveResult:
        """Move the stop loss to a normalized breakeven level."""

    async def fetch_futures_ticker(self, symbol: str) -> FuturesTickerSnapshot | None:
        """Return normalized futures ticker data used for basis calculations."""

    def supports_private_execution_stream(self) -> bool:
        """Return whether private execution streaming is available."""

    async def stream_private_execution_events(
        self,
        on_order_update,
        on_execution_update,
        on_position_update,
    ) -> None:
        """Run the private execution stream and invoke callbacks with normalized events."""

    async def close(self) -> None:
        """Release execution-layer resources."""
