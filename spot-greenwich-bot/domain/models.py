from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

SignalType = Literal["buy", "sell", "hold"]
ActionType = Literal["buy", "sell", "skip"]


@dataclass(frozen=True)
class SpotSignal:
    symbol: str
    signal_type: SignalType
    signal_price: Decimal
    close_time: object
    reason: str
    timeframe: str = "d1"
    candle_id: str | None = None


@dataclass(frozen=True)
class MultiTimeframeSignal:
    symbol: str
    d1_regime_blocked: bool
    h4: SpotSignal
    resolved: SpotSignal


@dataclass(frozen=True)
class PositionState:
    symbol: str
    quantity: Decimal
    avg_entry_price: Decimal
    total_cost: Decimal
    entry_count: int = 0
    first_take_profit_done: bool = False

    @property
    def has_position(self) -> bool:
        return self.quantity > 0


@dataclass(frozen=True)
class ExecutionDecision:
    action: ActionType
    symbol: str
    signal_price: Decimal
    quantity: Decimal
    quote_amount: Decimal
    reason: str
    signal_timeframe: str | None = None
    signal_candle_id: str | None = None


@dataclass(frozen=True)
class ExecutionResult:
    executed: bool
    symbol: str
    action: ActionType
    reason: str
    signal_price: Decimal
    executed_price: Decimal | None = None
    quantity: Decimal | None = None
    exchange_order_id: str | None = None
    dry_run: bool = False
    notification_only: bool = False


__all__ = [
    "ActionType",
    "ExecutionDecision",
    "ExecutionResult",
    "MultiTimeframeSignal",
    "PositionState",
    "SignalType",
    "SpotSignal",
]
