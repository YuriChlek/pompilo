from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from orderflow.market_data.models import ScalpSignal


class BotState(str, Enum):
    IDLE = "idle"
    DEGRADED = "degraded_market_data"
    CANDIDATE = "candidate_detected"
    ENTRY_PENDING = "entry_pending"
    IN_POSITION = "in_position"
    EXITING = "exiting"
    COOLDOWN = "cooldown"


@dataclass(slots=True)
class PendingOrderState:
    order_id: str
    symbol: str
    side: str
    price: float
    size: float
    created_at_ms: int
    signal: ScalpSignal
    status: str = "submitted"
    filled_size: float = 0.0
    average_fill_price: float | None = None


@dataclass(slots=True)
class PositionState:
    symbol: str
    side: str
    size: float
    entry_price: float
    tick_size: float
    opened_at_ms: int
    signal: ScalpSignal
    stop_price: float
    take_profit_price: float
    best_price_seen: float
    status: str = "open"


@dataclass(slots=True)
class SymbolRuntimeState:
    symbol: str
    state: BotState = BotState.IDLE
    last_signal_reason: str = ""
    cooldown_until_ms: int = 0
    active_signal: ScalpSignal | None = None
    pending_order: PendingOrderState | None = None
    position: PositionState | None = None
    last_transition_ms: int = 0
    metadata: dict[str, str | float | int] = field(default_factory=dict)
