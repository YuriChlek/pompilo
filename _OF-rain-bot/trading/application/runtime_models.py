from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from trading.domain.diagnostics import SignalDiagnostics


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    NONE = "none"


@dataclass(slots=True)
class BookLevel:
    price: float
    size: float
    notional: float
    distance_ticks: int
    distance_bps: float


@dataclass(slots=True)
class OrderBookSnapshot:
    exchange: str
    symbol: str
    timestamp_ms: int
    bids: list[BookLevel]
    asks: list[BookLevel]
    best_bid: float
    best_ask: float
    mid_price: float
    spread_ticks: int
    tick_size: float = 0.0


@dataclass(slots=True)
class TradePrint:
    exchange: str
    symbol: str
    timestamp_ms: int
    price: float
    size: float
    side: str
    notional: float


@dataclass(slots=True)
class FeedHealth:
    exchange: str
    connected: bool = False
    transport_connected: bool = False
    subscribed: bool = False
    last_connect_started_ms: int = 0
    last_transport_connected_ms: int = 0
    last_subscribed_at_ms: int = 0
    last_connected_at_ms: int = 0
    last_disconnected_at_ms: int = 0
    last_snapshot_at_ms: int = 0
    last_trade_at_ms: int = 0
    snapshot_count: int = 0
    trade_count: int = 0
    reconnect_count: int = 0
    connection_attempt_count: int = 0
    current_session_id: str = ""
    last_error: str = ""
    last_disconnect_reason: str = ""
    last_reconnect_reason: str = ""


@dataclass(slots=True)
class TapeWindowStats:
    symbol: str
    exchange: str | None
    window_ms: int
    buy_notional: float
    sell_notional: float
    buy_count: int
    sell_count: int
    last_price: float | None
    exchange_count: int = 1

    @property
    def delta_notional(self) -> float:
        return self.buy_notional - self.sell_notional

    @property
    def dominant_side(self) -> str:
        if self.buy_notional > self.sell_notional:
            return "buy"
        if self.sell_notional > self.buy_notional:
            return "sell"
        return "neutral"

@dataclass(slots=True)
class LiquidityWall:
    exchange: str
    symbol: str
    side: str
    price: float
    size: float
    notional: float
    distance_ticks: int
    distance_bps: float
    first_seen_ms: int
    last_seen_ms: int
    persistence_ms: int
    relative_size_ratio: float
    size_stability_score: float
    pull_count: int
    test_count: int
    reload_count: int
    defended_count: int
    chase_count: int
    score: float
    spoof_risk_score: float
    metadata: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(slots=True)
class ScalpSignal:
    symbol: str
    direction: SignalDirection
    wall: LiquidityWall | None
    confidence: float
    reason: str
    analysis_entry_price: float | None = None
    analysis_stop_price: float | None = None
    analysis_take_profit_price: float | None = None
    analysis_invalidation_price: float | None = None
    execution_entry_price: float | None = None
    execution_stop_price: float | None = None
    execution_take_profit_price: float | None = None
    execution_invalidation_price: float | None = None
    basis_bps: float | None = None
    tape_bias: str = "neutral"
    diagnostics: SignalDiagnostics | None = None
    metadata: dict[str, float | int | str] = field(default_factory=dict)

    @property
    def entry_price(self) -> float | None:
        return self.execution_entry_price

    @property
    def stop_price(self) -> float | None:
        return self.execution_stop_price

    @property
    def take_profit_price(self) -> float | None:
        return self.execution_take_profit_price

    @property
    def invalidation_price(self) -> float | None:
        return self.execution_invalidation_price
