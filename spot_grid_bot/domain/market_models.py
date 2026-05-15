from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from domain.inventory_models import InventorySnapshot

if TYPE_CHECKING:
    from domain.order_models import LiveOrder


class RegimeType(str, Enum):
    """High-level market regime used by the strategy planner."""

    RANGE = "RANGE"
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    RISK_OFF = "RISK_OFF"


@dataclass(slots=True, frozen=True)
class Candle:
    """Immutable OHLCV candle used across indicators, regime, and backtesting flows."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(slots=True, frozen=True)
class IndicatorSnapshot:
    """Computed indicator bundle used by regime, risk, and strategy decisions."""

    ema20: float
    ema50: float
    ema200: float
    atr14: float
    realized_volatility: float
    ema50_slope: float
    range_width: float
    price_vs_ema50: float
    directional_move: float
    directional_sign: float
    abnormal_candle: bool
    atr_spike: bool
    range_position: float = 0.5
    rsi14: float = 50.0
    current_volume: float = 0.0
    volume_ma20: float = 0.0
    realized_volatility_short: float = 0.0
    volatility_regime_ratio: float = 1.0


@dataclass(slots=True, frozen=True)
class RegimeSnapshot:
    """Current regime classification with confidence and explanatory reasons."""

    regime: RegimeType
    confidence: float
    reasons: list[str] = field(default_factory=list)
    volume_confirmed: bool = True


@dataclass(slots=True, frozen=True)
class VenueConstraints:
    """Venue-specific symbol filters used to make planner output exchange-aware."""

    tick_size: float
    qty_step: float
    min_order_qty: float
    min_order_amt: float


@dataclass(slots=True, frozen=True)
class MarketContext:
    """Input market context required to plan one symbol trading cycle."""

    symbol: str
    candles: list[Candle]
    inventory: InventorySnapshot
    live_orders: list[LiveOrder]
    venue_constraints: VenueConstraints | None = None
    higher_timeframe_candles: list[Candle] | None = None
