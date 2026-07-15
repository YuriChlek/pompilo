from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum


class GreenwichSignalType(StrEnum):
    """Pure Greenwich signal types."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class GreenwichActionType(StrEnum):
    """Pure Greenwich decision actions."""

    BUY = "buy"
    SELL = "sell"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class GreenwichCandle:
    """Pure candle model used by platform-native Greenwich planning."""

    timestamp: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass(frozen=True, slots=True)
class GreenwichSignalConfig:
    """Signal-generation settings for Greenwich bands and filters."""

    length: int = 98
    basis_type: str = "WMA"
    multiplier_1: Decimal = Decimal("5.5")
    multiplier_2: Decimal = Decimal("4.5")
    multiplier_3: Decimal = Decimal("3.5")
    confirmation_candle_enabled: bool = True
    anti_crash_buy_block_enabled: bool = True
    anti_crash_lookback_candles: int = 3
    anti_crash_max_drop_ratio: Decimal = Decimal("0.10")
    atr_position_sizing_enabled: bool = True
    atr_position_sizing_median_window: int = 50
    atr_position_sizing_min_multiplier: Decimal = Decimal("0.5")
    atr_position_sizing_max_multiplier: Decimal = Decimal("1.5")


@dataclass(frozen=True, slots=True)
class GreenwichExecutionConfig:
    """Execution-policy settings used before conversion to platform signals."""

    deposit_percent: Decimal = Decimal("5")
    averaging_entry_limit: int = 3
    averaging_entry_2_size_percent: Decimal = Decimal("60")
    averaging_entry_3_size_percent: Decimal = Decimal("30")
    min_profit_ratio: Decimal = Decimal("0.01")
    portfolio_cap_enabled: bool = True
    portfolio_position_limit: int = 3
    portfolio_priority_symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")


@dataclass(frozen=True, slots=True)
class GreenwichConfig:
    """Combined strategy settings for one platform-native Greenwich run."""

    symbols: tuple[str, ...]
    primary_timeframe: str
    supporting_timeframes: tuple[str, ...]
    signal: GreenwichSignalConfig = GreenwichSignalConfig()
    execution: GreenwichExecutionConfig = GreenwichExecutionConfig()
    emit_diagnostics: bool = True


@dataclass(frozen=True, slots=True)
class GreenwichSignalSnapshot:
    """Latest Greenwich indicator state for one candle history."""

    basis: Decimal
    upper1: Decimal
    upper2: Decimal
    upper3: Decimal
    lower1: Decimal
    lower2: Decimal
    lower3: Decimal
    buy_signal: bool
    sell_signal: bool
    signal_price: Decimal
    signal_high: Decimal
    close_time: str


@dataclass(frozen=True, slots=True)
class GreenwichSpotSignal:
    """Greenwich strategy signal before platform BotSignal conversion."""

    symbol: str
    signal_type: GreenwichSignalType
    signal_price: Decimal
    close_time: str
    reason: str
    timeframe: str = "1d"
    candle_id: str | None = None


@dataclass(frozen=True, slots=True)
class GreenwichMultiTimeframeSignal:
    """Raw and resolved multi-timeframe Greenwich signals."""

    symbol: str
    d1_regime_blocked: bool
    h4: GreenwichSpotSignal
    resolved: GreenwichSpotSignal


@dataclass(frozen=True, slots=True)
class GreenwichPositionState:
    """Position state consumed by the pure Greenwich decision policy."""

    symbol: str
    quantity: Decimal
    avg_entry_price: Decimal
    total_cost: Decimal
    entry_count: int = 0
    first_take_profit_done: bool = False

    @property
    def has_position(self) -> bool:
        return self.quantity > 0


@dataclass(frozen=True, slots=True)
class GreenwichExecutionDecision:
    """Signal-only execution decision before adapter-level signal mapping."""

    action: GreenwichActionType
    symbol: str
    signal_price: Decimal
    quantity: Decimal
    quote_amount: Decimal
    reason: str
    signal_timeframe: str | None = None
    signal_candle_id: str | None = None


@dataclass(frozen=True, slots=True)
class GreenwichTradingPlan:
    """Combined output of Greenwich signal generation and decision planning."""

    signal: GreenwichSpotSignal
    decision: GreenwichExecutionDecision
    diagnostics: dict[str, object]


@dataclass(frozen=True, slots=True)
class GreenwichMultiTimeframePlan:
    """Multi-timeframe Greenwich plan with D1 regime and 4H signal context."""

    signal: GreenwichMultiTimeframeSignal
    decision: GreenwichExecutionDecision
    diagnostics: dict[str, object]

