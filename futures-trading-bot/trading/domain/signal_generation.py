from __future__ import annotations

import logging
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional

from trading.domain.models import MarketRegime, SignalContext, TradeSignal
from trading.domain.signal_common import _as_decimal, _build_position_payload, calculate_position_size
from trading.domain.strategy_config import DEFAULT_STRATEGY_CONFIG, StrategyConfig
from trading.domain.execution import resolve_symbol_cluster
from utils.config import BUY_DIRECTION, SELL_DIRECTION

if TYPE_CHECKING:
    from indicators import TrendResult


logger = logging.getLogger(__name__)


class SignalGenerationError(RuntimeError):
    """Raised when strategy signal generation fails for a symbol."""

    def __init__(self, symbol: str, message: str) -> None:
        self.symbol = symbol
        super().__init__(message)


def check_candle_type(candle):
    """Classify candle close location inside its range as bullish, bearish, or neutral."""
    high = Decimal(candle["high"])
    low = Decimal(candle["low"])
    close = Decimal(candle["close"])

    if high == low:
        return "neutral"

    position = (close - low) / (high - low)
    if position > Decimal("0.50"):
        return "bullish"
    if position < Decimal("0.50"):
        return "bearish"
    return "neutral"


def get_gmma_ma_value(trend_data: "TrendResult", ma_period="ema_30") -> float:
    """Return a requested long GMMA EMA value when it is available in the trend snapshot."""
    if trend_data.gmma_analysis and "long_emas" in trend_data.gmma_analysis:
        long_emas = trend_data.gmma_analysis["long_emas"]
        return long_emas.get(ma_period)
    return None


def _history_rows(indicators_history: Any) -> list[dict[str, Any]]:
    """Return indicator-history rows as a normalized list of dictionaries."""
    if indicators_history is None:
        return []

    if hasattr(indicators_history, "to_dict"):
        try:
            return list(indicators_history.to_dict("records"))
        except TypeError:
            pass

    if hasattr(indicators_history, "iterrows"):
        return [dict(row) for _, row in indicators_history.iterrows()]

    if isinstance(indicators_history, Iterable):
        return [dict(row) if isinstance(row, dict) else row for row in indicators_history]

    return []


def _excluding_current_candle(history_rows: list[dict[str, Any]], current_candle: Dict[str, Any]) -> list[dict[str, Any]]:
    """Drop the current candle from history rows when the caller already passes it separately."""
    if not history_rows:
        return []
    latest_row = history_rows[-1]
    row_close_time = latest_row.get("close_time") or latest_row.get("open_time")
    candle_close_time = current_candle.get("close_time") or current_candle.get("open_time")
    if row_close_time is not None and candle_close_time is not None and str(row_close_time) == str(candle_close_time):
        return history_rows[:-1]
    return history_rows


def _has_min_trend_strength(trend_strength: str, minimum: str) -> bool:
    """Check whether the current trend-strength label meets the configured minimum."""
    strength_order = {"neutral": 0, "weak": 1, "medium": 2, "strong": 3}
    return strength_order.get(str(trend_strength), 0) >= strength_order.get(str(minimum), 0)


def detect_market_regime(trend_data, strategy_config: StrategyConfig) -> MarketRegime:
    """Classify the current market regime for breakout filtering."""
    h1_range = trend_data.range_analysis_h1
    h4_range = trend_data.range_analysis_h4
    atr_pct = _as_decimal(getattr(h1_range, "atr_pct", None))
    is_high_vol = atr_pct > strategy_config.regime.max_atr_pct if atr_pct > 0 else False

    range_threshold = strategy_config.regime.range_confidence_threshold
    if (
        getattr(h4_range, "is_range", False)
        and _as_decimal(getattr(h4_range, "confidence", None)) >= range_threshold
    ):
        return MarketRegime("range", "neutral", False, reason="range_market")

    gmma = trend_data.gmma_analysis or {}
    trend_strength = str(gmma.get("trend_strength", "neutral"))
    if not _has_min_trend_strength(trend_strength, strategy_config.regime.min_trend_strength):
        return MarketRegime("neutral", "neutral", False, reason="trend_strength_below_threshold")

    h1_signal = str(getattr(trend_data, "super_trend_signal", "neutral"))
    h4_signal = str(getattr(trend_data, "super_trend_h4_signal", "neutral"))
    d1_signal = str(getattr(trend_data, "super_trend_d1_signal", "neutral"))
    gmma_trend = str(gmma.get("trend", "neutral"))

    bullish = h1_signal == "bullish" and gmma_trend == "bullish"
    bearish = h1_signal == "bearish" and gmma_trend == "bearish"

    if bullish:
        if strategy_config.regime.require_h4_alignment and h4_signal == "bearish":
            return MarketRegime("neutral", "neutral", False, reason="h4_not_aligned")
        if strategy_config.regime.require_d1_alignment and d1_signal == "bearish":
            return MarketRegime("neutral", "neutral", False, reason="d1_not_aligned")
        if (
            getattr(h1_range, "is_range", False)
            and _as_decimal(getattr(h1_range, "confidence", None)) >= range_threshold
            and not (
                strategy_config.breakout.allow_h1_range_in_strong_h4_trend
                and trend_strength == "strong"
                and h4_signal == "bullish"
            )
        ):
            return MarketRegime("range", "neutral", False, is_high_vol=is_high_vol, reason="range_market")
        return MarketRegime("bull_trend", BUY_DIRECTION, True, is_high_vol=is_high_vol)

    if bearish:
        if strategy_config.regime.require_h4_alignment and h4_signal == "bullish":
            return MarketRegime("neutral", "neutral", False, reason="h4_not_aligned")
        if strategy_config.regime.require_d1_alignment and d1_signal == "bullish":
            return MarketRegime("neutral", "neutral", False, reason="d1_not_aligned")
        if (
            getattr(h1_range, "is_range", False)
            and _as_decimal(getattr(h1_range, "confidence", None)) >= range_threshold
            and not (
                strategy_config.breakout.allow_h1_range_in_strong_h4_trend
                and trend_strength == "strong"
                and h4_signal == "bearish"
            )
        ):
            return MarketRegime("range", "neutral", False, is_high_vol=is_high_vol, reason="range_market")
        return MarketRegime("bear_trend", SELL_DIRECTION, True, is_high_vol=is_high_vol)

    return MarketRegime("neutral", "neutral", False, reason="trend_not_aligned")


def _funding_allows_entry(direction: str, trend_data, strategy_config: StrategyConfig) -> bool:
    """Check whether the optional funding-rate filter allows the breakout entry."""
    if not strategy_config.funding.enabled:
        return True

    funding_rate = getattr(trend_data, "funding_rate", None)
    funding_decimal = _as_decimal(funding_rate, default=Decimal("0"))
    if direction == BUY_DIRECTION:
        return funding_decimal <= strategy_config.funding.long_funding_cap
    return funding_decimal >= strategy_config.funding.short_funding_floor


def _resolve_trend_strength(trend_data) -> str:
    """Return the current GMMA trend-strength label."""
    gmma = trend_data.gmma_analysis or {}
    return str(gmma.get("trend_strength", "neutral"))


def _required_volume_ratio(trend_strength: str, strategy_config: StrategyConfig) -> Decimal:
    """Return the current minimum volume confirmation threshold for entry.

    Stronger trends are allowed to confirm breakouts with lighter volume expansion.
    """
    if trend_strength == "strong":
        return strategy_config.breakout.strong_trend_volume_ratio
    if trend_strength == "medium":
        return strategy_config.breakout.medium_trend_volume_ratio
    return strategy_config.breakout.min_volume_spike_ratio


def _build_breakout_buffer(candle_close: Decimal, atr_value: Decimal, strategy_config: StrategyConfig) -> Decimal:
    """Return the current breakout confirmation buffer."""
    return max(
        candle_close * strategy_config.breakout.min_breakout_close_pct,
        atr_value * strategy_config.breakout.atr_breakout_buffer_fraction,
    )


def _is_h1_range_soft_allowed(trend_data, regime: MarketRegime, strategy_config: StrategyConfig) -> bool:
    """Return whether an H1 range state may still be allowed for entry."""
    if not strategy_config.breakout.allow_h1_range_in_strong_h4_trend:
        return False

    h1_range = getattr(trend_data, "range_analysis_h1", None)
    range_threshold = strategy_config.regime.range_confidence_threshold
    if not (
        getattr(h1_range, "is_range", False)
        and _as_decimal(getattr(h1_range, "confidence", None)) >= range_threshold
    ):
        return False

    trend_strength = _resolve_trend_strength(trend_data)
    if trend_strength != "strong":
        return False

    h4_signal = str(getattr(trend_data, "super_trend_h4_signal", "neutral"))
    if regime.direction == BUY_DIRECTION:
        return h4_signal == "bullish"
    if regime.direction == SELL_DIRECTION:
        return h4_signal == "bearish"
    return False


def _resolve_entry_risk_pct(regime: MarketRegime, strategy_config: StrategyConfig) -> Decimal:
    """Return the risk percent used for position sizing."""
    risk_pct = strategy_config.risk.primary_risk_pct
    if regime.is_high_vol:
        return risk_pct * strategy_config.breakout.high_vol_risk_multiplier
    return risk_pct


def _build_breakout_context(
    *,
    trend_data,
    indicators_history,
    strategy_config: StrategyConfig,
) -> Optional[dict[str, Any]]:
    """Build the shared breakout context used by all entry setups."""
    regime = detect_market_regime(trend_data, strategy_config)
    if not regime.is_breakout_enabled:
        return None

    candle = trend_data.candle
    candle_close = Decimal(str(candle["close"]))
    candle_high = Decimal(str(candle["high"]))
    candle_low = Decimal(str(candle["low"]))
    candle_open = Decimal(str(candle["open"]))
    atr_value = Decimal(str(trend_data.atr))
    if atr_value <= 0:
        return None

    candle_range = candle_high - candle_low
    if candle_range > atr_value * strategy_config.breakout.max_breakout_candle_atr:
        return None

    history_rows = _excluding_current_candle(_history_rows(indicators_history), candle)
    if len(history_rows) < strategy_config.breakout.lookback_candles:
        return None
    recent_rows = history_rows[-strategy_config.breakout.lookback_candles :]
    breakout_high = max(_as_decimal(row.get("high")) for row in recent_rows)
    breakout_low = min(_as_decimal(row.get("low")) for row in recent_rows)

    trend_strength = _resolve_trend_strength(trend_data)
    if trend_strength == "weak" and not strategy_config.breakout.weak_trend_allows_entry:
        return None

    h1_range = getattr(trend_data, "range_analysis_h1", None)
    range_threshold = strategy_config.regime.range_confidence_threshold
    if (
        getattr(h1_range, "is_range", False)
        and _as_decimal(getattr(h1_range, "confidence", None)) >= range_threshold
        and not _is_h1_range_soft_allowed(trend_data, regime, strategy_config)
    ):
        return None

    volume_analysis = trend_data.volume_analysis or {}
    spike_ratio = _as_decimal(volume_analysis.get("spike_ratio"), default=Decimal("0"))
    if spike_ratio < _required_volume_ratio(trend_strength, strategy_config):
        return None

    context = SignalContext(
        regime=regime,
        breakout_high=breakout_high,
        breakout_low=breakout_low,
        volume_spike_ratio=spike_ratio,
        funding_rate=_as_decimal(getattr(trend_data, "funding_rate", None), default=Decimal("0")),
    )

    return {
        "regime": regime,
        "context": context,
        "candle": candle,
        "candle_close": candle_close,
        "candle_high": candle_high,
        "candle_low": candle_low,
        "candle_open": candle_open,
        "atr_value": atr_value,
        "recent_rows": recent_rows,
        "trend_strength": trend_strength,
        "breakout_buffer": _build_breakout_buffer(candle_close, atr_value, strategy_config),
        "effective_risk_pct": _resolve_entry_risk_pct(regime, strategy_config),
    }


def _build_breakout_close_signal(
    *,
    symbol: str,
    breakout_context: dict[str, Any],
    trend_data,
    strategy_config: StrategyConfig,
) -> Optional[TradeSignal]:
    """Build the current candle-close breakout signal."""
    regime = breakout_context["regime"]
    context = breakout_context["context"]
    candle = breakout_context["candle"]
    candle_close = breakout_context["candle_close"]
    candle_open = breakout_context["candle_open"]
    atr_value = breakout_context["atr_value"]
    recent_rows = breakout_context["recent_rows"]
    breakout_buffer = breakout_context["breakout_buffer"]
    effective_risk_pct = breakout_context["effective_risk_pct"]

    if regime.direction == BUY_DIRECTION:
        if candle_close <= context.breakout_high + breakout_buffer or candle_close <= candle_open:
            return None
        if not _funding_allows_entry(BUY_DIRECTION, trend_data, strategy_config):
            return None
        swing_low = min(_as_decimal(row.get("low")) for row in recent_rows[-5:])
        stop_loss = min(swing_low, context.breakout_high - (atr_value * strategy_config.breakout.stop_atr_multiplier))
        if stop_loss >= candle_close:
            return None
        risk_distance = candle_close - stop_loss
        take_profit = candle_close + (risk_distance * strategy_config.breakout.take_profit_r)
        size = calculate_position_size(symbol, effective_risk_pct, candle_close, stop_loss)
        return _build_position_payload(
            symbol=symbol,
            order_type=strategy_config.breakout.order_type,
            direction=BUY_DIRECTION,
            price=candle_close,
            size=size,
            take_profit=take_profit,
            stop_loss=stop_loss,
            timestamp=candle.get("close_time") or trend_data.timestamp,
            strategy_mode="trend_breakout",
            metadata={
                "setup_type": "breakout_close",
                "regime": context.regime.name,
                "breakout_level": context.breakout_high,
                "volume_spike_ratio": context.volume_spike_ratio,
                "risk_distance": risk_distance,
                "effective_risk_pct": effective_risk_pct,
                "cluster": resolve_symbol_cluster(symbol, strategy_config.portfolio),
            },
        )

    if candle_close >= context.breakout_low - breakout_buffer or candle_close >= candle_open:
        return None
    if not _funding_allows_entry(SELL_DIRECTION, trend_data, strategy_config):
        return None
    swing_high = max(_as_decimal(row.get("high")) for row in recent_rows[-5:])
    stop_loss = max(swing_high, context.breakout_low + (atr_value * strategy_config.breakout.stop_atr_multiplier))
    if stop_loss <= candle_close:
        return None
    risk_distance = stop_loss - candle_close
    take_profit = candle_close - (risk_distance * strategy_config.breakout.take_profit_r)
    size = calculate_position_size(symbol, effective_risk_pct, candle_close, stop_loss)
    return _build_position_payload(
        symbol=symbol,
        order_type=strategy_config.breakout.order_type,
        direction=SELL_DIRECTION,
        price=candle_close,
        size=size,
        take_profit=take_profit,
        stop_loss=stop_loss,
        timestamp=candle.get("close_time") or trend_data.timestamp,
        strategy_mode="trend_breakout",
        metadata={
            "setup_type": "breakout_close",
            "regime": context.regime.name,
            "breakout_level": context.breakout_low,
            "volume_spike_ratio": context.volume_spike_ratio,
            "risk_distance": risk_distance,
            "effective_risk_pct": effective_risk_pct,
            "cluster": resolve_symbol_cluster(symbol, strategy_config.portfolio),
        },
    )


def _build_breakout_reclaim_signal(
    *,
    symbol: str,
    breakout_context: dict[str, Any],
    trend_data,
    strategy_config: StrategyConfig,
) -> Optional[TradeSignal]:
    """Build a continuation entry after a prior breakout candle reclaims the broken level."""
    if not strategy_config.breakout.reclaim_enabled:
        return None

    regime = breakout_context["regime"]
    candle = breakout_context["candle"]
    candle_close = breakout_context["candle_close"]
    candle_high = breakout_context["candle_high"]
    candle_low = breakout_context["candle_low"]
    candle_open = breakout_context["candle_open"]
    atr_value = breakout_context["atr_value"]
    recent_rows = breakout_context["recent_rows"]
    effective_risk_pct = breakout_context["effective_risk_pct"]
    current_candle_range = candle_high - candle_low

    if len(recent_rows) < 2:
        return None
    if current_candle_range > atr_value * strategy_config.breakout.reclaim_max_candle_atr:
        return None

    previous_candle = recent_rows[-1]
    prior_rows = recent_rows[:-1]
    if not prior_rows:
        return None

    previous_open = _as_decimal(previous_candle.get("open"))
    previous_close = _as_decimal(previous_candle.get("close"))
    previous_high = _as_decimal(previous_candle.get("high"))
    previous_low = _as_decimal(previous_candle.get("low"))
    reclaim_tolerance = atr_value * strategy_config.breakout.reclaim_tolerance_atr_fraction
    previous_breakout_buffer = _build_breakout_buffer(previous_close, atr_value, strategy_config)

    if regime.direction == BUY_DIRECTION:
        breakout_level = max(_as_decimal(row.get("high")) for row in prior_rows)
        breakout_happened = (
            previous_close > breakout_level + previous_breakout_buffer
            and previous_close > previous_open
            and previous_high >= breakout_level
        )
        if not breakout_happened:
            return None
        if candle_low > breakout_level + reclaim_tolerance:
            return None
        if candle_close <= breakout_level:
            return None
        if strategy_config.breakout.require_reclaim_close_in_breakout_direction and candle_close <= candle_open:
            return None
        if not _funding_allows_entry(BUY_DIRECTION, trend_data, strategy_config):
            return None

        stop_loss = min(candle_low, breakout_level - (atr_value * strategy_config.breakout.reclaim_stop_atr_multiplier))
        if stop_loss >= candle_close:
            return None
        risk_distance = candle_close - stop_loss
        take_profit = candle_close + (risk_distance * strategy_config.breakout.take_profit_r)
        size = calculate_position_size(symbol, effective_risk_pct, candle_close, stop_loss)
        return _build_position_payload(
            symbol=symbol,
            order_type=strategy_config.breakout.order_type,
            direction=BUY_DIRECTION,
            price=candle_close,
            size=size,
            take_profit=take_profit,
            stop_loss=stop_loss,
            timestamp=candle.get("close_time") or trend_data.timestamp,
            strategy_mode="trend_breakout",
            metadata={
                "setup_type": "breakout_reclaim",
                "regime": regime.name,
                "breakout_level": breakout_level,
                "volume_spike_ratio": breakout_context["context"].volume_spike_ratio,
                "risk_distance": risk_distance,
                "effective_risk_pct": effective_risk_pct,
                "cluster": resolve_symbol_cluster(symbol, strategy_config.portfolio),
            },
        )

    breakout_level = min(_as_decimal(row.get("low")) for row in prior_rows)
    breakout_happened = (
        previous_close < breakout_level - previous_breakout_buffer
        and previous_close < previous_open
        and previous_low <= breakout_level
    )
    if not breakout_happened:
        return None
    if candle_high < breakout_level - reclaim_tolerance:
        return None
    if candle_close >= breakout_level:
        return None
    if strategy_config.breakout.require_reclaim_close_in_breakout_direction and candle_close >= candle_open:
        return None
    if not _funding_allows_entry(SELL_DIRECTION, trend_data, strategy_config):
        return None

    stop_loss = max(candle_high, breakout_level + (atr_value * strategy_config.breakout.reclaim_stop_atr_multiplier))
    if stop_loss <= candle_close:
        return None
    risk_distance = stop_loss - candle_close
    take_profit = candle_close - (risk_distance * strategy_config.breakout.take_profit_r)
    size = calculate_position_size(symbol, effective_risk_pct, candle_close, stop_loss)
    return _build_position_payload(
        symbol=symbol,
        order_type=strategy_config.breakout.order_type,
        direction=SELL_DIRECTION,
        price=candle_close,
        size=size,
        take_profit=take_profit,
        stop_loss=stop_loss,
        timestamp=candle.get("close_time") or trend_data.timestamp,
        strategy_mode="trend_breakout",
        metadata={
            "setup_type": "breakout_reclaim",
            "regime": regime.name,
            "breakout_level": breakout_level,
            "volume_spike_ratio": breakout_context["context"].volume_spike_ratio,
            "risk_distance": risk_distance,
            "effective_risk_pct": effective_risk_pct,
            "cluster": resolve_symbol_cluster(symbol, strategy_config.portfolio),
        },
    )


def _entry_setup_builders():
    """Return setup builders in deterministic priority order."""
    return (
        _build_breakout_close_signal,
        _build_breakout_reclaim_signal,
    )


def _build_breakout_signal(
    *,
    symbol: str,
    trend_data,
    indicators_history,
    strategy_config: StrategyConfig,
) -> Optional[TradeSignal]:
    """Build the active breakout signal using the current entry-pipeline helpers."""
    if not strategy_config.breakout.enabled:
        return None

    breakout_context = _build_breakout_context(
        trend_data=trend_data,
        indicators_history=indicators_history,
        strategy_config=strategy_config,
    )
    if not breakout_context:
        return None

    for setup_builder in _entry_setup_builders():
        signal = setup_builder(
            symbol=symbol,
            breakout_context=breakout_context,
            trend_data=trend_data,
            strategy_config=strategy_config,
        )
        if signal:
            return signal
    return None


async def generate_strategy_signal(
    symbol,
    trend_data,
    indicators_history,
    is_test,
    strategy_config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
):
    """Build a signal using only the active trend-breakout strategy."""
    try:
        return _build_breakout_signal(
            symbol=symbol,
            trend_data=trend_data,
            indicators_history=indicators_history,
            strategy_config=strategy_config,
        )
    except Exception as exc:
        logger.exception("signal_generation_failed symbol=%s", symbol)
        raise SignalGenerationError(str(symbol), f"Failed to generate strategy signal for {symbol}") from exc
