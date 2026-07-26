from __future__ import annotations

from decimal import Decimal, localcontext

from bot_platform_service.trading_bots.spot_grid.domain.models import (
    IndicatorSnapshot,
    MarketRegime,
    MarketStructureBias,
    MarketStructureSnapshot,
    RegimeSnapshot,
)

HIGH_VOLATILITY_THRESHOLD = Decimal("0.04")
ATR_RATIO_HIGH_VOLATILITY_THRESHOLD = Decimal("0.08")
DOWNTREND_RSI_THRESHOLD = Decimal("40")
UPTREND_RSI_THRESHOLD = Decimal("55")


def detect_single_timeframe_regime(
    *,
    indicators: IndicatorSnapshot,
    market_structure: MarketStructureSnapshot,
) -> RegimeSnapshot:
    """Classify the current single-timeframe Spot Grid regime deterministically."""

    diagnostics = {
        "indicator_candle_count": indicators.candle_count,
        "indicator_has_required_history": indicators.has_required_history,
        "market_structure_bias": market_structure.bias.value,
        "market_structure_has_required_history": market_structure.has_required_history,
        "ema_alignment": _ema_alignment(indicators),
        "atr_ratio": _atr_ratio(indicators),
        "rsi14": indicators.rsi14,
        "realized_volatility": indicators.realized_volatility,
        "volume_ratio": indicators.volume_ratio,
    }
    reasons: list[str] = []

    if not indicators.has_required_history:
        reasons.append("insufficient_indicator_history")
    if not market_structure.has_required_history:
        reasons.append("insufficient_market_structure_history")
    if reasons:
        return RegimeSnapshot(
            regime=MarketRegime.RANGE,
            confidence=Decimal("0.50"),
            reasons=tuple((*reasons, "range_short_history_fallback")),
            diagnostics=diagnostics,
        )

    volatility_high = _is_high_volatility(indicators)
    bearish = _is_bearish(indicators, market_structure)
    bullish = _is_bullish(indicators, market_structure)

    if _is_risk_off(indicators, market_structure, volatility_high):
        return RegimeSnapshot(
            regime=MarketRegime.RISK_OFF,
            confidence=Decimal("0.90"),
            reasons=tuple((*reasons, "risk_off_bearish_high_volatility")),
            diagnostics=diagnostics,
        )

    if volatility_high:
        return RegimeSnapshot(
            regime=MarketRegime.HIGH_VOLATILITY,
            confidence=Decimal("0.80"),
            reasons=tuple((*reasons, "high_volatility")),
            diagnostics=diagnostics,
        )

    if bearish:
        return RegimeSnapshot(
            regime=MarketRegime.DOWNTREND,
            confidence=Decimal("0.75"),
            reasons=tuple((*reasons, "bearish_single_timeframe")),
            diagnostics=diagnostics,
        )

    if bullish:
        return RegimeSnapshot(
            regime=MarketRegime.UPTREND,
            confidence=Decimal("0.70"),
            reasons=tuple((*reasons, "bullish_single_timeframe")),
            diagnostics=diagnostics,
        )

    return RegimeSnapshot(
        regime=MarketRegime.RANGE,
        confidence=Decimal("0.60"),
        reasons=tuple((*reasons, "range_single_timeframe")),
        diagnostics=diagnostics,
    )


def _is_risk_off(
    indicators: IndicatorSnapshot,
    market_structure: MarketStructureSnapshot,
    volatility_high: bool,
) -> bool:
    del indicators
    if market_structure.bias is not MarketStructureBias.BEARISH:
        return False
    return volatility_high


def _is_high_volatility(indicators: IndicatorSnapshot) -> bool:
    realized = indicators.realized_volatility
    if realized is not None and realized >= HIGH_VOLATILITY_THRESHOLD:
        return True
    atr_ratio = _atr_ratio(indicators)
    return atr_ratio is not None and atr_ratio >= ATR_RATIO_HIGH_VOLATILITY_THRESHOLD


def _is_bearish(indicators: IndicatorSnapshot, market_structure: MarketStructureSnapshot) -> bool:
    if market_structure.bias is MarketStructureBias.BEARISH:
        return True
    if _ema_alignment(indicators) == "bearish":
        return True
    return indicators.rsi14 is not None and indicators.rsi14 <= DOWNTREND_RSI_THRESHOLD


def _is_bullish(indicators: IndicatorSnapshot, market_structure: MarketStructureSnapshot) -> bool:
    if market_structure.bias is MarketStructureBias.BULLISH:
        return True
    if _ema_alignment(indicators) == "bullish":
        return True
    return indicators.rsi14 is not None and indicators.rsi14 >= UPTREND_RSI_THRESHOLD


def _ema_alignment(indicators: IndicatorSnapshot) -> str:
    if indicators.ema20 is None or indicators.ema50 is None or indicators.ema200 is None:
        return "unknown"
    if indicators.ema20 > indicators.ema50 > indicators.ema200:
        return "bullish"
    if indicators.ema20 < indicators.ema50 < indicators.ema200:
        return "bearish"
    return "mixed"


def _atr_ratio(indicators: IndicatorSnapshot) -> Decimal | None:
    if indicators.atr14 is None or indicators.ema50 is None or indicators.ema50 <= 0:
        return None
    with localcontext() as context:
        context.prec = 34
        return +(indicators.atr14 / indicators.ema50)


__all__ = [
    "ATR_RATIO_HIGH_VOLATILITY_THRESHOLD",
    "DOWNTREND_RSI_THRESHOLD",
    "HIGH_VOLATILITY_THRESHOLD",
    "UPTREND_RSI_THRESHOLD",
    "detect_single_timeframe_regime",
]
