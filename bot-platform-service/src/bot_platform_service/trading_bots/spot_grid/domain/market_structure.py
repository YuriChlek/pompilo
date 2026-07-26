from __future__ import annotations

from decimal import Decimal, localcontext

from bot_platform_service.trading_bots.spot_grid.domain.models import (
    IndicatorCandle,
    MarketStructureBias,
    MarketStructureSnapshot,
    SupportResistanceCandidate,
    SwingPoint,
)

DEFAULT_STRUCTURE_LOOKBACK = 60
DEFAULT_STRUCTURE_SWING_WINDOW = 2
MIN_STRUCTURE_CANDLES = 5
SUPPORT_RESISTANCE_CANDIDATE_LIMIT = 3


def compute_market_structure(
    candles: tuple[IndicatorCandle, ...],
    *,
    swing_window: int = DEFAULT_STRUCTURE_SWING_WINDOW,
    lookback: int = DEFAULT_STRUCTURE_LOOKBACK,
) -> MarketStructureSnapshot:
    """Compute platform-native swing, range, and support/resistance structure."""
    if not candles:
        return MarketStructureSnapshot(
            bias=MarketStructureBias.NEUTRAL,
            candle_count=0,
            has_required_history=False,
            range_low=None,
            range_high=None,
            range_position=None,
            swing_highs=(),
            swing_lows=(),
            support_candidates=(),
            resistance_candidates=(),
            breakout_direction=None,
            breakout_reference_price=None,
            reasons=("insufficient_candles",),
        )

    normalized_window = max(swing_window, 1)
    required_history = max(MIN_STRUCTURE_CANDLES, normalized_window * 2 + 1)
    recent = candles[-lookback:] if lookback > 0 else candles
    range_low = min(candle.low for candle in recent)
    range_high = max(candle.high for candle in recent)
    current_close = recent[-1].close
    range_position = _range_position(current_close, range_low, range_high)
    has_required_history = len(recent) >= required_history
    if not has_required_history:
        return MarketStructureSnapshot(
            bias=MarketStructureBias.NEUTRAL,
            candle_count=len(recent),
            has_required_history=False,
            range_low=range_low,
            range_high=range_high,
            range_position=range_position,
            swing_highs=(),
            swing_lows=(),
            support_candidates=(),
            resistance_candidates=(),
            breakout_direction=None,
            breakout_reference_price=None,
            reasons=("insufficient_candles",),
        )

    swing_highs = _find_swing_highs(recent, normalized_window)
    swing_lows = _find_swing_lows(recent, normalized_window)
    reasons: list[str] = []
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        swing_highs, swing_lows = _build_segment_extrema(recent, segments=4)
        reasons.append("segment_extrema_fallback")

    breakout_direction, breakout_reference_price = _breakout(recent, current_close)
    if breakout_direction is not None:
        reasons.append(f"{breakout_direction}_breakout")

    bias = _bias(swing_highs, swing_lows, breakout_direction)
    if not reasons:
        reasons.append("swing_structure")

    return MarketStructureSnapshot(
        bias=bias,
        candle_count=len(recent),
        has_required_history=True,
        range_low=range_low,
        range_high=range_high,
        range_position=range_position,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        support_candidates=_support_candidates(swing_lows, current_close),
        resistance_candidates=_resistance_candidates(swing_highs, current_close),
        breakout_direction=breakout_direction,
        breakout_reference_price=breakout_reference_price,
        reasons=tuple(reasons),
    )


def _find_swing_highs(candles: tuple[IndicatorCandle, ...], swing_window: int) -> tuple[SwingPoint, ...]:
    swings: list[SwingPoint] = []
    for index in range(swing_window, len(candles) - swing_window):
        price = candles[index].high
        left = candles[index - swing_window : index]
        right = candles[index + 1 : index + swing_window + 1]
        if all(price > candle.high for candle in left) and all(price >= candle.high for candle in right):
            swings.append(SwingPoint(candle_index=index, timestamp=candles[index].timestamp, price=price))
    return tuple(swings)


def _find_swing_lows(candles: tuple[IndicatorCandle, ...], swing_window: int) -> tuple[SwingPoint, ...]:
    swings: list[SwingPoint] = []
    for index in range(swing_window, len(candles) - swing_window):
        price = candles[index].low
        left = candles[index - swing_window : index]
        right = candles[index + 1 : index + swing_window + 1]
        if all(price < candle.low for candle in left) and all(price <= candle.low for candle in right):
            swings.append(SwingPoint(candle_index=index, timestamp=candles[index].timestamp, price=price))
    return tuple(swings)


def _build_segment_extrema(
    candles: tuple[IndicatorCandle, ...],
    *,
    segments: int,
) -> tuple[tuple[SwingPoint, ...], tuple[SwingPoint, ...]]:
    segment_size = max(len(candles) // max(segments, 1), 1)
    swing_highs: list[SwingPoint] = []
    swing_lows: list[SwingPoint] = []
    for start in range(0, len(candles), segment_size):
        chunk = candles[start : start + segment_size]
        if not chunk:
            continue
        high_offset, high_candle = max(enumerate(chunk), key=lambda item: item[1].high)
        low_offset, low_candle = min(enumerate(chunk), key=lambda item: item[1].low)
        high_index = start + high_offset
        low_index = start + low_offset
        swing_highs.append(
            SwingPoint(candle_index=high_index, timestamp=high_candle.timestamp, price=high_candle.high)
        )
        swing_lows.append(SwingPoint(candle_index=low_index, timestamp=low_candle.timestamp, price=low_candle.low))
        if len(swing_highs) >= segments and len(swing_lows) >= segments:
            break
    return tuple(swing_highs), tuple(swing_lows)


def _support_candidates(
    swing_lows: tuple[SwingPoint, ...],
    current_close: Decimal,
) -> tuple[SupportResistanceCandidate, ...]:
    candidates = [
        SupportResistanceCandidate(
            price=swing.price,
            source="swing_low",
            candle_index=swing.candle_index,
            distance_from_close=current_close - swing.price,
        )
        for swing in swing_lows
        if swing.price <= current_close
    ]
    return tuple(
        sorted(candidates, key=lambda candidate: (candidate.distance_from_close, -candidate.candle_index))[
            :SUPPORT_RESISTANCE_CANDIDATE_LIMIT
        ]
    )


def _resistance_candidates(
    swing_highs: tuple[SwingPoint, ...],
    current_close: Decimal,
) -> tuple[SupportResistanceCandidate, ...]:
    candidates = [
        SupportResistanceCandidate(
            price=swing.price,
            source="swing_high",
            candle_index=swing.candle_index,
            distance_from_close=swing.price - current_close,
        )
        for swing in swing_highs
        if swing.price >= current_close
    ]
    return tuple(
        sorted(candidates, key=lambda candidate: (candidate.distance_from_close, -candidate.candle_index))[
            :SUPPORT_RESISTANCE_CANDIDATE_LIMIT
        ]
    )


def _range_position(current_close: Decimal, range_low: Decimal, range_high: Decimal) -> Decimal | None:
    if range_high <= range_low:
        return None
    with localcontext() as context:
        context.prec = 34
        return +((current_close - range_low) / (range_high - range_low))


def _breakout(candles: tuple[IndicatorCandle, ...], current_close: Decimal) -> tuple[str | None, Decimal | None]:
    if len(candles) < 2:
        return None, None
    previous = candles[:-1]
    previous_high = max(candle.high for candle in previous)
    previous_low = min(candle.low for candle in previous)
    if current_close > previous_high:
        return "up", previous_high
    if current_close < previous_low:
        return "down", previous_low
    return None, None


def _bias(
    swing_highs: tuple[SwingPoint, ...],
    swing_lows: tuple[SwingPoint, ...],
    breakout_direction: str | None,
) -> MarketStructureBias:
    if breakout_direction == "up":
        return MarketStructureBias.BULLISH
    if breakout_direction == "down":
        return MarketStructureBias.BEARISH

    higher_highs = _count_directional_swings(swing_highs, rising=True)
    lower_highs = _count_directional_swings(swing_highs, rising=False)
    higher_lows = _count_directional_swings(swing_lows, rising=True)
    lower_lows = _count_directional_swings(swing_lows, rising=False)
    bullish_score = higher_highs + higher_lows
    bearish_score = lower_highs + lower_lows
    if bullish_score >= 2 and bearish_score == 0:
        return MarketStructureBias.BULLISH
    if bearish_score >= 2 and bullish_score == 0:
        return MarketStructureBias.BEARISH
    if bullish_score > 0 and bearish_score > 0:
        return MarketStructureBias.MIXED
    return MarketStructureBias.RANGE


def _count_directional_swings(swings: tuple[SwingPoint, ...], *, rising: bool) -> int:
    count = 0
    for previous, current in zip(swings, swings[1:]):
        if rising and current.price > previous.price:
            count += 1
        if not rising and current.price < previous.price:
            count += 1
    return count


__all__ = [
    "DEFAULT_STRUCTURE_LOOKBACK",
    "DEFAULT_STRUCTURE_SWING_WINDOW",
    "MIN_STRUCTURE_CANDLES",
    "SUPPORT_RESISTANCE_CANDIDATE_LIMIT",
    "compute_market_structure",
]
