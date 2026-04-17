from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from domain.models import Candle


class StructureBias(str, Enum):
    """Directional classification for recent swing structure."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    RANGE = "RANGE"
    MIXED = "MIXED"
    NEUTRAL = "NEUTRAL"


@dataclass(slots=True, frozen=True)
class SwingPoint:
    """Single swing high or low used for structure analysis."""

    index: int
    price: float


@dataclass(slots=True, frozen=True)
class StructureSnapshot:
    """Summary of recent price structure extracted from swing highs and lows."""

    bias: StructureBias
    confidence: float
    swing_highs: list[SwingPoint] = field(default_factory=list)
    swing_lows: list[SwingPoint] = field(default_factory=list)
    higher_highs: int = 0
    higher_lows: int = 0
    lower_highs: int = 0
    lower_lows: int = 0
    reasons: list[str] = field(default_factory=list)


def detect_market_structure(
    candles: list[Candle],
    *,
    swing_window: int = 2,
    lookback: int = 30,
) -> StructureSnapshot:
    """Classify recent price structure from swing highs, lows, and fallback extrema."""
    if len(candles) < max(5, swing_window * 2 + 1):
        return StructureSnapshot(bias=StructureBias.NEUTRAL, confidence=0.0, reasons=["insufficient_candles"])

    recent = candles[-lookback:] if lookback > 0 else candles
    swing_highs = _find_swing_highs(recent, swing_window)
    swing_lows = _find_swing_lows(recent, swing_window)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        swing_highs, swing_lows = _build_segment_extrema(recent, segments=4)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return StructureSnapshot(
            bias=StructureBias.NEUTRAL,
            confidence=0.2,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            reasons=["insufficient_swings"],
        )

    higher_highs = _count_directional_swings(swing_highs, bullish=True)
    lower_highs = _count_directional_swings(swing_highs, bullish=False)
    higher_lows = _count_directional_swings(swing_lows, bullish=True)
    lower_lows = _count_directional_swings(swing_lows, bullish=False)

    reasons: list[str] = []
    if higher_highs > 0:
        reasons.append("higher_highs")
    if higher_lows > 0:
        reasons.append("higher_lows")
    if lower_highs > 0:
        reasons.append("lower_highs")
    if lower_lows > 0:
        reasons.append("lower_lows")

    bullish_score = higher_highs + higher_lows
    bearish_score = lower_highs + lower_lows
    total_signals = max(bullish_score + bearish_score, 1)

    if bullish_score >= 2 and bearish_score == 0:
        return StructureSnapshot(
            bias=StructureBias.BULLISH,
            confidence=min(0.95, bullish_score / total_signals),
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            higher_highs=higher_highs,
            higher_lows=higher_lows,
            lower_highs=lower_highs,
            lower_lows=lower_lows,
            reasons=reasons,
        )

    if bearish_score >= 2 and bullish_score == 0:
        return StructureSnapshot(
            bias=StructureBias.BEARISH,
            confidence=min(0.95, bearish_score / total_signals),
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            higher_highs=higher_highs,
            higher_lows=higher_lows,
            lower_highs=lower_highs,
            lower_lows=lower_lows,
            reasons=reasons,
        )

    if bullish_score > 0 and bearish_score > 0:
        return StructureSnapshot(
            bias=StructureBias.MIXED,
            confidence=0.45,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            higher_highs=higher_highs,
            higher_lows=higher_lows,
            lower_highs=lower_highs,
            lower_lows=lower_lows,
            reasons=reasons or ["mixed_structure"],
        )

    return StructureSnapshot(
        bias=StructureBias.RANGE,
        confidence=0.55,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        higher_highs=higher_highs,
        higher_lows=higher_lows,
        lower_highs=lower_highs,
        lower_lows=lower_lows,
        reasons=reasons or ["range_like_structure"],
    )


def _find_swing_highs(candles: list[Candle], swing_window: int) -> list[SwingPoint]:
    """Return local swing highs using a symmetric lookback window."""
    swings: list[SwingPoint] = []
    for index in range(swing_window, len(candles) - swing_window):
        price = candles[index].high
        left = candles[index - swing_window : index]
        right = candles[index + 1 : index + swing_window + 1]
        if all(price > candle.high for candle in left) and all(price >= candle.high for candle in right):
            swings.append(SwingPoint(index=index, price=price))
    return swings


def _find_swing_lows(candles: list[Candle], swing_window: int) -> list[SwingPoint]:
    """Return local swing lows using a symmetric lookback window."""
    swings: list[SwingPoint] = []
    for index in range(swing_window, len(candles) - swing_window):
        price = candles[index].low
        left = candles[index - swing_window : index]
        right = candles[index + 1 : index + swing_window + 1]
        if all(price < candle.low for candle in left) and all(price <= candle.low for candle in right):
            swings.append(SwingPoint(index=index, price=price))
    return swings


def _count_directional_swings(swings: list[SwingPoint], *, bullish: bool) -> int:
    """Count monotonic swing advances or declines across consecutive points."""
    count = 0
    for previous, current in zip(swings, swings[1:]):
        if bullish and current.price > previous.price:
            count += 1
        if not bullish and current.price < previous.price:
            count += 1
    return count


def _build_segment_extrema(candles: list[Candle], *, segments: int) -> tuple[list[SwingPoint], list[SwingPoint]]:
    """Approximate swings by taking extrema from fixed candle segments when pivots are sparse."""
    if len(candles) < max(segments, 2):
        return [], []
    segment_size = max(len(candles) // segments, 1)
    swing_highs: list[SwingPoint] = []
    swing_lows: list[SwingPoint] = []
    for start in range(0, len(candles), segment_size):
        chunk = candles[start : start + segment_size]
        if not chunk:
            continue
        high_offset, high_candle = max(enumerate(chunk), key=lambda item: item[1].high)
        low_offset, low_candle = min(enumerate(chunk), key=lambda item: item[1].low)
        swing_highs.append(SwingPoint(index=start + high_offset, price=high_candle.high))
        swing_lows.append(SwingPoint(index=start + low_offset, price=low_candle.low))
        if len(swing_highs) >= segments and len(swing_lows) >= segments:
            break
    return swing_highs, swing_lows
