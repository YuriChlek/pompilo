from __future__ import annotations

from bot_platform_service.domain.exceptions import UnsupportedTimeframeError

CANONICAL_TIMEFRAMES = frozenset({"1h", "4h", "1d"})

_TIMEFRAME_ALIASES = {
    "1h": "1h",
    "h1": "1h",
    "1H": "1h",
    "H1": "1h",
    "4h": "4h",
    "h4": "4h",
    "4H": "4h",
    "H4": "4h",
    "1d": "1d",
    "d1": "1d",
    "1D": "1d",
    "D1": "1d",
}


def normalize_timeframe(timeframe: str) -> str:
    """Normalize legacy timeframe aliases into canonical platform values."""
    normalized = _TIMEFRAME_ALIASES.get(timeframe.strip())
    if normalized is None:
        raise UnsupportedTimeframeError(timeframe)
    return normalized


def normalize_timeframes(timeframes: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize a tuple of timeframes while preserving order."""
    return tuple(normalize_timeframe(timeframe) for timeframe in timeframes)
