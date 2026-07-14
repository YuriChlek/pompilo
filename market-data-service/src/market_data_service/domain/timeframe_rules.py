from __future__ import annotations

from datetime import timedelta

SUPPORTED_TIMEFRAME_DURATIONS = {
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
}


def get_timeframe_duration(timeframe: str) -> timedelta:
    normalized_timeframe = timeframe.strip().lower()
    try:
        return SUPPORTED_TIMEFRAME_DURATIONS[normalized_timeframe]
    except KeyError as exc:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}") from exc
