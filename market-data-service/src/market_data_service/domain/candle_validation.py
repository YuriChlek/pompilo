from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

from market_data_service.domain.candle_models import CanonicalCandle
from market_data_service.domain.timeframe_rules import get_timeframe_duration


class CandleValidationError(ValueError):
    """Raised when a candle or candle sequence is invalid."""


def validate_closed_candle(candle: CanonicalCandle, *, now: datetime, safety_delay: timedelta) -> None:
    if candle.close_time - candle.open_time != get_timeframe_duration(candle.timeframe):
        raise CandleValidationError("Candle duration does not match timeframe")
    if candle.close_time > now.astimezone(candle.close_time.tzinfo) - safety_delay:
        raise CandleValidationError("Candle is not closed beyond safety delay")


def validate_candle_sequence(candles: Sequence[CanonicalCandle]) -> None:
    if not candles:
        raise CandleValidationError("Candle sequence must not be empty")

    previous: CanonicalCandle | None = None
    seen_open_times: set[datetime] = set()
    for candle in candles:
        if candle.open_time in seen_open_times:
            raise CandleValidationError("Duplicate candle open_time")
        seen_open_times.add(candle.open_time)
        if previous is not None:
            expected_open_time = previous.open_time + get_timeframe_duration(previous.timeframe)
            if candle.open_time != expected_open_time:
                raise CandleValidationError("Candle sequence is not continuous and monotonic")
        previous = candle
