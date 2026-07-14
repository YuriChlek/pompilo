from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from market_data_service.domain.candle_models import CanonicalCandle
from market_data_service.domain.enums import CandleRangeStatus
from market_data_service.domain.timeframe_rules import get_timeframe_duration


@dataclass(frozen=True, slots=True)
class MissingInterval:
    open_time: datetime
    close_time: datetime


@dataclass(frozen=True, slots=True)
class CandleRangeValidationResult:
    status: CandleRangeStatus
    expected_count: int
    actual_count: int
    gap_count: int
    duplicate_count: int
    wrong_duration_count: int
    missing_intervals: tuple[MissingInterval, ...]

    @property
    def is_complete(self) -> bool:
        return self.status == CandleRangeStatus.COMPLETE


def detect_candle_range_status(
    candles: list[CanonicalCandle],
    *,
    timeframe: str,
    expected_from: datetime,
    expected_to: datetime,
    now: datetime | None = None,
    max_allowed_lag: timedelta | None = None,
) -> CandleRangeValidationResult:
    duration = get_timeframe_duration(timeframe)
    if expected_from >= expected_to:
        raise ValueError("expected_from must be earlier than expected_to")

    expected_open_times = _build_expected_open_times(expected_from, expected_to, duration)
    candles_by_open_time: dict[datetime, CanonicalCandle] = {}
    duplicate_count = 0
    wrong_duration_count = 0

    for candle in sorted(candles, key=lambda item: item.open_time):
        if candle.timeframe != timeframe:
            wrong_duration_count += 1
            continue
        if candle.close_time - candle.open_time != duration:
            wrong_duration_count += 1
        if candle.open_time in candles_by_open_time:
            duplicate_count += 1
            continue
        candles_by_open_time[candle.open_time] = candle

    missing_intervals = tuple(
        MissingInterval(open_time=open_time, close_time=open_time + duration)
        for open_time in expected_open_times
        if open_time not in candles_by_open_time
    )

    if wrong_duration_count > 0 or duplicate_count > 0:
        status = CandleRangeStatus.GAP_DETECTED
    elif missing_intervals:
        status = _status_for_missing_intervals(missing_intervals, expected_open_times)
    elif _is_stale(candles_by_open_time, expected_open_times, now=now, max_allowed_lag=max_allowed_lag):
        status = CandleRangeStatus.STALE
    else:
        status = CandleRangeStatus.COMPLETE

    return CandleRangeValidationResult(
        status=status,
        expected_count=len(expected_open_times),
        actual_count=len(candles_by_open_time),
        gap_count=len(missing_intervals),
        duplicate_count=duplicate_count,
        wrong_duration_count=wrong_duration_count,
        missing_intervals=missing_intervals,
    )


def _build_expected_open_times(expected_from: datetime, expected_to: datetime, duration: timedelta) -> tuple[datetime, ...]:
    open_times: list[datetime] = []
    current = expected_from
    while current < expected_to:
        open_times.append(current)
        current += duration
    return tuple(open_times)


def _status_for_missing_intervals(
    missing_intervals: tuple[MissingInterval, ...],
    expected_open_times: tuple[datetime, ...],
) -> CandleRangeStatus:
    missing_open_times = {interval.open_time for interval in missing_intervals}
    if expected_open_times[-1] in missing_open_times:
        return CandleRangeStatus.INCOMPLETE
    return CandleRangeStatus.GAP_DETECTED


def _is_stale(
    candles_by_open_time: dict[datetime, CanonicalCandle],
    expected_open_times: tuple[datetime, ...],
    *,
    now: datetime | None,
    max_allowed_lag: timedelta | None,
) -> bool:
    if now is None or max_allowed_lag is None or not expected_open_times:
        return False
    latest_candle = candles_by_open_time.get(expected_open_times[-1])
    if latest_candle is None:
        return False
    return latest_candle.close_time < now.astimezone(latest_candle.close_time.tzinfo) - max_allowed_lag
