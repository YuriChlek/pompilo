from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, time, timedelta
from random import Random

from market_data_service.domain.enums import MarketDataSource
from market_data_service.domain.scheduler_models import MarketDataSyncJob
from market_data_service.domain.timeframe_rules import get_timeframe_duration

SUPPORTED_SCHEDULE_TIMEFRAMES = ("1h", "4h", "1d")


def latest_closed_candle_time(now: datetime, timeframe: str) -> datetime:
    normalized_timeframe = timeframe.strip().lower()
    current = now.astimezone(UTC)
    if normalized_timeframe == "1h":
        return current.replace(minute=0, second=0, microsecond=0)
    if normalized_timeframe == "4h":
        hour = current.hour - (current.hour % 4)
        return current.replace(hour=hour, minute=0, second=0, microsecond=0)
    if normalized_timeframe == "1d":
        return datetime.combine(current.date(), time.min, tzinfo=UTC)
    get_timeframe_duration(normalized_timeframe)
    raise ValueError(f"Unsupported timeframe: {timeframe!r}")


def due_closed_candle_time(
    *,
    now: datetime,
    timeframe: str,
    safety_delay: timedelta,
) -> datetime | None:
    close_time = latest_closed_candle_time(now, timeframe)
    if now.astimezone(UTC) < close_time + safety_delay:
        close_time -= get_timeframe_duration(timeframe)
    if now.astimezone(UTC) < close_time + safety_delay:
        return None
    return close_time


def build_sync_jobs(
    *,
    source: MarketDataSource,
    provider_symbols: Iterable[str],
    timeframes: Iterable[str],
    now: datetime,
    safety_delay_by_timeframe: Mapping[str, timedelta],
    jitter_seconds: int,
    random_seed: int | None = None,
) -> list[MarketDataSyncJob]:
    rng = Random(random_seed)
    jobs: list[MarketDataSyncJob] = []
    for timeframe in timeframes:
        normalized_timeframe = timeframe.strip().lower()
        close_time = due_closed_candle_time(
            now=now,
            timeframe=normalized_timeframe,
            safety_delay=safety_delay_by_timeframe[normalized_timeframe],
        )
        if close_time is None:
            continue
        for provider_symbol in provider_symbols:
            normalized_symbol = provider_symbol.strip().upper()
            jitter = timedelta(seconds=rng.randint(0, jitter_seconds)) if jitter_seconds > 0 else timedelta(0)
            jobs.append(
                MarketDataSyncJob(
                    source=source,
                    provider_symbol=normalized_symbol,
                    timeframe=normalized_timeframe,
                    expected_close_time=close_time,
                    scheduled_for=close_time + safety_delay_by_timeframe[normalized_timeframe] + jitter,
                    idempotency_key=build_sync_job_idempotency_key(
                        source=source,
                        provider_symbol=normalized_symbol,
                        timeframe=normalized_timeframe,
                        expected_close_time=close_time,
                    ),
                )
            )
    return jobs


def build_sync_job_idempotency_key(
    *,
    source: MarketDataSource,
    provider_symbol: str,
    timeframe: str,
    expected_close_time: datetime,
) -> str:
    return "|".join(
        (
            source.value,
            provider_symbol.strip().upper(),
            timeframe.strip().lower(),
            expected_close_time.astimezone(UTC).isoformat(),
        )
    )
