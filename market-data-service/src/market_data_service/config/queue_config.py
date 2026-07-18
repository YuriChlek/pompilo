from __future__ import annotations

from dataclasses import dataclass
import os

DEFAULT_REDIS_EVENT_RETENTION_DAYS = 2
REDIS_EVENTS_PER_DAY_THROUGHPUT_ASSUMPTION = 10000
REDIS_STREAM_MAXLEN_SAFETY_FACTOR = 5


@dataclass(frozen=True, slots=True)
class RedisStreamBrokerConfig:
    redis_url: str
    stream_name: str
    maxlen: int | None


def get_redis_stream_broker_config() -> RedisStreamBrokerConfig:
    raw_maxlen = os.getenv("MARKET_DATA_OUTBOX_STREAM_MAXLEN", "").strip()
    raw_retention_days = os.getenv("MARKET_DATA_REDIS_EVENT_RETENTION_DAYS", "").strip()
    maxlen = _parse_positive_int(raw_maxlen, "MARKET_DATA_OUTBOX_STREAM_MAXLEN") if raw_maxlen else None
    if maxlen is None:
        retention_days = (
            _parse_positive_int(raw_retention_days, "MARKET_DATA_REDIS_EVENT_RETENTION_DAYS")
            if raw_retention_days
            else DEFAULT_REDIS_EVENT_RETENTION_DAYS
        )
        maxlen = calculate_redis_stream_maxlen(retention_days)

    return RedisStreamBrokerConfig(
        redis_url=os.getenv("MARKET_DATA_REDIS_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0")),
        stream_name=os.getenv("MARKET_DATA_OUTBOX_STREAM", "market-data-events").strip(),
        maxlen=maxlen,
    )


def calculate_redis_stream_maxlen(retention_days: int) -> int:
    if retention_days <= 0:
        raise ValueError("retention_days must be positive")
    return retention_days * REDIS_EVENTS_PER_DAY_THROUGHPUT_ASSUMPTION * REDIS_STREAM_MAXLEN_SAFETY_FACTOR


def _parse_positive_int(raw_value: str, name: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
