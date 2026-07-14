from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True, slots=True)
class RedisStreamBrokerConfig:
    redis_url: str
    stream_name: str
    maxlen: int | None


def get_redis_stream_broker_config() -> RedisStreamBrokerConfig:
    raw_maxlen = os.getenv("MARKET_DATA_OUTBOX_STREAM_MAXLEN", "").strip()
    return RedisStreamBrokerConfig(
        redis_url=os.getenv("MARKET_DATA_REDIS_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0")),
        stream_name=os.getenv("MARKET_DATA_OUTBOX_STREAM", "market-data-events").strip(),
        maxlen=int(raw_maxlen) if raw_maxlen else None,
    )
