from __future__ import annotations

import json
from typing import Any

from market_data_service.config.queue_config import RedisStreamBrokerConfig
from market_data_service.domain.outbox_models import OutboxEvent


class RedisStreamEventBroker:
    def __init__(self, *, redis_client: Any, stream_name: str, maxlen: int | None = None) -> None:
        self.redis_client = redis_client
        self.stream_name = stream_name
        self.maxlen = maxlen

    @classmethod
    def from_url(cls, config: RedisStreamBrokerConfig) -> "RedisStreamEventBroker":
        from redis.asyncio import Redis

        return cls(
            redis_client=Redis.from_url(config.redis_url, decode_responses=True),
            stream_name=config.stream_name,
            maxlen=config.maxlen,
        )

    async def publish(self, event: OutboxEvent) -> None:
        fields = {
            "event_id": event.id,
            "event_type": event.event_type,
            "aggregate_type": event.aggregate_type,
            "aggregate_id": event.aggregate_id,
            "idempotency_key": event.idempotency_key,
            "payload_json": json.dumps(event.payload, sort_keys=True, separators=(",", ":")),
        }
        kwargs: dict[str, object] = {}
        if self.maxlen is not None:
            kwargs["maxlen"] = self.maxlen
            kwargs["approximate"] = True
        await self.redis_client.xadd(self.stream_name, fields, **kwargs)
