from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any


@dataclass(frozen=True, slots=True)
class RedisStreamMessage:
    """One decoded Redis Stream message."""

    message_id: str
    fields: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class RedisStreamMarketDataEventConsumer:
    """Redis Stream consumer for Market Data events."""

    redis_client: Any
    stream_name: str
    consumer_group: str
    consumer_name: str
    read_count: int = 10
    block_milliseconds: int = 5000

    async def ensure_consumer_group(self) -> None:
        """Create the configured consumer group if it does not already exist."""

        try:
            await self.redis_client.xgroup_create(
                self.stream_name,
                self.consumer_group,
                id="0",
                mkstream=True,
            )
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def read_batch(self) -> tuple[RedisStreamMessage, ...]:
        """Read a batch of messages through the configured consumer group."""

        response = await self.redis_client.xreadgroup(
            self.consumer_group,
            self.consumer_name,
            streams={self.stream_name: ">"},
            count=self.read_count,
            block=self.block_milliseconds,
        )
        messages: list[RedisStreamMessage] = []
        for _stream_name, stream_messages in response:
            for message_id, fields in stream_messages:
                messages.append(RedisStreamMessage(message_id=str(message_id), fields=_decode_fields(fields)))
        return tuple(messages)

    async def ack(self, message_id: str) -> None:
        """Acknowledge one processed stream message."""

        await self.redis_client.xack(self.stream_name, self.consumer_group, message_id)


def build_redis_market_data_event_consumer(
    *,
    redis_url: str,
    stream_name: str,
    consumer_group: str,
    consumer_name: str,
    read_count: int,
    block_milliseconds: int,
) -> RedisStreamMarketDataEventConsumer:
    """Build Redis-backed Market Data event consumer using runtime dependency."""

    from redis.asyncio import Redis

    return RedisStreamMarketDataEventConsumer(
        redis_client=Redis.from_url(redis_url, decode_responses=True),
        stream_name=stream_name,
        consumer_group=consumer_group,
        consumer_name=consumer_name,
        read_count=read_count,
        block_milliseconds=block_milliseconds,
    )


def _decode_fields(fields: Mapping[str, object]) -> Mapping[str, object]:
    payload_json = fields.get("payload_json")
    if isinstance(payload_json, str) and payload_json.strip():
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            return dict(fields)
        if isinstance(payload, dict):
            return payload
    return dict(fields)
