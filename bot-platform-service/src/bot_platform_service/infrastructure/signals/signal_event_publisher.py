from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from bot_platform_service.domain import BotSignal


@dataclass(frozen=True, slots=True)
class DisabledSignalEventPublisher:
    """No-op downstream signal event publisher used when event publishing is disabled."""

    async def publish_persisted_signal(
        self,
        *,
        signal_id: str,
        run_id: str,
        signal: BotSignal,
        correlation_id: str | None = None,
    ) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class RedisStreamSignalEventPublisher:
    """Duplicate-safe Redis Stream publisher for already persisted signals."""

    redis_client: Any
    stream_name: str
    max_retries: int = 3
    retry_backoff_seconds: float = 0.25

    async def publish_persisted_signal(
        self,
        *,
        signal_id: str,
        run_id: str,
        signal: BotSignal,
        correlation_id: str | None = None,
    ) -> bool:
        """Publish a persisted signal event and return False for duplicate idempotency keys."""

        idempotency_key = _idempotency_key(signal_id)
        acquired = await self.redis_client.set(idempotency_key, "1", nx=True)
        if not acquired:
            return False
        try:
            await self._publish_with_retries(
                {
                    "event_type": "bot_signal.persisted.v1",
                    "idempotency_key": idempotency_key,
                    "signal_id": signal_id,
                    "signal_key": signal.signal_key,
                    "run_id": run_id,
                    "instance_id": signal.instance_id,
                    "module_id": signal.module_id,
                    "symbol": signal.symbol,
                    "timeframe": signal.timeframe,
                    "snapshot_id": signal.snapshot_id,
                    "signal_type": signal.signal_type.value,
                    "side": signal.side.value if signal.side is not None else "",
                    "payload_schema": signal.payload_schema,
                    "payload_schema_version": str(signal.payload_schema_version),
                    "payload_hash": signal.payload_hash,
                    "correlation_id": correlation_id or "",
                }
            )
        except Exception:
            await self.redis_client.delete(idempotency_key)
            raise
        return True

    async def _publish_with_retries(self, fields: dict[str, str]) -> None:
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                await self.redis_client.xadd(self.stream_name, fields)
                return
            except Exception:
                if attempt >= attempts - 1:
                    raise
                if self.retry_backoff_seconds > 0:
                    await asyncio.sleep(self.retry_backoff_seconds * (2**attempt))


def build_redis_signal_event_publisher(
    *,
    redis_url: str,
    stream_name: str,
    max_retries: int,
    retry_backoff_seconds: float,
) -> RedisStreamSignalEventPublisher:
    """Build Redis-backed signal event publisher using an optional runtime dependency."""

    from redis.asyncio import Redis

    return RedisStreamSignalEventPublisher(
        redis_client=Redis.from_url(redis_url, decode_responses=True),
        stream_name=stream_name,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )


def _idempotency_key(signal_id: str) -> str:
    return f"bot-platform:signal-event:{signal_id}"
