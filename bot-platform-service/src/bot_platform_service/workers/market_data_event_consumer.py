from __future__ import annotations

import asyncio
from contextlib import suppress
from collections.abc import Awaitable, Callable
from typing import Protocol

from bot_platform_service.application.market_data_event_consumer_service import MarketDataEventConsumerService


class MarketDataEventStreamConsumer(Protocol):
    """Redis Stream consumer boundary used by the worker."""

    async def ensure_consumer_group(self) -> None:
        """Ensure the consumer group exists."""

    async def read_batch(self):
        """Read available messages."""

    async def ack(self, message_id: str) -> None:
        """Acknowledge one message."""


class MarketDataEventConsumerWorker:
    """Long-running no-op Market Data event consumer skeleton."""

    def __init__(
        self,
        *,
        stream_consumer: MarketDataEventStreamConsumer,
        service: MarketDataEventConsumerService,
        retry_backoff_seconds: float,
        commit: Callable[[], Awaitable[None]] | None = None,
        rollback: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.stream_consumer = stream_consumer
        self.service = service
        self.retry_backoff_seconds = retry_backoff_seconds
        self.commit = commit
        self.rollback = rollback
        self._stop_event = asyncio.Event()

    async def run_forever(self) -> None:
        """Consume stream messages until stopped without running bot adapters."""

        while not self._stop_event.is_set():
            try:
                await self.run_once()
            except Exception:
                with suppress(TimeoutError):
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self.retry_backoff_seconds)

    async def run_once(self) -> int:
        """Read, validate, log, and ACK one batch of Market Data events."""

        await self.stream_consumer.ensure_consumer_group()
        messages = await self.stream_consumer.read_batch()
        ack_count = 0
        for message in messages:
            try:
                result = await self.service.handle_message(
                    message_id=message.message_id,
                    payload=message.fields,
                )
            except Exception:
                if self.rollback is not None:
                    await self.rollback()
                raise
            if result.ack:
                if self.commit is not None:
                    await self.commit()
                await self.stream_consumer.ack(message.message_id)
                ack_count += 1
            elif self.rollback is not None:
                await self.rollback()
        return ack_count

    def stop(self) -> None:
        """Request graceful worker shutdown."""

        self._stop_event.set()
