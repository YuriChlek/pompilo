from __future__ import annotations

import asyncio
from collections.abc import Callable

from market_data_service.application.services.outbox_publisher_service import (
    OutboxPublishBatchResult,
    OutboxPublisherService,
)


class OutboxPublisherWorker:
    def __init__(
        self,
        publisher_service: OutboxPublisherService,
        *,
        poll_interval_seconds: float = 1.0,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        self.publisher_service = publisher_service
        self.poll_interval_seconds = poll_interval_seconds
        self.should_stop = should_stop or (lambda: False)

    async def run_once(self) -> OutboxPublishBatchResult:
        return await self.publisher_service.publish_once()

    async def run_forever(self) -> None:
        while not self.should_stop():
            await self.run_once()
            await asyncio.sleep(self.poll_interval_seconds)
