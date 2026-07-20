from __future__ import annotations

import asyncio
from collections.abc import Callable

from market_data_service.application.services.candle_collection_service import (
    CandleCollectionService,
    CollectionResult,
)


class MarketDataSchedulerWorker:
    def __init__(
        self,
        candle_collection: CandleCollectionService,
        *,
        poll_interval_seconds: float,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        self.candle_collection = candle_collection
        self.poll_interval_seconds = poll_interval_seconds
        self.should_stop = should_stop or (lambda: False)

    async def run_once(self) -> CollectionResult:
        return await self.candle_collection.collect()

    async def run_forever(self) -> None:
        while not self.should_stop():
            await self.run_once()

            sleep_remaining = self.poll_interval_seconds
            while sleep_remaining > 0 and not self.should_stop():
                sleep_chunk = min(0.1, sleep_remaining)
                await asyncio.sleep(sleep_chunk)
                sleep_remaining -= sleep_chunk
