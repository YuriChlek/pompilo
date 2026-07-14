from __future__ import annotations

import asyncio
from collections.abc import Callable

from market_data_service.application.services.market_data_scheduler_service import MarketDataSchedulerService
from market_data_service.domain.scheduler_models import SchedulerTickResult


class MarketDataSchedulerWorker:
    def __init__(
        self,
        scheduler_service: MarketDataSchedulerService,
        *,
        poll_interval_seconds: float,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        self.scheduler_service = scheduler_service
        self.poll_interval_seconds = poll_interval_seconds
        self.should_stop = should_stop or (lambda: False)

    async def run_once(self) -> SchedulerTickResult:
        return await self.scheduler_service.tick()

    async def run_forever(self) -> None:
        while not self.should_stop():
            await self.run_once()
            await asyncio.sleep(self.poll_interval_seconds)
