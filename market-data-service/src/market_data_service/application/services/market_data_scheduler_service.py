from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from market_data_service.application.scheduler_ports import SyncJobQueuePort
from market_data_service.domain.enums import MarketDataSource
from market_data_service.domain.scheduler import build_sync_jobs
from market_data_service.domain.scheduler_models import SchedulerTickResult


@dataclass(frozen=True, slots=True)
class MarketDataSchedulerConfig:
    source: MarketDataSource
    provider_symbols: tuple[str, ...]
    timeframes: tuple[str, ...]
    safety_delay_by_timeframe: Mapping[str, timedelta]
    jitter_seconds: int


class MarketDataSchedulerService:
    def __init__(
        self,
        *,
        sync_job_queue: SyncJobQueuePort,
        config: MarketDataSchedulerConfig,
        now_provider=None,
    ) -> None:
        self.sync_job_queue = sync_job_queue
        self.config = config
        self.now_provider = now_provider or (lambda: datetime.now(UTC))

    async def tick(self, *, random_seed: int | None = None) -> SchedulerTickResult:
        jobs = build_sync_jobs(
            source=self.config.source,
            provider_symbols=self.config.provider_symbols,
            timeframes=self.config.timeframes,
            now=self.now_provider(),
            safety_delay_by_timeframe=self.config.safety_delay_by_timeframe,
            jitter_seconds=self.config.jitter_seconds,
            random_seed=random_seed,
        )
        created_count = 0
        skipped_count = 0
        for job in jobs:
            if await self.sync_job_queue.enqueue_sync_job(job):
                created_count += 1
            else:
                skipped_count += 1
        return SchedulerTickResult(created_count=created_count, skipped_count=skipped_count)
