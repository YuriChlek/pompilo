from __future__ import annotations

from datetime import UTC, datetime, timedelta
import unittest

from market_data_service.application.services.market_data_scheduler_service import (
    MarketDataSchedulerConfig,
    MarketDataSchedulerService,
)
from market_data_service.domain.enums import MarketDataSource
from market_data_service.domain.scheduler_models import MarketDataSyncJob


class FakeSyncJobQueue:
    def __init__(self, outcomes: tuple[bool, ...]) -> None:
        self.outcomes = list(outcomes)
        self.jobs: list[MarketDataSyncJob] = []

    async def enqueue_sync_job(self, job: MarketDataSyncJob) -> bool:
        self.jobs.append(job)
        return self.outcomes.pop(0)


class MarketDataSchedulerServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_tick_enqueues_due_jobs_and_counts_duplicates(self) -> None:
        queue = FakeSyncJobQueue(outcomes=(True, False))
        service = MarketDataSchedulerService(
            sync_job_queue=queue,
            config=MarketDataSchedulerConfig(
                source=MarketDataSource.BINANCE_SPOT,
                provider_symbols=("ETHUSDT", "BTCUSDT"),
                timeframes=("1h",),
                safety_delay_by_timeframe={"1h": timedelta(seconds=30)},
                jitter_seconds=0,
            ),
            now_provider=lambda: datetime(2026, 7, 14, 10, 0, 31, tzinfo=UTC),
        )

        result = await service.tick()

        self.assertEqual(result.created_count, 1)
        self.assertEqual(result.skipped_count, 1)
        self.assertEqual(len(queue.jobs), 2)
        self.assertEqual({job.expected_close_time for job in queue.jobs}, {datetime(2026, 7, 14, 10, tzinfo=UTC)})
