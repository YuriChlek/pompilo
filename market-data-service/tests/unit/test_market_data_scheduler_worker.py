from __future__ import annotations

import unittest

from market_data_service.domain.scheduler_models import SchedulerTickResult
from market_data_service.workers.market_data_scheduler_worker import MarketDataSchedulerWorker


class FakeSchedulerService:
    def __init__(self) -> None:
        self.calls = 0

    async def tick(self) -> SchedulerTickResult:
        self.calls += 1
        return SchedulerTickResult(created_count=1, skipped_count=0)


class MarketDataSchedulerWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_once_delegates_to_service(self) -> None:
        service = FakeSchedulerService()
        worker = MarketDataSchedulerWorker(service, poll_interval_seconds=0)

        result = await worker.run_once()

        self.assertEqual(result.created_count, 1)
        self.assertEqual(service.calls, 1)

    async def test_run_forever_stops_when_stop_hook_is_true(self) -> None:
        service = FakeSchedulerService()
        checks = 0

        def should_stop() -> bool:
            nonlocal checks
            checks += 1
            return checks > 2

        worker = MarketDataSchedulerWorker(service, poll_interval_seconds=0, should_stop=should_stop)

        await worker.run_forever()

        self.assertEqual(service.calls, 2)
