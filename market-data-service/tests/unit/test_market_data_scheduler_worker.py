from __future__ import annotations

import unittest

from market_data_service.application.services.candle_collection_service import CollectionResult
from market_data_service.workers.market_data_scheduler_worker import MarketDataSchedulerWorker


class FakeCandleCollectionService:
    def __init__(self) -> None:
        self.calls = 0

    async def collect(self) -> CollectionResult:
        self.calls += 1
        return CollectionResult(scheduled_count=1, processed_count=0, failed_count=0, published_count=0)


class MarketDataSchedulerWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_once_delegates_to_service(self) -> None:
        service = FakeCandleCollectionService()
        worker = MarketDataSchedulerWorker(service, poll_interval_seconds=0)

        result = await worker.run_once()

        self.assertEqual(result.scheduled_count, 1)
        self.assertEqual(service.calls, 1)

    async def test_run_forever_stops_when_stop_hook_is_true(self) -> None:
        service = FakeCandleCollectionService()
        checks = 0

        def should_stop() -> bool:
            nonlocal checks
            checks += 1
            return checks > 2

        worker = MarketDataSchedulerWorker(service, poll_interval_seconds=0, should_stop=should_stop)

        await worker.run_forever()

        self.assertEqual(service.calls, 2)
