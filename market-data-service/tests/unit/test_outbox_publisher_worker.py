from __future__ import annotations

import unittest

from market_data_service.application.services.outbox_publisher_service import OutboxPublishBatchResult
from market_data_service.workers.outbox_publisher_worker import OutboxPublisherWorker


class FakePublisherService:
    def __init__(self) -> None:
        self.calls = 0

    async def publish_once(self) -> OutboxPublishBatchResult:
        self.calls += 1
        return OutboxPublishBatchResult(
            fetched_count=0,
            published_count=0,
            retry_count=0,
            failed_count=0,
            publisher_lag_seconds=0.0,
        )


class OutboxPublisherWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_once_delegates_to_service(self) -> None:
        service = FakePublisherService()
        worker = OutboxPublisherWorker(service)

        await worker.run_once()

        self.assertEqual(service.calls, 1)

    async def test_run_forever_stops_when_stop_hook_is_true(self) -> None:
        service = FakePublisherService()
        checks = 0

        def should_stop() -> bool:
            nonlocal checks
            checks += 1
            return checks > 2

        worker = OutboxPublisherWorker(service, poll_interval_seconds=0, should_stop=should_stop)

        await worker.run_forever()

        self.assertEqual(service.calls, 2)
