from __future__ import annotations

import unittest

from market_data_service.application.services.outbox_publisher_service import OutboxPublishBatchResult
from market_data_service.observability.metrics import (
    InMemoryMetricsRecorder,
    MARKET_DATA_OUTBOX_EVENTS_FAILED_TOTAL,
    MARKET_DATA_OUTBOX_EVENTS_FETCHED_TOTAL,
    MARKET_DATA_OUTBOX_EVENTS_PUBLISHED_TOTAL,
)
from market_data_service.observability.structured_logging import InMemoryStructuredLogger
from market_data_service.workers.outbox_publisher_lifecycle import OutboxPublisherLifecycle


class OutboxPublisherLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_publish_batch_is_isolated_and_next_batch_can_succeed(self) -> None:
        worker = FakeOutboxPublisherWorker(
            [
                RuntimeError("redis unavailable"),
                OutboxPublishBatchResult(
                    fetched_count=3,
                    published_count=2,
                    retry_count=0,
                    failed_count=1,
                    publisher_lag_seconds=4.0,
                ),
            ]
        )
        metrics = InMemoryMetricsRecorder()
        logs = InMemoryStructuredLogger()
        lifecycle = OutboxPublisherLifecycle(worker, metrics_recorder=metrics, structured_logger=logs)

        await lifecycle._run_publish_safely()
        await lifecycle._run_publish_safely()

        health = lifecycle.health()
        self.assertFalse(health.running)
        self.assertEqual(health.failed_batches, 1)
        self.assertEqual(health.successful_batches, 1)
        self.assertTrue(health.last_publish_ok)
        self.assertIsNone(health.last_error)
        self.assertEqual(
            [sample.name for sample in metrics.samples],
            [
                MARKET_DATA_OUTBOX_EVENTS_FETCHED_TOTAL,
                MARKET_DATA_OUTBOX_EVENTS_PUBLISHED_TOTAL,
                MARKET_DATA_OUTBOX_EVENTS_FAILED_TOTAL,
                MARKET_DATA_OUTBOX_EVENTS_FETCHED_TOTAL,
                MARKET_DATA_OUTBOX_EVENTS_PUBLISHED_TOTAL,
                MARKET_DATA_OUTBOX_EVENTS_FAILED_TOTAL,
            ],
        )
        self.assertEqual([event.fields["status"] for event in logs.events], ["FAILED", "DEGRADED"])

    async def test_stop_waits_for_current_publish_batch_and_prevents_next_batch(self) -> None:
        worker = FakeOutboxPublisherWorker(
            [
                OutboxPublishBatchResult(
                    fetched_count=1,
                    published_count=1,
                    retry_count=0,
                    failed_count=0,
                    publisher_lag_seconds=0.0,
                ),
                OutboxPublishBatchResult(
                    fetched_count=1,
                    published_count=1,
                    retry_count=0,
                    failed_count=0,
                    publisher_lag_seconds=0.0,
                ),
            ]
        )
        worker.poll_interval_seconds = 10
        lifecycle = OutboxPublisherLifecycle(worker)

        lifecycle.start()
        while worker.call_count == 0:
            await worker.wait_for_call()
        await lifecycle.stop()

        self.assertEqual(worker.call_count, 1)
        self.assertFalse(lifecycle.health().running)


class FakeOutboxPublisherWorker:
    def __init__(self, results) -> None:
        self.results = list(results)
        self.poll_interval_seconds = 0.01
        self.call_count = 0
        self._called = False

    async def run_once(self):
        self.call_count += 1
        self._called = True
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def wait_for_call(self) -> None:
        while not self._called:
            import asyncio

            await asyncio.sleep(0)
