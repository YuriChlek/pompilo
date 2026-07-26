from __future__ import annotations

import asyncio
import unittest

from market_data_service.application.services.candle_collection_service import CollectionResult
from market_data_service.application.services.outbox_publisher_service import OutboxPublishBatchResult
from market_data_service.observability.metrics import (
    MARKET_DATA_COLLECTION_EVENTS_PUBLISHED_TOTAL,
    MARKET_DATA_COLLECTION_TICKS_TOTAL,
    InMemoryMetricsRecorder,
)
from market_data_service.workers.scheduler_lifecycle import MarketDataSchedulerLifecycle


class SchedulerLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_collection_tick_commits_before_publishing_outbox(self) -> None:
        events: list[str] = []
        metrics = InMemoryMetricsRecorder()
        lifecycle = MarketDataSchedulerLifecycle(
            FakeCollectionWorker(events),
            connection=FakeConnection(events),
            outbox_publisher=FakeOutboxPublisher(events),
            metrics_recorder=metrics,
        )

        await lifecycle._run_tick_safely()

        self.assertEqual(events, ["collect", "commit", "publish", "commit"])
        self.assertTrue(lifecycle.health().last_tick_ok)
        self.assertIn(MARKET_DATA_COLLECTION_TICKS_TOTAL, {sample.name for sample in metrics.samples})
        published = [sample for sample in metrics.samples if sample.name == MARKET_DATA_COLLECTION_EVENTS_PUBLISHED_TOTAL]
        self.assertEqual(published[0].value, 1.0)

    async def test_failed_collection_rolls_back_and_records_failed_tick(self) -> None:
        events: list[str] = []
        metrics = InMemoryMetricsRecorder()
        lifecycle = MarketDataSchedulerLifecycle(
            FailingCollectionWorker(events),
            connection=FakeConnection(events),
            metrics_recorder=metrics,
        )

        await lifecycle._run_tick_safely()

        self.assertEqual(events, ["collect", "rollback"])
        self.assertFalse(lifecycle.health().last_tick_ok)
        self.assertEqual(metrics.samples[1].labels["status"], "failed")

    async def test_stop_waits_for_active_collection_tick(self) -> None:
        events: list[str] = []
        worker = SlowCollectionWorker(events)
        lifecycle = MarketDataSchedulerLifecycle(worker, connection=FakeConnection(events))

        lifecycle.start()
        await worker.wait_for_call()
        stop_task = asyncio.create_task(lifecycle.stop())
        self.assertFalse(stop_task.done())
        worker.release()
        await stop_task

        self.assertEqual(events, ["collect:start", "collect:done", "commit"])
        self.assertFalse(lifecycle.health().running)


class FakeConnection:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.open_transaction = True

    def in_transaction(self) -> bool:
        return self.open_transaction

    async def commit(self) -> None:
        self.events.append("commit")
        self.open_transaction = True

    async def rollback(self) -> None:
        self.events.append("rollback")
        self.open_transaction = False


class FakeCollectionWorker:
    poll_interval_seconds = 0.01

    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def run_once(self) -> CollectionResult:
        self.events.append("collect")
        return CollectionResult(scheduled_count=1, processed_count=1, failed_count=0, published_count=0)


class FailingCollectionWorker(FakeCollectionWorker):
    async def run_once(self) -> CollectionResult:
        self.events.append("collect")
        raise RuntimeError("collection failed")


class SlowCollectionWorker(FakeCollectionWorker):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self._called = asyncio.Event()
        self._release = asyncio.Event()

    async def run_once(self) -> CollectionResult:
        self.events.append("collect:start")
        self._called.set()
        await self._release.wait()
        self.events.append("collect:done")
        return CollectionResult(scheduled_count=1, processed_count=1, failed_count=0, published_count=0)

    async def wait_for_call(self) -> None:
        await self._called.wait()

    def release(self) -> None:
        self._release.set()


class FakeOutboxPublisher:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def run_once(self) -> OutboxPublishBatchResult:
        self.events.append("publish")
        return OutboxPublishBatchResult(
            fetched_count=1,
            published_count=1,
            retry_count=0,
            failed_count=0,
            publisher_lag_seconds=0.0,
        )
