from __future__ import annotations

import unittest

from market_data_service.domain.scheduler_models import SchedulerTickResult
from market_data_service.observability.metrics import (
    InMemoryMetricsRecorder,
    MARKET_DATA_SCHEDULER_TICKS_TOTAL,
)
from market_data_service.observability.structured_logging import InMemoryStructuredLogger
from market_data_service.workers.scheduler_lifecycle import MarketDataSchedulerLifecycle


class SchedulerLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_tick_is_isolated_and_next_tick_can_succeed(self) -> None:
        worker = FakeSchedulerWorker([RuntimeError("db temporarily unavailable"), SchedulerTickResult(1, 2)])
        metrics = InMemoryMetricsRecorder()
        logs = InMemoryStructuredLogger()
        lifecycle = MarketDataSchedulerLifecycle(worker, metrics_recorder=metrics, structured_logger=logs)

        await lifecycle._run_tick_safely()
        await lifecycle._run_tick_safely()

        health = lifecycle.health()
        self.assertFalse(health.running)
        self.assertEqual(health.failed_ticks, 1)
        self.assertEqual(health.successful_ticks, 1)
        self.assertTrue(health.last_tick_ok)
        self.assertIsNone(health.last_error)
        self.assertEqual([sample.name for sample in metrics.samples], [MARKET_DATA_SCHEDULER_TICKS_TOTAL] * 2)
        self.assertEqual([sample.labels["status"] for sample in metrics.samples], ["failed", "success"])
        self.assertEqual([event.fields["status"] for event in logs.events], ["FAILED", "COMPLETE"])

    async def test_stop_prevents_additional_scheduler_ticks(self) -> None:
        worker = FakeSchedulerWorker([SchedulerTickResult(1, 0), SchedulerTickResult(1, 0)])
        worker.poll_interval_seconds = 10
        lifecycle = MarketDataSchedulerLifecycle(worker)

        lifecycle.start()
        while worker.call_count == 0:
            await worker.wait_for_call()
        await lifecycle.stop()

        self.assertEqual(worker.call_count, 1)
        self.assertFalse(lifecycle.health().running)


class FakeSchedulerWorker:
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
