from __future__ import annotations

from datetime import UTC, datetime
import unittest

from market_data_service.application.services.outbox_publisher_service import (
    OutboxPublisherConfig,
    OutboxPublisherService,
)
from market_data_service.domain.enums import OutboxStatus
from market_data_service.domain.outbox_models import OutboxEvent
from market_data_service.observability.metrics import MARKET_DATA_OUTBOX_LAG_SECONDS, InMemoryMetricsRecorder
from market_data_service.observability.structured_logging import InMemoryStructuredLogger


class FakeOutboxStore:
    def __init__(self, events: list[OutboxEvent]) -> None:
        self.events = events
        self.published: list[dict[str, object]] = []
        self.retries: list[dict[str, object]] = []
        self.failed: list[dict[str, object]] = []

    async def fetch_publishable_events(self, *, limit: int, now: datetime) -> list[OutboxEvent]:
        return self.events[:limit]

    async def mark_published(self, *, event_id: str, published_at: datetime) -> None:
        self.published.append({"event_id": event_id, "published_at": published_at})

    async def mark_retry(self, *, event_id: str, attempts: int, next_attempt_at: datetime) -> None:
        self.retries.append({"event_id": event_id, "attempts": attempts, "next_attempt_at": next_attempt_at})

    async def mark_failed(self, *, event_id: str, attempts: int) -> None:
        self.failed.append({"event_id": event_id, "attempts": attempts})


class FakeBroker:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.published: list[OutboxEvent] = []

    async def publish(self, event: OutboxEvent) -> None:
        if self.fail:
            raise RuntimeError("broker unavailable")
        self.published.append(event)


class OutboxPublisherServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_publish_once_marks_published_only_after_broker_success(self) -> None:
        event = _event("event-1", attempts=0)
        store = FakeOutboxStore([event])
        broker = FakeBroker()
        service = OutboxPublisherService(
            outbox_store=store,
            broker=broker,
            now_provider=lambda: datetime(2026, 7, 14, 10, tzinfo=UTC),
        )

        result = await service.publish_once()

        self.assertEqual(result.published_count, 1)
        self.assertEqual(broker.published, [event])
        self.assertEqual(store.published[0]["event_id"], "event-1")

    async def test_publish_once_schedules_retry_before_max_attempts(self) -> None:
        store = FakeOutboxStore([_event("event-1", attempts=1)])
        service = OutboxPublisherService(
            outbox_store=store,
            broker=FakeBroker(fail=True),
            config=OutboxPublisherConfig(max_attempts=3, retry_backoff_seconds=10),
            now_provider=lambda: datetime(2026, 7, 14, 10, tzinfo=UTC),
        )

        result = await service.publish_once()

        self.assertEqual(result.retry_count, 1)
        self.assertEqual(store.retries[0]["attempts"], 2)
        self.assertEqual(store.retries[0]["next_attempt_at"], datetime(2026, 7, 14, 10, 0, 20, tzinfo=UTC))
        self.assertEqual(store.failed, [])

    async def test_publish_once_marks_failed_at_max_attempts(self) -> None:
        store = FakeOutboxStore([_event("event-1", attempts=2)])
        service = OutboxPublisherService(
            outbox_store=store,
            broker=FakeBroker(fail=True),
            config=OutboxPublisherConfig(max_attempts=3, retry_backoff_seconds=10),
            now_provider=lambda: datetime(2026, 7, 14, 10, tzinfo=UTC),
        )

        result = await service.publish_once()

        self.assertEqual(result.failed_count, 1)
        self.assertEqual(store.failed[0]["attempts"], 3)
        self.assertEqual(store.retries, [])

    async def test_publish_once_reports_publisher_lag_metric(self) -> None:
        store = FakeOutboxStore([_event("event-1", attempts=0)])
        service = OutboxPublisherService(
            outbox_store=store,
            broker=FakeBroker(),
            now_provider=lambda: datetime(2026, 7, 14, 10, tzinfo=UTC),
        )

        result = await service.publish_once()

        self.assertEqual(result.publisher_lag_seconds, 3600.0)

    async def test_publish_once_emits_observability_hooks(self) -> None:
        metrics = InMemoryMetricsRecorder()
        logs = InMemoryStructuredLogger()
        service = OutboxPublisherService(
            outbox_store=FakeOutboxStore([_event("event-1", attempts=0)]),
            broker=FakeBroker(),
            now_provider=lambda: datetime(2026, 7, 14, 10, tzinfo=UTC),
            metrics_recorder=metrics,
            structured_logger=logs,
            correlation_id_provider=lambda: "corr-outbox",
        )

        await service.publish_once()

        self.assertEqual(metrics.samples[0].name, MARKET_DATA_OUTBOX_LAG_SECONDS)
        self.assertEqual(metrics.samples[0].value, 3600.0)
        self.assertEqual(logs.events[0].fields["correlation_id"], "corr-outbox")
        self.assertEqual(logs.events[0].fields["published_count"], 1)


def _event(event_id: str, *, attempts: int) -> OutboxEvent:
    return OutboxEvent(
        id=event_id,
        event_type="CandleBatchReady",
        aggregate_type="market_snapshot",
        aggregate_id="snapshot-1",
        payload={"event_id": event_id},
        idempotency_key=f"key-{event_id}",
        status=OutboxStatus.PENDING,
        attempts=attempts,
        next_attempt_at=None,
        created_at=datetime(2026, 7, 14, 9, tzinfo=UTC),
        published_at=None,
    )
