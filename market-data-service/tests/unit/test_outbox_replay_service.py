from __future__ import annotations

from datetime import UTC, datetime
import unittest

from market_data_service.application.services.outbox_replay_service import OutboxReplayCommand, OutboxReplayService
from market_data_service.domain.enums import OutboxStatus
from market_data_service.domain.outbox_models import OutboxEvent


class OutboxReplayServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_dry_run_does_not_publish_or_mark_state(self) -> None:
        store = FakeOutboxReplayStore(
            [
                _event("event-1", OutboxStatus.PENDING),
                _event("event-2", OutboxStatus.PUBLISHED),
            ]
        )
        broker = FakeBroker()
        service = OutboxReplayService(outbox_store=store, broker=broker, now_provider=_now)

        result = await service.replay(OutboxReplayCommand(from_id="event-1", to_id="event-9", dry_run=True))

        self.assertTrue(result.dry_run)
        self.assertEqual(result.matched_count, 2)
        self.assertEqual(result.publishable_count, 1)
        self.assertEqual(result.published_count, 0)
        self.assertEqual(result.skipped_published_count, 1)
        self.assertEqual(broker.published, [])
        self.assertEqual(store.marked_published, [])

    async def test_replay_skips_already_published_events(self) -> None:
        store = FakeOutboxReplayStore(
            [
                _event("event-1", OutboxStatus.PENDING),
                _event("event-2", OutboxStatus.PUBLISHED),
                _event("event-3", OutboxStatus.FAILED),
            ]
        )
        broker = FakeBroker()
        service = OutboxReplayService(outbox_store=store, broker=broker, now_provider=_now)

        result = await service.replay(OutboxReplayCommand())

        self.assertFalse(result.dry_run)
        self.assertEqual(result.matched_count, 3)
        self.assertEqual(result.publishable_count, 2)
        self.assertEqual(result.published_count, 2)
        self.assertEqual(result.skipped_published_count, 1)
        self.assertEqual([event.id for event in broker.published], ["event-1", "event-3"])
        self.assertEqual(store.marked_published, ["event-1", "event-3"])

    async def test_broker_failure_is_propagated_without_marking_published(self) -> None:
        store = FakeOutboxReplayStore([_event("event-1", OutboxStatus.PENDING)])
        service = OutboxReplayService(outbox_store=store, broker=FakeBroker(fail=True), now_provider=_now)

        with self.assertRaisesRegex(RuntimeError, "redis unavailable"):
            await service.replay(OutboxReplayCommand())

        self.assertEqual(store.marked_published, [])


class FakeOutboxReplayStore:
    def __init__(self, events: list[OutboxEvent]) -> None:
        self.events = events
        self.marked_published: list[str] = []

    async def list_replay_events(self, *, from_id: str | None, to_id: str | None) -> tuple[OutboxEvent, ...]:
        return tuple(
            event
            for event in self.events
            if (from_id is None or event.id >= from_id) and (to_id is None or event.id <= to_id)
        )

    async def mark_published(self, *, event_id: str, published_at) -> None:
        self.marked_published.append(event_id)


class FakeBroker:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.published: list[OutboxEvent] = []

    async def publish(self, event: OutboxEvent) -> None:
        if self.fail:
            raise RuntimeError("redis unavailable")
        self.published.append(event)


def _event(event_id: str, status: OutboxStatus) -> OutboxEvent:
    return OutboxEvent(
        id=event_id,
        event_type="CandleBatchReady",
        aggregate_type="market_snapshot",
        aggregate_id=f"snapshot-{event_id}",
        payload={"event_id": event_id},
        idempotency_key=f"idem-{event_id}",
        status=status,
        attempts=0,
        next_attempt_at=None,
        created_at=_now(),
        published_at=_now() if status == OutboxStatus.PUBLISHED else None,
    )


def _now() -> datetime:
    return datetime(2026, 7, 14, 12, tzinfo=UTC)
