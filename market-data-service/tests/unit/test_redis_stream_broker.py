from __future__ import annotations

from datetime import UTC, datetime
import json
import unittest

from market_data_service.domain.enums import OutboxStatus
from market_data_service.domain.outbox_models import OutboxEvent
from market_data_service.infrastructure.queues.redis_stream_broker import RedisStreamEventBroker


class FakeRedisClient:
    def __init__(self) -> None:
        self.xadd_calls: list[dict[str, object]] = []

    async def xadd(self, stream_name: str, fields: dict[str, str], **kwargs):
        self.xadd_calls.append({"stream_name": stream_name, "fields": fields, "kwargs": kwargs})
        return "stream-id-1"


class RedisStreamEventBrokerTests(unittest.IsolatedAsyncioTestCase):
    async def test_publish_writes_outbox_event_to_redis_stream(self) -> None:
        redis_client = FakeRedisClient()
        broker = RedisStreamEventBroker(redis_client=redis_client, stream_name="market-data-events", maxlen=1000)

        await broker.publish(_event())

        call = redis_client.xadd_calls[0]
        self.assertEqual(call["stream_name"], "market-data-events")
        self.assertEqual(call["kwargs"], {"maxlen": 1000, "approximate": True})
        fields = call["fields"]
        self.assertEqual(fields["event_id"], "event-1")
        self.assertEqual(fields["event_type"], "CandleBatchReady")
        self.assertEqual(json.loads(fields["payload_json"]), {"snapshot_id": "snapshot-1"})


def _event() -> OutboxEvent:
    return OutboxEvent(
        id="event-1",
        event_type="CandleBatchReady",
        aggregate_type="market_snapshot",
        aggregate_id="snapshot-1",
        payload={"snapshot_id": "snapshot-1"},
        idempotency_key="key-1",
        status=OutboxStatus.PENDING,
        attempts=0,
        next_attempt_at=None,
        created_at=datetime(2026, 7, 14, 9, tzinfo=UTC),
        published_at=None,
    )
