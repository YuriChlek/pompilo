from __future__ import annotations

import asyncio
from pathlib import Path

from bot_platform_service.application import MarketDataEventConsumerService
from bot_platform_service.infrastructure.market_data import RedisStreamMessage
from bot_platform_service.persistence.tables import market_data_processed_events
from bot_platform_service.workers import MarketDataEventConsumerWorker


REPO_ROOT = Path(__file__).resolve().parents[3]


class _Logger:
    def __init__(self) -> None:
        self.info_records: list[tuple[str, dict[str, object]]] = []
        self.warning_records: list[tuple[str, dict[str, object]]] = []

    def info(self, event: str, **fields: object) -> None:
        self.info_records.append((event, dict(fields)))

    def warning(self, event: str, **fields: object) -> None:
        self.warning_records.append((event, dict(fields)))


class _IdempotencyStore:
    def __init__(self) -> None:
        self.keys: set[str] = set()
        self.calls: list[tuple[str, str]] = []

    async def record_processed_event(self, *, event, redis_message_id: str, payload_json) -> bool:
        self.calls.append((event.idempotency_key, redis_message_id))
        if event.idempotency_key in self.keys:
            return False
        self.keys.add(event.idempotency_key)
        return True


class _FailingIdempotencyStore:
    async def record_processed_event(self, **kwargs) -> bool:
        raise RuntimeError("database unavailable")


class _StreamConsumer:
    def __init__(self, messages: tuple[RedisStreamMessage, ...]) -> None:
        self.messages = messages
        self.acked: list[str] = []

    async def ensure_consumer_group(self) -> None:
        return None

    async def read_batch(self) -> tuple[RedisStreamMessage, ...]:
        return self.messages

    async def ack(self, message_id: str) -> None:
        self.acked.append(message_id)


def test_stage_30_processed_event_table_and_migration_exist() -> None:
    migration = (
        REPO_ROOT
        / "bot-platform-service/alembic/versions/20260720_0003_create_market_data_processed_events.py"
    ).read_text(encoding="utf-8")

    assert market_data_processed_events.c.idempotency_key.primary_key is True
    assert "payload_json" in market_data_processed_events.c
    assert "processing_status" in market_data_processed_events.c
    assert "market_data_processed_events" in migration
    assert "market_data_processed_events_redis_message_id_uq" in migration
    assert "market_data_processed_events_processing_status_values_ck" in migration


def test_stage_30_duplicate_event_delivery_is_acknowledged_as_noop() -> None:
    async def run() -> None:
        logger = _Logger()
        store = _IdempotencyStore()
        service = MarketDataEventConsumerService(logger=logger, idempotency_store=store)

        first = await service.handle_message(message_id="1-0", payload=_valid_event_payload())
        second = await service.handle_message(message_id="2-0", payload=_valid_event_payload())

        assert first.ack is True
        assert first.recognized is True
        assert first.duplicate is False
        assert second.ack is True
        assert second.recognized is False
        assert second.duplicate is True
        assert store.calls == [
            ("BINANCE_SPOT:BTCUSDT:1h:2026-07-16T01:00:00Z", "1-0"),
            ("BINANCE_SPOT:BTCUSDT:1h:2026-07-16T01:00:00Z", "2-0"),
        ]
        assert [record[0] for record in logger.info_records] == [
            "bot_platform.market_data_event.recognized",
            "bot_platform.market_data_event.duplicate",
        ]

    asyncio.run(run())


def test_stage_30_worker_does_not_ack_when_idempotency_store_fails() -> None:
    async def run() -> None:
        stream = _StreamConsumer((RedisStreamMessage(message_id="1-0", fields=_valid_event_payload()),))
        worker = MarketDataEventConsumerWorker(
            stream_consumer=stream,
            service=MarketDataEventConsumerService(
                logger=_Logger(),
                idempotency_store=_FailingIdempotencyStore(),
            ),
            retry_backoff_seconds=0,
        )

        try:
            await worker.run_once()
        except RuntimeError as exc:
            assert str(exc) == "database unavailable"
        else:
            raise AssertionError("expected idempotency store failure")

        assert stream.acked == []

    asyncio.run(run())


def _valid_event_payload() -> dict[str, object]:
    return {
        "event_type": "market_data.candles_collected",
        "contract_version": "market-data-event.v1",
        "source": "BINANCE_SPOT",
        "symbol": "BTCUSDT",
        "provider_symbol": "BTCUSDT",
        "timeframe": "1h",
        "from": "2026-07-16T00:00:00Z",
        "to": "2026-07-16T01:00:00Z",
        "batch_id": "batch-1",
        "snapshot_id": "snapshot-1",
        "closed_at": "2026-07-16T01:00:00Z",
        "idempotency_key": "BINANCE_SPOT:BTCUSDT:1h:2026-07-16T01:00:00Z",
    }
