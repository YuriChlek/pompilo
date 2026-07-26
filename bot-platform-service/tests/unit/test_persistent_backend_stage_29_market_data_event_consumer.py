from __future__ import annotations

import asyncio
import json
from pathlib import Path

from bot_platform_service.application import MarketDataEventConsumerService
from bot_platform_service.config.settings import BotPlatformSettings
from bot_platform_service.domain import parse_market_data_candles_collected_event
from bot_platform_service.infrastructure.market_data import RedisStreamMarketDataEventConsumer, RedisStreamMessage
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


class _StreamConsumer:
    def __init__(self, messages: tuple[RedisStreamMessage, ...]) -> None:
        self.messages = messages
        self.ensure_calls = 0
        self.acked: list[str] = []

    async def ensure_consumer_group(self) -> None:
        self.ensure_calls += 1

    async def read_batch(self) -> tuple[RedisStreamMessage, ...]:
        return self.messages

    async def ack(self, message_id: str) -> None:
        self.acked.append(message_id)


class _RedisClient:
    def __init__(self, response) -> None:
        self.response = response

    async def xgroup_create(self, *args, **kwargs) -> None:
        return None

    async def xreadgroup(self, *args, **kwargs):
        return self.response

    async def xack(self, *args, **kwargs) -> None:
        return None


def test_stage_29_settings_and_compose_expose_market_data_event_consumer(monkeypatch) -> None:
    monkeypatch.setenv("BOT_PLATFORM_MARKET_DATA_EVENTS_ENABLED", "true")
    monkeypatch.setenv("BOT_PLATFORM_MARKET_DATA_EVENTS_REDIS_URL", "redis://events.local:6379/2")
    monkeypatch.setenv("BOT_PLATFORM_MARKET_DATA_EVENTS_STREAM", "market-data-events")
    monkeypatch.setenv("BOT_PLATFORM_MARKET_DATA_EVENTS_CONSUMER_GROUP", "bot-platform")

    settings = BotPlatformSettings.from_env()
    compose = (REPO_ROOT / "infra/compose/docker-compose.platform.yaml").read_text(encoding="utf-8")

    assert settings.market_data_events.enabled is True
    assert settings.market_data_events.redis_url == "redis://events.local:6379/2"
    assert settings.market_data_events.stream_name == "market-data-events"
    assert settings.market_data_events.consumer_group == "bot-platform"
    assert "BOT_PLATFORM_MARKET_DATA_EVENTS_ENABLED" in compose
    assert "BOT_PLATFORM_MARKET_DATA_EVENTS_STREAM" in compose


def test_stage_29_validates_market_data_candles_collected_contract() -> None:
    event = parse_market_data_candles_collected_event(_valid_event_payload())

    assert event.source == "BINANCE_SPOT"
    assert event.symbol == "BTCUSDT"
    assert event.timeframe == "1h"
    assert event.snapshot_id == "snapshot-1"
    assert event.idempotency_key == "BINANCE_SPOT:BTCUSDT:1h:2026-07-16T01:00:00Z"


def test_stage_29_consumer_logs_recognized_event_without_adapter_execution() -> None:
    async def run() -> None:
        logger = _Logger()
        service = MarketDataEventConsumerService(logger=logger)

        result = await service.handle_message(message_id="1-0", payload=_valid_event_payload())

        assert result.ack is True
        assert result.recognized is True
        assert result.terminal is True
        assert result.event is not None
        assert logger.info_records == [
            (
                "bot_platform.market_data_event.recognized",
                {
                    "message_id": "1-0",
                    "source": "BINANCE_SPOT",
                    "symbol": "BTCUSDT",
                    "timeframe": "1h",
                    "snapshot_id": "snapshot-1",
                    "idempotency_key": "BINANCE_SPOT:BTCUSDT:1h:2026-07-16T01:00:00Z",
                },
            )
        ]

    asyncio.run(run())


def test_stage_29_consumer_ignores_candle_batch_ready_events() -> None:
    async def run() -> None:
        logger = _Logger()
        service = MarketDataEventConsumerService(logger=logger)

        result = await service.handle_message(
            message_id="1-0",
            payload={
                "event_type": "CandleBatchReady",
                "snapshot_id": "snapshot-1",
                "source": "BINANCE_SPOT",
            },
        )

        assert result.ack is True
        assert result.recognized is False
        assert result.terminal is True
        assert logger.info_records == [
            (
                "bot_platform.market_data_event.ignored",
                {
                    "message_id": "1-0",
                    "event_type": "CandleBatchReady",
                },
            )
        ]
        assert logger.warning_records == []

    asyncio.run(run())


def test_stage_29_redis_consumer_decodes_outbox_payload_json() -> None:
    async def run() -> None:
        redis = _RedisClient(
            [
                (
                    "market-data-events",
                    [
                        (
                            "1-0",
                            {
                                "event_type": "market_data.candles_collected",
                                "payload_json": json.dumps(_valid_event_payload()),
                            },
                        )
                    ],
                )
            ]
        )
        consumer = RedisStreamMarketDataEventConsumer(
            redis_client=redis,
            stream_name="market-data-events",
            consumer_group="bot-platform",
            consumer_name="test",
        )

        messages = await consumer.read_batch()

        assert len(messages) == 1
        assert messages[0].fields["contract_version"] == "market-data-event.v1"
        assert messages[0].fields["snapshot_id"] == "snapshot-1"

    asyncio.run(run())


def test_stage_29_worker_acks_valid_and_invalid_events_as_terminal_noop() -> None:
    async def run() -> None:
        logger = _Logger()
        valid = RedisStreamMessage(message_id="1-0", fields=_valid_event_payload())
        invalid = RedisStreamMessage(message_id="2-0", fields={"event_type": "market_data.unknown"})
        stream = _StreamConsumer((valid, invalid))
        worker = MarketDataEventConsumerWorker(
            stream_consumer=stream,
            service=MarketDataEventConsumerService(logger=logger),
            retry_backoff_seconds=0,
        )

        ack_count = await worker.run_once()

        assert stream.ensure_calls == 1
        assert ack_count == 2
        assert stream.acked == ["1-0", "2-0"]
        assert logger.info_records[0][0] == "bot_platform.market_data_event.recognized"
        assert logger.warning_records[0][0] == "bot_platform.market_data_event.invalid"

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
