from __future__ import annotations

import unittest

from market_data_service.config.settings import load_settings
from market_data_service.infrastructure.providers.binance_spot_adapter import BinanceSpotAdapter
from market_data_service.infrastructure.providers.fixture_candle_provider import FixtureCandleProvider
from market_data_service.infrastructure.queues.redis_stream_broker import RedisStreamEventBroker
from market_data_service.persistence.repositories.outbox_repository import OutboxRepository
from market_data_service.persistence.repositories.symbol_registry_repository import SymbolRegistryRepository
from market_data_service.persistence.repositories.sync_job_repository import SyncJobRepository
from market_data_service.runtime.container import build_runtime_container
from market_data_service.workers.market_data_scheduler_worker import MarketDataSchedulerWorker
from market_data_service.workers.outbox_publisher_lifecycle import OutboxPublisherLifecycle
from market_data_service.workers.outbox_publisher_worker import OutboxPublisherWorker
from market_data_service.workers.scheduler_lifecycle import MarketDataSchedulerLifecycle


class RuntimeContainerTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_container_builds_object_graph_and_closes_resources(self) -> None:
        engine = FakeEngine()
        redis_client = FakeRedisClient()
        settings = load_settings(
            {
                "MARKET_DATA_PROVIDER_SYMBOLS": "ethusdt,btcusdt",
                "MARKET_DATA_TIMEFRAMES": "1h,4h",
                "MARKET_DATA_REDIS_URL": "redis://redis:6379/0",
                "DB_HOST": "postgres",
                "DB_NAME": "pampilo_db",
            }
        )

        container = await build_runtime_container(
            settings,
            engine_factory=lambda *args, **kwargs: engine,
            redis_client_factory=lambda redis_url: redis_client,
        )

        self.assertIs(container.engine, engine)
        self.assertIs(container.connection, engine.connection)
        self.assertIs(container.redis_client, redis_client)
        self.assertIsInstance(container.provider_adapter, BinanceSpotAdapter)
        self.assertIsInstance(container.event_broker, RedisStreamEventBroker)
        self.assertIsInstance(container.repositories.sync_job, SyncJobRepository)
        self.assertIsInstance(container.repositories.outbox, OutboxRepository)
        self.assertIsInstance(container.repositories.symbol_registry, SymbolRegistryRepository)
        self.assertIsInstance(container.workers.market_data_scheduler, MarketDataSchedulerWorker)
        self.assertIsInstance(container.workers.outbox_publisher, OutboxPublisherWorker)
        self.assertIsInstance(container.scheduler_lifecycle, MarketDataSchedulerLifecycle)
        self.assertIsInstance(container.outbox_publisher_lifecycle, OutboxPublisherLifecycle)
        self.assertEqual(
            container.services.market_data_scheduler.config.provider_symbols,
            ("ETHUSDT", "BTCUSDT"),
        )
        self.assertIs(container.services.backfill_command.backfill_requester, container.repositories.backfill_request)
        self.assertIs(container.services.gap_scan.candle_reader, container.repositories.gap_scan)
        self.assertIs(container.services.outbox_replay.outbox_store, container.repositories.outbox)
        self.assertIs(container.services.snapshot_read.snapshot_reader, container.repositories.snapshot)
        self.assertIs(container.services.symbol_registry_sync.repository, container.repositories.symbol_registry)
        self.assertEqual(container.workers.market_data_scheduler.poll_interval_seconds, 30.0)

        await container.close()
        await container.close()

        self.assertEqual(redis_client.close_count, 1)
        self.assertEqual(engine.connection.close_count, 1)
        self.assertEqual(engine.dispose_count, 1)

    async def test_runtime_container_uses_fixture_provider_only_when_explicitly_configured(self) -> None:
        settings = load_settings(
            {
                "MARKET_DATA_PROVIDER_MODE": "fixture",
                "MARKET_DATA_REDIS_URL": "redis://redis:6379/0",
                "DB_HOST": "postgres",
                "DB_NAME": "pampilo_db",
            }
        )
        engine = FakeEngine()

        container = await build_runtime_container(
            settings,
            engine_factory=lambda *args, **kwargs: engine,
            redis_client_factory=lambda redis_url: FakeRedisClient(),
        )

        self.assertIsInstance(container.provider_adapter, FixtureCandleProvider)
        await container.close()


class FakeConnection:
    def __init__(self) -> None:
        self.close_count = 0

    async def close(self) -> None:
        self.close_count += 1


class FakeEngine:
    def __init__(self) -> None:
        self.connection = FakeConnection()
        self.dispose_count = 0

    async def connect(self) -> FakeConnection:
        return self.connection

    async def dispose(self) -> None:
        self.dispose_count += 1


class FakeRedisClient:
    def __init__(self) -> None:
        self.close_count = 0

    async def aclose(self) -> None:
        self.close_count += 1
