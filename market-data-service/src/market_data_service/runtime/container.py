from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from market_data_service.application.services.backfill_planning_service import BackfillPlanningService
from market_data_service.application.services.backfill_command_service import BackfillCommandService
from market_data_service.application.services.gap_scan_service import GapScanService
from market_data_service.application.services.market_data_scheduler_service import (
    MarketDataSchedulerConfig,
    MarketDataSchedulerService,
)
from market_data_service.application.services.outbox_publisher_service import OutboxPublisherService
from market_data_service.application.services.outbox_replay_service import OutboxReplayService
from market_data_service.application.services.snapshot_read_service import SnapshotReadService
from market_data_service.application.services.single_symbol_sync_service import SingleSymbolSyncService
from market_data_service.application.services.symbol_registry_sync_service import SymbolRegistrySyncService
from market_data_service.config.settings import MarketDataServiceSettings, load_settings
from market_data_service.infrastructure.concurrency_limiter import AsyncConcurrencyLimiter
from market_data_service.infrastructure.providers.binance_spot_adapter import BinanceSpotAdapter
from market_data_service.infrastructure.providers.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from market_data_service.infrastructure.providers.fixture_candle_provider import FixtureCandleProvider
from market_data_service.infrastructure.queues.redis_stream_broker import RedisStreamEventBroker
from market_data_service.persistence.repositories.advisory_lock_repository import AdvisoryLockRepository
from market_data_service.persistence.repositories.backfill_request_repository import BackfillRequestRepository
from market_data_service.persistence.repositories.batch_repository import BatchRepository
from market_data_service.persistence.repositories.candle_repository import CandleRepository
from market_data_service.persistence.repositories.gap_scan_repository import GapScanRepository
from market_data_service.persistence.repositories.outbox_repository import OutboxRepository
from market_data_service.persistence.repositories.provider_symbol_repository import ProviderSymbolRepository
from market_data_service.persistence.repositories.snapshot_repository import SnapshotRepository
from market_data_service.persistence.repositories.symbol_registry_repository import SymbolRegistryRepository
from market_data_service.persistence.repositories.sync_completion_repository import SyncCompletionRepository
from market_data_service.persistence.repositories.sync_job_repository import SyncJobRepository
from market_data_service.workers.market_data_scheduler_worker import MarketDataSchedulerWorker
from market_data_service.workers.outbox_publisher_lifecycle import OutboxPublisherLifecycle
from market_data_service.workers.outbox_publisher_worker import OutboxPublisherWorker
from market_data_service.workers.scheduler_lifecycle import MarketDataSchedulerLifecycle

EngineFactory = Callable[..., AsyncEngine]
RedisClientFactory = Callable[[str], Any]


@dataclass(frozen=True, slots=True)
class RuntimeRepositories:
    advisory_lock: AdvisoryLockRepository
    backfill_request: BackfillRequestRepository
    batch: BatchRepository
    candle: CandleRepository
    gap_scan: GapScanRepository
    outbox: OutboxRepository
    provider_symbol: ProviderSymbolRepository
    snapshot: SnapshotRepository
    symbol_registry: SymbolRegistryRepository
    sync_completion: SyncCompletionRepository
    sync_job: SyncJobRepository


@dataclass(frozen=True, slots=True)
class RuntimeServices:
    backfill_command: BackfillCommandService
    backfill_planning: BackfillPlanningService
    gap_scan: GapScanService
    market_data_scheduler: MarketDataSchedulerService
    outbox_publisher: OutboxPublisherService
    outbox_replay: OutboxReplayService
    single_symbol_sync: SingleSymbolSyncService
    snapshot_read: SnapshotReadService
    symbol_registry_sync: SymbolRegistrySyncService


@dataclass(frozen=True, slots=True)
class RuntimeWorkers:
    market_data_scheduler: MarketDataSchedulerWorker
    outbox_publisher: OutboxPublisherWorker


@dataclass(slots=True)
class MarketDataRuntimeContainer:
    settings: MarketDataServiceSettings
    engine: AsyncEngine
    connection: AsyncConnection
    redis_client: Any
    provider_adapter: BinanceSpotAdapter | FixtureCandleProvider
    event_broker: RedisStreamEventBroker
    repositories: RuntimeRepositories
    services: RuntimeServices
    workers: RuntimeWorkers
    scheduler_lifecycle: MarketDataSchedulerLifecycle
    outbox_publisher_lifecycle: OutboxPublisherLifecycle
    _closed: bool = False

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await _close_if_supported(self.redis_client)
        await self.connection.close()
        await self.engine.dispose()


async def build_runtime_container(
    settings: MarketDataServiceSettings | None = None,
    *,
    engine_factory: EngineFactory = create_async_engine,
    redis_client_factory: RedisClientFactory | None = None,
) -> MarketDataRuntimeContainer:
    runtime_settings = settings or load_settings()
    engine = engine_factory(runtime_settings.database.database_url, pool_pre_ping=True)
    connection = await engine.connect()
    redis_client = _create_redis_client(runtime_settings.redis.redis_url, redis_client_factory)

    repositories = _build_repositories(connection)
    event_broker = RedisStreamEventBroker(
        redis_client=redis_client,
        stream_name=runtime_settings.redis.stream_name,
        maxlen=runtime_settings.redis.maxlen,
    )
    provider_adapter = _build_provider_adapter(runtime_settings)
    services = _build_services(
        settings=runtime_settings,
        repositories=repositories,
        event_broker=event_broker,
        provider_adapter=provider_adapter,
    )
    workers = _build_workers(settings=runtime_settings, services=services)
    scheduler_lifecycle = MarketDataSchedulerLifecycle(workers.market_data_scheduler)
    outbox_publisher_lifecycle = OutboxPublisherLifecycle(workers.outbox_publisher)

    return MarketDataRuntimeContainer(
        settings=runtime_settings,
        engine=engine,
        connection=connection,
        redis_client=redis_client,
        provider_adapter=provider_adapter,
        event_broker=event_broker,
        repositories=repositories,
        services=services,
        workers=workers,
        scheduler_lifecycle=scheduler_lifecycle,
        outbox_publisher_lifecycle=outbox_publisher_lifecycle,
    )


def _build_repositories(connection: AsyncConnection) -> RuntimeRepositories:
    return RuntimeRepositories(
        advisory_lock=AdvisoryLockRepository(connection),
        backfill_request=BackfillRequestRepository(connection),
        batch=BatchRepository(connection),
        candle=CandleRepository(connection),
        gap_scan=GapScanRepository(connection),
        outbox=OutboxRepository(connection),
        provider_symbol=ProviderSymbolRepository(connection),
        snapshot=SnapshotRepository(connection),
        symbol_registry=SymbolRegistryRepository(connection),
        sync_completion=SyncCompletionRepository(connection),
        sync_job=SyncJobRepository(connection),
    )


def _build_provider_adapter(settings: MarketDataServiceSettings) -> BinanceSpotAdapter | FixtureCandleProvider:
    if settings.provider_mode == "fixture":
        return FixtureCandleProvider()
    return BinanceSpotAdapter(
        settings.provider,
        concurrency_limiter=AsyncConcurrencyLimiter(settings.provider.max_concurrent_requests),
        circuit_breaker=CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=settings.provider.circuit_breaker_failure_threshold,
                recovery_timeout=timedelta(seconds=settings.provider.circuit_breaker_recovery_timeout_seconds),
            ),
        ),
    )


def _build_services(
    *,
    settings: MarketDataServiceSettings,
    repositories: RuntimeRepositories,
    event_broker: RedisStreamEventBroker,
    provider_adapter: BinanceSpotAdapter,
) -> RuntimeServices:
    backfill_planning = BackfillPlanningService(repositories.backfill_request)
    backfill_command = BackfillCommandService(
        symbol_registry=repositories.provider_symbol,
        backfill_requester=repositories.backfill_request,
    )
    gap_scan = GapScanService(
        symbol_registry=repositories.provider_symbol,
        candle_reader=repositories.gap_scan,
        backfill_requester=repositories.backfill_request,
    )
    market_data_scheduler = MarketDataSchedulerService(
        sync_job_queue=repositories.sync_job,
        config=MarketDataSchedulerConfig(
            source=settings.scheduler.source,
            provider_symbols=settings.scheduler.provider_symbols,
            timeframes=settings.scheduler.timeframes,
            safety_delay_by_timeframe=settings.scheduler.safety_delay_by_timeframe,
            jitter_seconds=settings.scheduler.jitter_seconds,
        ),
    )
    outbox_publisher = OutboxPublisherService(
        outbox_store=repositories.outbox,
        broker=event_broker,
    )
    outbox_replay = OutboxReplayService(
        outbox_store=repositories.outbox,
        broker=event_broker,
        now_provider=_utc_now,
    )
    snapshot_read = SnapshotReadService(
        symbol_registry=repositories.provider_symbol,
        snapshot_reader=repositories.snapshot,
        now_provider=_utc_now,
    )
    single_symbol_sync = SingleSymbolSyncService(
        symbol_registry=repositories.provider_symbol,
        candle_provider=provider_adapter,
        candle_writer=repositories.candle,
        advisory_lock=repositories.advisory_lock,
        batch_tracker=repositories.batch,
        sync_completion=repositories.sync_completion,
        backfill_planning=backfill_planning,
    )
    symbol_registry_sync = SymbolRegistrySyncService(repositories.symbol_registry)
    return RuntimeServices(
        backfill_command=backfill_command,
        backfill_planning=backfill_planning,
        gap_scan=gap_scan,
        market_data_scheduler=market_data_scheduler,
        outbox_publisher=outbox_publisher,
        outbox_replay=outbox_replay,
        single_symbol_sync=single_symbol_sync,
        snapshot_read=snapshot_read,
        symbol_registry_sync=symbol_registry_sync,
    )


def _build_workers(*, settings: MarketDataServiceSettings, services: RuntimeServices) -> RuntimeWorkers:
    return RuntimeWorkers(
        market_data_scheduler=MarketDataSchedulerWorker(
            services.market_data_scheduler,
            poll_interval_seconds=settings.scheduler.poll_interval_seconds,
        ),
        outbox_publisher=OutboxPublisherWorker(services.outbox_publisher),
    )


def _create_redis_client(redis_url: str, redis_client_factory: RedisClientFactory | None) -> Any:
    if redis_client_factory is not None:
        return redis_client_factory(redis_url)

    from redis.asyncio import Redis

    return Redis.from_url(redis_url, decode_responses=True)


async def _close_if_supported(client: Any) -> None:
    close = getattr(client, "aclose", None) or getattr(client, "close", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def _utc_now():
    from datetime import UTC, datetime

    return datetime.now(UTC)
