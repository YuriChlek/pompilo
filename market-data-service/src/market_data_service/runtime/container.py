from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import timedelta
from collections.abc import Mapping
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
from market_data_service.application.services.multi_provider_symbol_resolver import MultiProviderSymbolResolver
from market_data_service.application.services.candles_get_fetch_service import CandlesGetFetchService
from market_data_service.application.services.outbox_cleanup_service import OutboxCleanupService
from market_data_service.application.services.candle_collection_service import CandleCollectionService
from market_data_service.infrastructure.providers.routing_candle_provider import RoutingCandleProvider
from market_data_service.application.market_data_ports import CandleProviderPort
from market_data_service.domain.enums import MarketDataSource
from market_data_service.config.settings import MarketDataServiceSettings, load_settings
from market_data_service.infrastructure.concurrency_limiter import AsyncConcurrencyLimiter
from market_data_service.infrastructure.providers.binance_spot_adapter import BinanceSpotAdapter
from market_data_service.infrastructure.providers.bybit_spot_adapter import BybitSpotAdapter
from market_data_service.infrastructure.providers.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from market_data_service.infrastructure.providers.fixture_candle_provider import FixtureCandleProvider
from market_data_service.infrastructure.queues.redis_stream_broker import RedisStreamEventBroker
from market_data_service.persistence.repositories.advisory_lock_repository import AdvisoryLockRepository
from market_data_service.persistence.repositories.backfill_request_repository import BackfillRequestRepository
from market_data_service.persistence.repositories.batch_repository import BatchRepository
from market_data_service.persistence.repositories.candle_repository import CandleRepository
from market_data_service.persistence.repositories.collection_state_repository import CollectionStateRepository
from market_data_service.persistence.repositories.gap_scan_repository import GapScanRepository
from market_data_service.persistence.repositories.outbox_repository import OutboxRepository
from market_data_service.persistence.repositories.provider_symbol_repository import ProviderSymbolRepository
from market_data_service.persistence.repositories.provider_symbol_availability_repository import (
    ProviderSymbolAvailabilityRepository,
)
from market_data_service.persistence.repositories.snapshot_repository import SnapshotRepository
from market_data_service.persistence.repositories.symbol_registry_repository import SymbolRegistryRepository
from market_data_service.persistence.repositories.sync_completion_repository import SyncCompletionRepository
from market_data_service.persistence.repositories.sync_job_repository import SyncJobRepository
from market_data_service.domain.availability_rules import AvailabilityCachePolicy
from market_data_service.observability.metrics import PrometheusMetricsRecorder
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
    collection_state: CollectionStateRepository
    gap_scan: GapScanRepository
    outbox: OutboxRepository
    provider_symbol: ProviderSymbolRepository
    provider_symbol_availability: ProviderSymbolAvailabilityRepository
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
    symbol_resolver: MultiProviderSymbolResolver
    candles_get_fetch: CandlesGetFetchService
    outbox_cleanup: OutboxCleanupService
    candle_collection: CandleCollectionService


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
    provider_adapter: CandleProviderPort
    adapters: Mapping[MarketDataSource, CandleProviderPort]
    event_broker: RedisStreamEventBroker
    metrics_recorder: PrometheusMetricsRecorder
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
    metrics_recorder = PrometheusMetricsRecorder()
    adapters = _build_adapters(runtime_settings, metrics_recorder=metrics_recorder)
    provider_adapter = adapters[MarketDataSource.BINANCE_SPOT]
    services = _build_services(
        settings=runtime_settings,
        repositories=repositories,
        event_broker=event_broker,
        provider_adapter=provider_adapter,
        adapters=adapters,
        metrics_recorder=metrics_recorder,
    )
    workers = _build_workers(settings=runtime_settings, services=services)
    scheduler_lifecycle = MarketDataSchedulerLifecycle(
        workers.market_data_scheduler,
        connection=connection,
        outbox_publisher=workers.outbox_publisher if runtime_settings.outbox_publisher_enabled else None,
        collect_on_start=runtime_settings.collect.on_start,
        metrics_recorder=metrics_recorder,
    )
    outbox_publisher_lifecycle = OutboxPublisherLifecycle(
        workers.outbox_publisher,
        metrics_recorder=metrics_recorder,
    )

    return MarketDataRuntimeContainer(
        settings=runtime_settings,
        engine=engine,
        connection=connection,
        redis_client=redis_client,
        provider_adapter=provider_adapter,
        adapters=adapters,
        event_broker=event_broker,
        metrics_recorder=metrics_recorder,
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
        collection_state=CollectionStateRepository(connection),
        gap_scan=GapScanRepository(connection),
        outbox=OutboxRepository(connection),
        provider_symbol=ProviderSymbolRepository(connection),
        provider_symbol_availability=ProviderSymbolAvailabilityRepository(connection),
        snapshot=SnapshotRepository(connection),
        symbol_registry=SymbolRegistryRepository(connection),
        sync_completion=SyncCompletionRepository(connection),
        sync_job=SyncJobRepository(connection),
    )


def _build_adapters(
    settings: MarketDataServiceSettings,
    *,
    metrics_recorder: PrometheusMetricsRecorder,
) -> Mapping[MarketDataSource, CandleProviderPort]:
    if settings.provider_mode == "fixture":
        fixture = FixtureCandleProvider()
        return {
            MarketDataSource.BINANCE_SPOT: fixture,
            MarketDataSource.BYBIT_SPOT: fixture,
        }
    binance = BinanceSpotAdapter(
        settings.provider,
        concurrency_limiter=AsyncConcurrencyLimiter(settings.provider.max_concurrent_requests),
        metrics_recorder=metrics_recorder,
        circuit_breaker=CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=settings.provider.circuit_breaker_failure_threshold,
                recovery_timeout=timedelta(seconds=settings.provider.circuit_breaker_recovery_timeout_seconds),
            ),
        ),
    )
    bybit = BybitSpotAdapter(
        settings.bybit_provider,
        concurrency_limiter=AsyncConcurrencyLimiter(settings.bybit_provider.max_concurrent_requests),
        metrics_recorder=metrics_recorder,
        circuit_breaker=CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=settings.bybit_provider.circuit_breaker_failure_threshold,
                recovery_timeout=timedelta(seconds=settings.bybit_provider.circuit_breaker_recovery_timeout_seconds),
            ),
        ),
    )
    return {
        MarketDataSource.BINANCE_SPOT: binance,
        MarketDataSource.BYBIT_SPOT: bybit,
    }


def _build_services(
    *,
    settings: MarketDataServiceSettings,
    repositories: RuntimeRepositories,
    event_broker: RedisStreamEventBroker,
    provider_adapter: CandleProviderPort,
    adapters: Mapping[MarketDataSource, CandleProviderPort],
    metrics_recorder: PrometheusMetricsRecorder,
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
        metrics_recorder=metrics_recorder,
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
    routing_provider = RoutingCandleProvider(adapters)

    single_symbol_sync = SingleSymbolSyncService(
        symbol_registry=repositories.provider_symbol,
        candle_provider=routing_provider,
        candle_writer=repositories.candle,
        advisory_lock=repositories.advisory_lock,
        batch_tracker=repositories.batch,
        sync_completion=repositories.sync_completion,
        backfill_planning=backfill_planning,
        metrics_recorder=metrics_recorder,
    )
    symbol_registry_sync = SymbolRegistrySyncService(repositories.symbol_registry)

    cache_policy = AvailabilityCachePolicy(
        availability_ttl_hours=settings.availability.availability_ttl_hours,
        unsupported_recheck_hours=settings.availability.unsupported_recheck_hours,
        temporary_error_recheck_minutes=settings.availability.temporary_error_recheck_minutes,
    )
    symbol_resolver = MultiProviderSymbolResolver(
        availability_repository=repositories.provider_symbol_availability,
        adapters=adapters,
        priority=settings.provider_priority,
        cache_policy=cache_policy,
    )
    candles_get_fetch = CandlesGetFetchService(
        resolver=symbol_resolver,
        candle_provider=routing_provider,
        candle_writer=repositories.candle,
    )
    outbox_cleanup = OutboxCleanupService(
        outbox_repository=repositories.outbox,
        retention_days=settings.retention.outbox_retention_days,
        metrics_recorder=metrics_recorder,
    )
    collection_concurrency_limiter = AsyncConcurrencyLimiter(settings.bootstrap.provider_max_concurrency)
    candle_collection = CandleCollectionService(
        resolver=symbol_resolver,
        single_symbol_sync=single_symbol_sync,
        sync_job_queue=repositories.sync_job,
        candle_history=repositories.candle,
        collection_state=repositories.collection_state,
        provider_symbols=settings.scheduler.provider_symbols,
        timeframes=settings.scheduler.timeframes,
        safety_delay_by_timeframe=settings.scheduler.safety_delay_by_timeframe,
        jitter_seconds=settings.scheduler.jitter_seconds,
        bootstrap_lookback_years=settings.bootstrap.bootstrap_lookback_years,
        bootstrap_max_chunks_per_tick=settings.bootstrap.bootstrap_max_chunks_per_tick,
        max_jobs_per_tick=settings.collect.max_jobs_per_tick,
        concurrency_limiter=collection_concurrency_limiter,
    )

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
        symbol_resolver=symbol_resolver,
        candles_get_fetch=candles_get_fetch,
        outbox_cleanup=outbox_cleanup,
        candle_collection=candle_collection,
    )


def _build_workers(*, settings: MarketDataServiceSettings, services: RuntimeServices) -> RuntimeWorkers:
    return RuntimeWorkers(
        market_data_scheduler=MarketDataSchedulerWorker(
            services.candle_collection,
            poll_interval_seconds=settings.collect.poll_interval_seconds,
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
