from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Protocol

from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from datetime import UTC, datetime
from typing import Any

from bot_platform_service.application import (
    AdminBotInstanceService,
    AdminMetadataService,
    BotInstanceLifecycleService,
    EventDataCleanupService,
    ManualBotRunService,
    RuntimeCapabilities,
)
from bot_platform_service.application.config_validation_service import BotConfigValidationService
from bot_platform_service.config.settings import BotPlatformSettings
from bot_platform_service.domain import build_payload_hash, canonical_json
from bot_platform_service.infrastructure.market_data.http_snapshot_client import MarketDataHttpSnapshotClient
from bot_platform_service.infrastructure.signals import DisabledSignalEventPublisher, build_redis_signal_event_publisher
from bot_platform_service.observability.metrics import InMemoryMetricsRecorder
from bot_platform_service.persistence.repositories.bot_audit_event_repository import BotAuditEventRepository
from bot_platform_service.persistence.repositories.bot_instance_repository import BotInstanceRepository
from bot_platform_service.persistence.repositories.bot_module_repository import BotModuleRepository
from bot_platform_service.persistence.repositories.bot_run_repository import BotRunRepository
from bot_platform_service.persistence.repositories.bot_signal_repository import BotSignalRepository
from bot_platform_service.persistence.repositories.market_data_event_repository import MarketDataEventRepository
from bot_platform_service.registry.module_resolver import PersistedBotModuleResolver


class AsyncEngineFactory(Protocol):
    """Factory boundary for creating async SQLAlchemy engines."""

    def __call__(self, database_url: str) -> AsyncEngine:
        """Return an async SQLAlchemy engine."""


@dataclass(frozen=True, slots=True)
class BotPlatformRepositories:
    """Repository bundle owned by the runtime composition root."""

    bot_modules: BotModuleRepository
    bot_instances: BotInstanceRepository
    bot_audit_events: BotAuditEventRepository
    bot_runs: BotRunRepository
    bot_signals: BotSignalRepository
    market_data_events: MarketDataEventRepository | None = None


@dataclass(slots=True)
class BotPlatformRuntimeContainer:
    """Runtime dependency graph for Bot Platform."""

    settings: BotPlatformSettings
    engine: AsyncEngine
    connection: AsyncConnection
    repositories: BotPlatformRepositories
    admin_metadata_service: AdminMetadataService
    admin_instance_service: AdminBotInstanceService
    config_validation_service: BotConfigValidationService
    lifecycle_service: BotInstanceLifecycleService
    manual_run_service: ManualBotRunService
    metrics: InMemoryMetricsRecorder
    event_data_cleanup_service: EventDataCleanupService | None = None
    runner_started: bool = False
    _closed: bool = False

    async def close(self) -> None:
        """Close runtime resources without starting or stopping any runner loop."""

        if self._closed:
            return
        await self.connection.close()
        await self.engine.dispose()
        self._closed = True


async def build_runtime_container(
    settings: BotPlatformSettings | None = None,
    *,
    engine_factory: AsyncEngineFactory = create_async_engine,
) -> BotPlatformRuntimeContainer:
    """Build the Bot Platform runtime graph without starting bot execution."""

    resolved_settings = settings or BotPlatformSettings.from_env()
    engine = engine_factory(resolved_settings.database.url)
    connection = await engine.connect()
    bot_module_repository = BotModuleRepository(connection)
    bot_instance_repository = BotInstanceRepository(connection)
    bot_audit_event_repository = BotAuditEventRepository(connection)
    bot_run_repository = BotRunRepository(connection)
    bot_signal_repository = BotSignalRepository(connection)
    market_data_event_repository = MarketDataEventRepository(connection)
    repositories = BotPlatformRepositories(
        bot_modules=bot_module_repository,
        bot_instances=bot_instance_repository,
        bot_audit_events=bot_audit_event_repository,
        bot_runs=bot_run_repository,
        bot_signals=bot_signal_repository,
        market_data_events=market_data_event_repository,
    )
    metrics = InMemoryMetricsRecorder()
    module_resolver = PersistedBotModuleResolver(bot_module_repository)
    runtime_capabilities = RuntimeCapabilities(
        market_data=MarketDataHttpSnapshotClient(base_url=resolved_settings.market_data.base_url),
        signal_publisher=_NoopSignalPublisher(),
        state_store=PersistentRuntimeStateStore(bot_instance_repository),
        notification_publisher=_NoopNotificationPublisher(),
        secret_provider=_NoopSecretProvider(),
        logger=_NoopStructuredLogger(),
        metrics=metrics,
        clock=_SystemClock(),
    )
    signal_event_publisher = (
        build_redis_signal_event_publisher(
            redis_url=resolved_settings.signal_events.redis_url,
            stream_name=resolved_settings.signal_events.stream_name,
            max_retries=resolved_settings.signal_events.max_retries,
            retry_backoff_seconds=resolved_settings.signal_events.retry_backoff_seconds,
        )
        if resolved_settings.signal_events.enabled
        else DisabledSignalEventPublisher()
    )
    return BotPlatformRuntimeContainer(
        settings=resolved_settings,
        engine=engine,
        connection=connection,
        repositories=repositories,
        admin_metadata_service=AdminMetadataService(repository=bot_module_repository),
        admin_instance_service=AdminBotInstanceService(repository=bot_instance_repository),
        config_validation_service=BotConfigValidationService(repository=bot_module_repository),
        lifecycle_service=BotInstanceLifecycleService(
            instance_repository=bot_instance_repository,
            audit_repository=bot_audit_event_repository,
            module_resolver=module_resolver,
        ),
        manual_run_service=ManualBotRunService(
            instance_repository=bot_instance_repository,
            run_repository=bot_run_repository,
            signal_repository=bot_signal_repository,
            audit_repository=bot_audit_event_repository,
            signal_event_publisher=signal_event_publisher,
            module_resolver=module_resolver,
            runtime_capabilities=runtime_capabilities,
            market_data_source=resolved_settings.market_data.source,
        ),
        metrics=metrics,
        event_data_cleanup_service=EventDataCleanupService(
            repository=market_data_event_repository,
            idempotency_retention_days=resolved_settings.event_retention.idempotency_retention_days,
            audit_retention_days=resolved_settings.event_retention.audit_retention_days,
        ),
    )


class _NoopSignalPublisher:
    async def publish(self, signal) -> object:
        raise RuntimeError("Run-scoped signal persistence is configured outside runtime context fanout")


class PersistentRuntimeStateStore:
    """StateStore backed by `_bot_platform.bot_runtime_state`."""

    def __init__(self, repository: BotInstanceRepository) -> None:
        self.repository = repository

    async def load(self, *, instance_id: str, namespace: str, state_key: str) -> object | None:
        state = await self.repository.get_runtime_state(
            instance_id=instance_id,
            namespace=namespace,
            state_key=state_key,
        )
        return None if state is None else state.state_json

    async def save(self, *, instance_id: str, namespace: str, state_key: str, value: object) -> None:
        if not isinstance(value, Mapping):
            raise TypeError("runtime state value must be a JSON mapping")
        state_json = json.loads(canonical_json(value))
        state_hash = build_payload_hash(state_json)
        state_identity = {
            "instance_id": instance_id,
            "namespace": namespace,
            "state_key": state_key,
        }
        state_id = f"state_{build_payload_hash(state_identity)[:32]}"
        await self.repository.upsert_runtime_state(
            state_id=state_id,
            instance_id=instance_id,
            namespace=namespace,
            state_key=state_key,
            state_json=state_json,
            state_hash=state_hash,
        )


class _NoopNotificationPublisher:
    async def publish(self, *, instance_id: str, message_type: str, payload: object) -> None:
        return None


class _NoopSecretProvider:
    async def resolve(self, *, instance_id: str, secret_name: str) -> str:
        raise RuntimeError("Secret resolution is not configured")


class _NoopStructuredLogger:
    def info(self, event: str, **fields: object) -> None:
        return None

    def warning(self, event: str, **fields: object) -> None:
        return None

    def error(self, event: str, **fields: object) -> None:
        return None


class _SystemClock:
    def now(self) -> Any:
        return datetime.now(UTC)
