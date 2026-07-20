from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from bot_platform_service.application.runtime_context_service import RuntimeCapabilities, RuntimeContextFactory, RuntimeContextRequest
from bot_platform_service.domain import (
    BotMarketDataContext,
    BotInstanceConfig,
    BotInstanceStatus,
    BotMode,
    BotModule,
    BotPermission,
    BotRunRequest,
    BotRunStatus,
    BotSignal,
    BotSignalPublishStatus,
    BotTriggerType,
    build_payload_hash,
)


class ManualRunInstanceRepository(Protocol):
    """Instance reads required by manual run orchestration."""

    async def get_instance_config(self, instance_id: str) -> BotInstanceConfig | None:
        """Return the active instance config."""

    async def get_instance_status(self, instance_id: str) -> BotInstanceStatus | None:
        """Return current instance status."""


class ManualRunRepository(Protocol):
    """Run persistence and lock boundary required by manual runs."""

    async def acquire_instance_run_lock(self, *, instance_id: str) -> bool:
        """Acquire a transaction-scoped lock for one instance run."""

    async def create_run(
        self,
        *,
        run_id: str,
        instance_id: str,
        module_id: str,
        trigger_type: BotTriggerType,
        status: BotRunStatus = BotRunStatus.RUNNING,
        trigger_event_id: str | None = None,
        snapshot_id: str | None = None,
        idempotency_key: str | None = None,
        correlation_id: str | None = None,
    ) -> bool:
        """Create a run and return whether it was inserted."""

    async def complete_run(
        self,
        *,
        run_id: str,
        status: BotRunStatus,
        error_code: str | None = None,
        error_message_redacted: str | None = None,
    ) -> bool:
        """Complete one run."""

    async def append_run_event(
        self,
        *,
        event_id: str,
        run_id: str,
        instance_id: str,
        module_id: str,
        event_type: str,
        payload_json: Mapping[str, object],
        correlation_id: str | None = None,
    ) -> bool:
        """Append one duplicate-safe run event."""


class ManualRunSignalRepository(Protocol):
    """Signal persistence boundary for manual run results."""

    async def publish_signal(
        self,
        *,
        signal_id: str,
        run_id: str,
        signal: BotSignal,
        status: BotSignalPublishStatus = BotSignalPublishStatus.PUBLISHED,
        correlation_id: str | None = None,
    ) -> str:
        """Persist one signal idempotently and return its signal id."""


class ManualRunAuditRepository(Protocol):
    """Audit persistence boundary for manual run signal results."""

    async def append_audit_event(
        self,
        *,
        event_id: str,
        event_type: str,
        actor_type: str,
        actor_id: str,
        payload_json: Mapping[str, object],
        instance_id: str | None = None,
        module_id: str | None = None,
        correlation_id: str | None = None,
    ) -> bool:
        """Append one duplicate-safe audit event."""


class SignalEventPublisher(Protocol):
    """Downstream signal event publishing boundary."""

    async def publish_persisted_signal(
        self,
        *,
        signal_id: str,
        run_id: str,
        signal: BotSignal,
        correlation_id: str | None = None,
    ) -> bool:
        """Publish one already-persisted signal event duplicate-safely."""


class ManualRunModuleResolver(Protocol):
    """Resolve platform-native adapters for manual run execution."""

    async def resolve(self, module_id: str) -> BotModule | None:
        """Return a bot module by id."""


@dataclass(frozen=True, slots=True)
class ManualRunCommand:
    """Manual run request."""

    instance_id: str
    idempotency_key: str | None = None
    correlation_id: str | None = None


@dataclass(frozen=True, slots=True)
class EventRunCommand:
    """Market-data event triggered run request."""

    instance_id: str
    source: str
    canonical_symbol: str
    timeframe: str
    snapshot_id: str
    event_id: str
    idempotency_key: str
    correlation_id: str | None = None


@dataclass(frozen=True, slots=True)
class ManualRunResult:
    """Manual run outcome returned to admin callers."""

    accepted: bool
    instance_id: str
    run_id: str | None = None
    status: BotRunStatus | None = None
    error_code: str | None = None
    duplicate: bool = False


class ManualBotRunService:
    """Run one bot instance through a controlled manual path without signal fanout."""

    def __init__(
        self,
        *,
        instance_repository: ManualRunInstanceRepository,
        run_repository: ManualRunRepository,
        signal_repository: ManualRunSignalRepository,
        audit_repository: ManualRunAuditRepository,
        signal_event_publisher: SignalEventPublisher,
        module_resolver: ManualRunModuleResolver,
        runtime_capabilities: RuntimeCapabilities,
        market_data_source: str,
        runtime_context_factory: RuntimeContextFactory | None = None,
    ) -> None:
        self.instance_repository = instance_repository
        self.run_repository = run_repository
        self.signal_repository = signal_repository
        self.audit_repository = audit_repository
        self.signal_event_publisher = signal_event_publisher
        self.module_resolver = module_resolver
        self.runtime_capabilities = runtime_capabilities
        self.market_data_source = market_data_source
        self.runtime_context_factory = runtime_context_factory or RuntimeContextFactory()

    async def run_instance(self, command: ManualRunCommand) -> ManualRunResult:
        """Create and execute one manual run for an enabled instance."""

        config = await self.instance_repository.get_instance_config(command.instance_id)
        if config is None:
            return ManualRunResult(False, command.instance_id, error_code="INSTANCE_NOT_FOUND")
        status = await self.instance_repository.get_instance_status(command.instance_id)
        if status is not BotInstanceStatus.ENABLED:
            return ManualRunResult(False, command.instance_id, status=None, error_code="INSTANCE_NOT_ENABLED")
        locked = await self.run_repository.acquire_instance_run_lock(instance_id=command.instance_id)
        if not locked:
            return ManualRunResult(False, command.instance_id, error_code="INSTANCE_RUN_LOCKED", duplicate=True)

        idempotency_key = command.idempotency_key or f"manual:{command.instance_id}:{uuid4()}"
        run_id = _manual_run_id(instance_id=command.instance_id, idempotency_key=idempotency_key)
        try:
            market_data = await self.runtime_capabilities.market_data.build_context(
                source=self.market_data_source,
                canonical_symbol=config.symbols[0],
                primary_timeframe=config.timeframes[0],
                supporting_timeframes=config.timeframes[1:],
            )
        except Exception as exc:
            error_code = getattr(exc, "error_code", "MARKET_DATA_SNAPSHOT_ERROR")
            return ManualRunResult(False, command.instance_id, run_id=run_id, error_code=str(error_code))

        created = await self.run_repository.create_run(
            run_id=run_id,
            instance_id=config.instance_id,
            module_id=config.module_id,
            trigger_type=BotTriggerType.MANUAL,
            snapshot_id=market_data.primary_snapshot.snapshot_id,
            idempotency_key=idempotency_key,
            correlation_id=command.correlation_id,
        )
        if not created:
            return ManualRunResult(False, command.instance_id, run_id=run_id, error_code="DUPLICATE_RUN", duplicate=True)
        await self.run_repository.append_run_event(
            event_id=f"{run_id}:STARTED",
            run_id=run_id,
            instance_id=config.instance_id,
            module_id=config.module_id,
            event_type="STARTED",
            payload_json={"trigger_type": BotTriggerType.MANUAL.value, "snapshot_id": market_data.primary_snapshot.snapshot_id},
            correlation_id=command.correlation_id,
        )

        module = await self.module_resolver.resolve(config.module_id)
        if module is None:
            await self.run_repository.complete_run(
                run_id=run_id,
                status=BotRunStatus.FAILED,
                error_code="MODULE_NOT_FOUND",
                error_message_redacted="module not found",
            )
            await self._append_failed_event(
                run_id=run_id,
                config=config,
                error_code="MODULE_NOT_FOUND",
                correlation_id=command.correlation_id,
            )
            return ManualRunResult(False, command.instance_id, run_id=run_id, status=BotRunStatus.FAILED, error_code="MODULE_NOT_FOUND")

        try:
            await module.initialize(
                self.runtime_context_factory.build(
                    RuntimeContextRequest(
                        instance_id=config.instance_id,
                        permissions=_permissions_for_mode(config.mode),
                        capabilities=self.runtime_capabilities,
                    )
                )
            )
            result = await module.run_once(
                BotRunRequest(
                    run_id=run_id,
                    instance_id=config.instance_id,
                    module_id=config.module_id,
                    mode=config.mode,
                    trigger_type=BotTriggerType.MANUAL,
                    market_data=market_data,
                    correlation_id=command.correlation_id,
                )
            )
            invalid_signal = _first_invalid_signal(result.signals, config=config, snapshot_id=market_data.primary_snapshot.snapshot_id)
            if invalid_signal is not None:
                await self.run_repository.complete_run(
                    run_id=run_id,
                    status=BotRunStatus.FAILED,
                    error_code="INVALID_SIGNAL_CONTRACT",
                    error_message_redacted=invalid_signal,
                )
                await self.run_repository.append_run_event(
                    event_id=f"{run_id}:INVALID_SIGNAL",
                    run_id=run_id,
                    instance_id=config.instance_id,
                    module_id=config.module_id,
                    event_type="INVALID_SIGNAL_REJECTED",
                    payload_json={"error_code": "INVALID_SIGNAL_CONTRACT", "reason": invalid_signal},
                    correlation_id=command.correlation_id,
                )
                await self.audit_repository.append_audit_event(
                    event_id=f"{run_id}:INVALID_SIGNAL_CONTRACT",
                    event_type="INVALID_SIGNAL_REJECTED",
                    actor_type="bot_platform",
                    actor_id="manual_run",
                    instance_id=config.instance_id,
                    module_id=config.module_id,
                    payload_json={"run_id": run_id, "reason": invalid_signal},
                    correlation_id=command.correlation_id,
                )
                return ManualRunResult(
                    False,
                    command.instance_id,
                    run_id=run_id,
                    status=BotRunStatus.FAILED,
                    error_code="INVALID_SIGNAL_CONTRACT",
                )
            persisted_signal_count = 0
            for signal in result.signals:
                await self._persist_signal(run_id=run_id, signal=signal, correlation_id=command.correlation_id)
                persisted_signal_count += 1
            await self.run_repository.complete_run(
                run_id=run_id,
                status=result.status,
                error_code=result.error_code,
                error_message_redacted=result.error_message_redacted,
            )
            await self.run_repository.append_run_event(
                event_id=f"{run_id}:COMPLETED",
                run_id=run_id,
                instance_id=config.instance_id,
                module_id=config.module_id,
                event_type="COMPLETED",
                payload_json={
                    "status": result.status.value,
                    "signal_count": len(result.signals),
                    "persisted_signal_count": persisted_signal_count,
                    "diagnostics": dict(result.diagnostics),
                },
                correlation_id=command.correlation_id,
            )
            return ManualRunResult(True, command.instance_id, run_id=run_id, status=result.status, error_code=result.error_code)
        except Exception as exc:
            await self.run_repository.complete_run(
                run_id=run_id,
                status=BotRunStatus.FAILED,
                error_code="INSTANCE_RUN_FAILED",
                error_message_redacted=type(exc).__name__,
            )
            await self._append_failed_event(
                run_id=run_id,
                config=config,
                error_code="INSTANCE_RUN_FAILED",
                correlation_id=command.correlation_id,
            )
            return ManualRunResult(False, command.instance_id, run_id=run_id, status=BotRunStatus.FAILED, error_code="INSTANCE_RUN_FAILED")

    async def run_event_instance(self, command: EventRunCommand) -> ManualRunResult:
        """Create and execute one signal-only run triggered by a market-data event."""

        config = await self.instance_repository.get_instance_config(command.instance_id)
        if config is None:
            return ManualRunResult(False, command.instance_id, error_code="INSTANCE_NOT_FOUND")
        if config.mode is not BotMode.SIGNAL_ONLY:
            return ManualRunResult(False, command.instance_id, error_code="INSTANCE_MODE_NOT_SIGNAL_ONLY")
        status = await self.instance_repository.get_instance_status(command.instance_id)
        if status is not BotInstanceStatus.ENABLED:
            return ManualRunResult(False, command.instance_id, status=None, error_code="INSTANCE_NOT_ENABLED")
        locked = await self.run_repository.acquire_instance_run_lock(instance_id=command.instance_id)
        if not locked:
            return ManualRunResult(False, command.instance_id, error_code="INSTANCE_RUN_LOCKED", duplicate=True)

        run_id = _manual_run_id(instance_id=command.instance_id, idempotency_key=command.idempotency_key)
        try:
            market_data = await _build_event_market_data_context(
                self.runtime_capabilities.market_data,
                source=command.source,
                canonical_symbol=command.canonical_symbol,
                primary_timeframe=command.timeframe,
                snapshot_id=command.snapshot_id,
                supporting_timeframes=tuple(timeframe for timeframe in config.timeframes if timeframe != command.timeframe),
            )
        except Exception as exc:
            error_code = getattr(exc, "error_code", "MARKET_DATA_SNAPSHOT_ERROR")
            return ManualRunResult(False, command.instance_id, run_id=run_id, error_code=str(error_code))

        created = await self.run_repository.create_run(
            run_id=run_id,
            instance_id=config.instance_id,
            module_id=config.module_id,
            trigger_type=BotTriggerType.EVENT,
            trigger_event_id=command.event_id,
            snapshot_id=command.snapshot_id,
            idempotency_key=command.idempotency_key,
            correlation_id=command.correlation_id,
        )
        if not created:
            return ManualRunResult(False, command.instance_id, run_id=run_id, error_code="DUPLICATE_RUN", duplicate=True)
        await self.run_repository.append_run_event(
            event_id=f"{run_id}:STARTED",
            run_id=run_id,
            instance_id=config.instance_id,
            module_id=config.module_id,
            event_type="STARTED",
            payload_json={"trigger_type": BotTriggerType.EVENT.value, "snapshot_id": command.snapshot_id},
            correlation_id=command.correlation_id,
        )

        module = await self.module_resolver.resolve(config.module_id)
        if module is None:
            await self.run_repository.complete_run(
                run_id=run_id,
                status=BotRunStatus.FAILED,
                error_code="MODULE_NOT_FOUND",
                error_message_redacted="module not found",
            )
            await self._append_failed_event(
                run_id=run_id,
                config=config,
                error_code="MODULE_NOT_FOUND",
                correlation_id=command.correlation_id,
            )
            return ManualRunResult(False, command.instance_id, run_id=run_id, status=BotRunStatus.FAILED, error_code="MODULE_NOT_FOUND")

        try:
            await module.initialize(
                self.runtime_context_factory.build(
                    RuntimeContextRequest(
                        instance_id=config.instance_id,
                        permissions=_permissions_for_mode(config.mode),
                        capabilities=self.runtime_capabilities,
                    )
                )
            )
            result = await module.run_once(
                BotRunRequest(
                    run_id=run_id,
                    instance_id=config.instance_id,
                    module_id=config.module_id,
                    mode=config.mode,
                    trigger_type=BotTriggerType.EVENT,
                    market_data=market_data,
                    correlation_id=command.correlation_id,
                )
            )
            invalid_signal = _first_invalid_signal(result.signals, config=config, snapshot_id=command.snapshot_id)
            if invalid_signal is not None:
                await self.run_repository.complete_run(
                    run_id=run_id,
                    status=BotRunStatus.FAILED,
                    error_code="INVALID_SIGNAL_CONTRACT",
                    error_message_redacted=invalid_signal,
                )
                await self.run_repository.append_run_event(
                    event_id=f"{run_id}:INVALID_SIGNAL",
                    run_id=run_id,
                    instance_id=config.instance_id,
                    module_id=config.module_id,
                    event_type="INVALID_SIGNAL_REJECTED",
                    payload_json={"error_code": "INVALID_SIGNAL_CONTRACT", "reason": invalid_signal},
                    correlation_id=command.correlation_id,
                )
                await self.audit_repository.append_audit_event(
                    event_id=f"{run_id}:INVALID_SIGNAL_CONTRACT",
                    event_type="INVALID_SIGNAL_REJECTED",
                    actor_type="bot_platform",
                    actor_id="event_run",
                    instance_id=config.instance_id,
                    module_id=config.module_id,
                    payload_json={"run_id": run_id, "reason": invalid_signal},
                    correlation_id=command.correlation_id,
                )
                return ManualRunResult(
                    False,
                    command.instance_id,
                    run_id=run_id,
                    status=BotRunStatus.FAILED,
                    error_code="INVALID_SIGNAL_CONTRACT",
                )
            persisted_signal_count = 0
            for signal in result.signals:
                await self._persist_signal(
                    run_id=run_id,
                    signal=signal,
                    correlation_id=command.correlation_id,
                    actor_id="event_run",
                )
                persisted_signal_count += 1
            await self.run_repository.complete_run(
                run_id=run_id,
                status=result.status,
                error_code=result.error_code,
                error_message_redacted=result.error_message_redacted,
            )
            await self.run_repository.append_run_event(
                event_id=f"{run_id}:COMPLETED",
                run_id=run_id,
                instance_id=config.instance_id,
                module_id=config.module_id,
                event_type="COMPLETED",
                payload_json={
                    "status": result.status.value,
                    "signal_count": len(result.signals),
                    "persisted_signal_count": persisted_signal_count,
                    "diagnostics": dict(result.diagnostics),
                },
                correlation_id=command.correlation_id,
            )
            return ManualRunResult(True, command.instance_id, run_id=run_id, status=result.status, error_code=result.error_code)
        except Exception as exc:
            await self.run_repository.complete_run(
                run_id=run_id,
                status=BotRunStatus.FAILED,
                error_code="INSTANCE_RUN_FAILED",
                error_message_redacted=type(exc).__name__,
            )
            await self._append_failed_event(
                run_id=run_id,
                config=config,
                error_code="INSTANCE_RUN_FAILED",
                correlation_id=command.correlation_id,
            )
            return ManualRunResult(False, command.instance_id, run_id=run_id, status=BotRunStatus.FAILED, error_code="INSTANCE_RUN_FAILED")

    async def _persist_signal(
        self,
        *,
        run_id: str,
        signal: BotSignal,
        correlation_id: str | None,
        actor_id: str = "manual_run",
    ) -> str:
        signal_id = _signal_id(signal)
        persisted_signal_id = await self.signal_repository.publish_signal(
            signal_id=signal_id,
            run_id=run_id,
            signal=signal,
            status=BotSignalPublishStatus.PUBLISHED,
            correlation_id=correlation_id,
        )
        await self.audit_repository.append_audit_event(
            event_id=f"signal_persisted:{persisted_signal_id}",
            event_type="SIGNAL_PERSISTED",
            actor_type="bot_platform",
            actor_id=actor_id,
            instance_id=signal.instance_id,
            module_id=signal.module_id,
            payload_json={
                "signal_id": persisted_signal_id,
                "signal_key": signal.signal_key,
                "run_id": run_id,
                "symbol": signal.symbol,
                "timeframe": signal.timeframe,
                "snapshot_id": signal.snapshot_id,
                "signal_type": signal.signal_type.value,
                "side": signal.side.value if signal.side is not None else None,
                "payload_schema": signal.payload_schema,
                "payload_schema_version": signal.payload_schema_version,
                "payload_hash": signal.payload_hash,
                "boundary": "execution_service_reads_signals_only",
            },
            correlation_id=correlation_id,
        )
        try:
            published = await self.signal_event_publisher.publish_persisted_signal(
                signal_id=persisted_signal_id,
                run_id=run_id,
                signal=signal,
                correlation_id=correlation_id,
            )
            self.runtime_capabilities.metrics.increment(
                "bot_platform_signal_events_published_total",
                tags={"published": str(published).lower()},
            )
            self.runtime_capabilities.logger.info(
                "signal_event_publish_attempted",
                signal_id=persisted_signal_id,
                run_id=run_id,
                published=published,
            )
        except Exception as exc:
            self.runtime_capabilities.metrics.increment(
                "bot_platform_signal_event_publish_failures_total",
                tags={"error_type": type(exc).__name__},
            )
            self.runtime_capabilities.logger.warning(
                "signal_event_publish_failed",
                signal_id=persisted_signal_id,
                run_id=run_id,
                error_type=type(exc).__name__,
            )
        return persisted_signal_id

    async def _append_failed_event(
        self,
        *,
        run_id: str,
        config: BotInstanceConfig,
        error_code: str,
        correlation_id: str | None,
    ) -> None:
        await self.run_repository.append_run_event(
            event_id=f"{run_id}:FAILED",
            run_id=run_id,
            instance_id=config.instance_id,
            module_id=config.module_id,
            event_type="FAILED",
            payload_json={"error_code": error_code},
            correlation_id=correlation_id,
        )


def _manual_run_id(*, instance_id: str, idempotency_key: str) -> str:
    return f"run_{build_payload_hash({'instance_id': instance_id, 'idempotency_key': idempotency_key})[:32]}"


async def _build_event_market_data_context(
    snapshot_provider,
    *,
    source: str,
    canonical_symbol: str,
    primary_timeframe: str,
    snapshot_id: str,
    supporting_timeframes: tuple[str, ...],
) -> BotMarketDataContext:
    if hasattr(snapshot_provider, "get_snapshot"):
        primary_snapshot = await snapshot_provider.get_snapshot(snapshot_id=snapshot_id)
    else:
        primary_snapshot = await snapshot_provider.get_latest_complete_snapshot(
            source=source,
            canonical_symbol=canonical_symbol,
            timeframe=primary_timeframe,
        )
    if primary_snapshot.snapshot_id != snapshot_id:
        raise ValueError("event snapshot_id does not match loaded market snapshot")
    supporting_snapshots = tuple(
        [
            await snapshot_provider.get_latest_complete_snapshot(
                source=source,
                canonical_symbol=canonical_symbol,
                timeframe=timeframe,
            )
            for timeframe in supporting_timeframes
        ]
    )
    return BotMarketDataContext(primary_snapshot=primary_snapshot, supporting_snapshots=supporting_snapshots)


def _signal_id(signal: BotSignal) -> str:
    return f"sig_{build_payload_hash({'signal_key': signal.signal_key})[:32]}"


def _first_invalid_signal(signals: tuple[object, ...], *, config: BotInstanceConfig, snapshot_id: str) -> str | None:
    for signal in signals:
        reason = _invalid_signal_reason(signal, config=config, snapshot_id=snapshot_id)
        if reason is not None:
            return reason
    return None


def _invalid_signal_reason(signal: object, *, config: BotInstanceConfig, snapshot_id: str) -> str | None:
    if not isinstance(signal, BotSignal):
        return "signals must contain BotSignal items"
    if signal.instance_id != config.instance_id:
        return "signal instance_id must match run instance"
    if signal.module_id != config.module_id:
        return "signal module_id must match run module"
    if signal.snapshot_id != snapshot_id:
        return "signal snapshot_id must match run snapshot"
    if len(signal.signal_key) != 64 or signal.signal_key.lower() != signal.signal_key:
        return "signal_key must be lowercase sha256 hex"
    if len(signal.payload_hash) != 64 or signal.payload_hash.lower() != signal.payload_hash:
        return "payload_hash must be lowercase sha256 hex"
    if not signal.payload_schema:
        return "payload_schema must not be empty"
    if signal.payload_schema_version <= 0:
        return "payload_schema_version must be positive"
    return None


def _permissions_for_mode(mode: BotMode) -> frozenset[BotPermission]:
    permissions = {BotPermission.READ_MARKET_DATA, BotPermission.READ_STATE, BotPermission.WRITE_STATE}
    if mode is BotMode.NOTIFICATION_ONLY:
        permissions.add(BotPermission.SEND_NOTIFICATIONS)
    if mode is BotMode.SIGNAL_ONLY:
        permissions.add(BotPermission.PUBLISH_SIGNALS)
    return frozenset(permissions)
