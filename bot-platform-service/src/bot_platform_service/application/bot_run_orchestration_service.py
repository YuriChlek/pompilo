from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from bot_platform_service.application.runtime_context_service import RuntimeCapabilities, RuntimeContextFactory, RuntimeContextRequest
from bot_platform_service.application.state_change_applier_service import RunScopedStateStore, StateChangeApplierService
from bot_platform_service.domain import (
    BotInstanceConfig,
    BotInstanceStatus,
    BotMarketDataContext,
    BotMarketSnapshot,
    BotMode,
    BotModule,
    BotPermission,
    BotRunRequest,
    BotRunResult,
    BotRunStatus,
    BotTriggerType,
    MarketDataSnapshotProvider,
    build_payload_hash,
)


class BotRunOrchestrationInstanceRepository(Protocol):
    """Instance reads required by run orchestration."""

    async def list_enabled_instances_for_snapshot(self, *, source: str, canonical_symbol: str, timeframe: str) -> tuple[BotInstanceConfig, ...]:
        """Return enabled instances eligible for one market snapshot."""


class BotRunOrchestrationRunRepository(Protocol):
    """Run persistence boundary required by scheduler and event routing."""

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
        """Create a run idempotently and return whether it is new."""

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
        """Append one run event."""

    async def complete_run(
        self,
        *,
        run_id: str,
        status: BotRunStatus,
        error_code: str | None = None,
        error_message_redacted: str | None = None,
    ) -> bool:
        """Mark a run terminal."""

    async def find_stuck_runs(self, *, stale_before: datetime) -> tuple[str, ...]:
        """Return running run ids older than the threshold."""


class BotModuleResolver(Protocol):
    """Resolve modules without importing concrete adapters in application layer."""

    async def resolve(self, module_id: str) -> BotModule | None:
        """Return a bot module by id."""


@dataclass(frozen=True, slots=True)
class CandleBatchReadyEvent:
    """Normalized market-data event consumed by Bot Platform."""

    event_id: str
    snapshot_id: str
    source: str
    canonical_symbol: str
    timeframe: str
    correlation_id: str | None = None


@dataclass(frozen=True, slots=True)
class PollingScheduleTick:
    """Scheduler tick for polling configured instances."""

    tick_id: str
    source: str = "binance_spot"
    correlation_id: str | None = None


@dataclass(frozen=True, slots=True)
class BotRunDispatchResult:
    """Outcome of one attempted instance dispatch."""

    instance_id: str
    module_id: str
    run_id: str
    created: bool
    status: BotRunStatus | None
    error_code: str | None = None


class BotRunOrchestrationService:
    """Route polling and market-data events to eligible bot instances."""

    def __init__(
        self,
        *,
        instance_repository: BotRunOrchestrationInstanceRepository,
        run_repository: BotRunOrchestrationRunRepository,
        module_resolver: BotModuleResolver,
        runtime_capabilities: RuntimeCapabilities,
        runtime_context_factory: RuntimeContextFactory | None = None,
        state_change_applier: StateChangeApplierService | None = None,
    ) -> None:
        self.instance_repository = instance_repository
        self.run_repository = run_repository
        self.module_resolver = module_resolver
        self.runtime_capabilities = runtime_capabilities
        self.runtime_context_factory = runtime_context_factory or RuntimeContextFactory()
        self.state_change_applier = state_change_applier or StateChangeApplierService()

    async def handle_candle_batch_ready(self, event: CandleBatchReadyEvent) -> tuple[BotRunDispatchResult, ...]:
        """Route one snapshot event to all eligible enabled instances."""

        instances = await self.instance_repository.list_enabled_instances_for_snapshot(
            source=event.source,
            canonical_symbol=event.canonical_symbol,
            timeframe=event.timeframe,
        )
        market_data_context = await _build_market_data_context(
            self.runtime_capabilities.market_data,
            source=event.source,
            canonical_symbol=event.canonical_symbol,
            primary_timeframe=event.timeframe,
            supporting_timeframes=tuple(
                dict.fromkeys(
                    timeframe
                    for config in instances
                    for timeframe in config.timeframes
                    if timeframe != event.timeframe
                )
            ),
        )
        results = []
        for config in instances:
            results.append(
                await self._dispatch_instance(
                    config,
                    trigger_type=BotTriggerType.EVENT,
                    trigger_event_id=event.event_id,
                    snapshot_id=event.snapshot_id,
                    idempotency_key=build_trigger_idempotency_key(
                        instance_id=config.instance_id,
                        trigger_type=BotTriggerType.EVENT,
                        trigger_id=event.event_id,
                        snapshot_id=event.snapshot_id,
                    ),
                    correlation_id=event.correlation_id,
                    market_data=market_data_context,
                )
            )
        return tuple(results)

    async def run_polling_tick(self, tick: PollingScheduleTick, instances: tuple[BotInstanceConfig, ...]) -> tuple[BotRunDispatchResult, ...]:
        """Run a scheduler tick for already selected polling instances."""

        results = []
        for config in instances:
            market_data = await _build_market_data_for_instance(
                self.runtime_capabilities.market_data,
                source=tick.source,
                config=config,
            )
            results.append(
                await self._dispatch_instance(
                    config,
                    trigger_type=BotTriggerType.SCHEDULER,
                    trigger_event_id=tick.tick_id,
                    snapshot_id=None,
                    idempotency_key=build_trigger_idempotency_key(
                        instance_id=config.instance_id,
                        trigger_type=BotTriggerType.SCHEDULER,
                        trigger_id=tick.tick_id,
                        snapshot_id=None,
                    ),
                    correlation_id=tick.correlation_id,
                    market_data=market_data,
                )
            )
        return tuple(results)

    async def detect_stuck_runs(self, *, now: datetime, timeout: timedelta) -> tuple[str, ...]:
        """Return running run ids older than the timeout threshold."""

        return await self.run_repository.find_stuck_runs(stale_before=now - timeout)

    async def _dispatch_instance(
        self,
        config: BotInstanceConfig,
        *,
        trigger_type: BotTriggerType,
        trigger_event_id: str | None,
        snapshot_id: str | None,
        idempotency_key: str,
        correlation_id: str | None,
        market_data: BotMarketDataContext | None,
    ) -> BotRunDispatchResult:
        run_id = build_run_id(idempotency_key)
        created = await self.run_repository.create_run(
            run_id=run_id,
            instance_id=config.instance_id,
            module_id=config.module_id,
            trigger_type=trigger_type,
            trigger_event_id=trigger_event_id,
            snapshot_id=snapshot_id,
            idempotency_key=idempotency_key,
            correlation_id=correlation_id,
        )
        if not created:
            return BotRunDispatchResult(config.instance_id, config.module_id, run_id, created=False, status=None)
        await self.run_repository.append_run_event(
            event_id=f"{run_id}:STARTED",
            run_id=run_id,
            instance_id=config.instance_id,
            module_id=config.module_id,
            event_type="STARTED",
            payload_json={"trigger_type": trigger_type.value, "snapshot_id": snapshot_id},
            correlation_id=correlation_id,
        )
        try:
            module = await self.module_resolver.resolve(config.module_id)
            if module is None:
                raise ValueError(f"Unknown bot module: {config.module_id}")
            state_store = RunScopedStateStore(self.runtime_capabilities.state_store)
            await module.initialize(
                self.runtime_context_factory.build(
                    RuntimeContextRequest(
                        instance_id=config.instance_id,
                        permissions=_permissions_for_mode(config.mode),
                        capabilities=RuntimeCapabilities(
                            market_data=self.runtime_capabilities.market_data,
                            signal_publisher=self.runtime_capabilities.signal_publisher,
                            state_store=state_store,
                            notification_publisher=self.runtime_capabilities.notification_publisher,
                            secret_provider=self.runtime_capabilities.secret_provider,
                            logger=self.runtime_capabilities.logger,
                            metrics=self.runtime_capabilities.metrics,
                            clock=self.runtime_capabilities.clock,
                        ),
                    )
                )
            )
            run_result = await _run_module(module, config=config, run_id=run_id, trigger_type=trigger_type, market_data=market_data, correlation_id=correlation_id)
            signal_publish_count = await _persist_signals(
                run_result,
                signal_publisher=self.runtime_capabilities.signal_publisher,
            )
            notification_publish_count = await _persist_notifications(
                run_result,
                notification_publisher=self.runtime_capabilities.notification_publisher,
            )
            state_application = await self.state_change_applier.apply(
                run_result,
                state_store=state_store,
                already_applied_keys=state_store.saved_keys,
            )
            await self.run_repository.complete_run(
                run_id=run_id,
                status=run_result.status,
                error_code=run_result.error_code,
                error_message_redacted=run_result.error_message_redacted,
            )
            await self.run_repository.append_run_event(
                event_id=f"{run_id}:COMPLETED",
                run_id=run_id,
                instance_id=config.instance_id,
                module_id=config.module_id,
                event_type="COMPLETED",
                payload_json={
                    "status": run_result.status.value,
                    "signal_count": len(run_result.signals),
                    "signal_publish_count": signal_publish_count,
                    "notification_count": len(run_result.notifications),
                    "notification_publish_count": notification_publish_count,
                    "diagnostics": dict(run_result.diagnostics),
                    "state_changes_applied": state_application.applied,
                    "state_changes_skipped": state_application.skipped,
                },
                correlation_id=correlation_id,
            )
            return BotRunDispatchResult(config.instance_id, config.module_id, run_id, True, run_result.status, run_result.error_code)
        except Exception as exc:
            await self.run_repository.complete_run(
                run_id=run_id,
                status=BotRunStatus.FAILED,
                error_code="INSTANCE_RUN_FAILED",
                error_message_redacted=type(exc).__name__,
            )
            await self.run_repository.append_run_event(
                event_id=f"{run_id}:FAILED",
                run_id=run_id,
                instance_id=config.instance_id,
                module_id=config.module_id,
                event_type="FAILED",
                payload_json={"error_type": type(exc).__name__},
                correlation_id=correlation_id,
            )
            return BotRunDispatchResult(config.instance_id, config.module_id, run_id, True, BotRunStatus.FAILED, "INSTANCE_RUN_FAILED")


def build_trigger_idempotency_key(
    *,
    instance_id: str,
    trigger_type: BotTriggerType,
    trigger_id: str,
    snapshot_id: str | None,
) -> str:
    """Build deterministic run idempotency key for one instance trigger."""

    return build_payload_hash(
        {
            "instance_id": instance_id,
            "trigger_type": trigger_type.value,
            "trigger_id": trigger_id,
            "snapshot_id": snapshot_id,
        }
    )


def build_run_id(idempotency_key: str) -> str:
    """Build stable run id derived from the idempotency key."""

    return f"run_{idempotency_key[:32]}"


async def _run_module(
    module: BotModule,
    *,
    config: BotInstanceConfig,
    run_id: str,
    trigger_type: BotTriggerType,
    market_data: BotMarketDataContext | None,
    correlation_id: str | None,
) -> BotRunResult:
    request = BotRunRequest(
        run_id=run_id,
        instance_id=config.instance_id,
        module_id=config.module_id,
        mode=config.mode,
        trigger_type=trigger_type,
        market_data=market_data,
        correlation_id=correlation_id,
    )
    if config.mode is BotMode.DRY_RUN:
        return await module.dry_run(request)
    return await module.run_once(request)


async def _load_supporting_snapshots(
    snapshot_provider: MarketDataSnapshotProvider,
    *,
    source: str,
    canonical_symbol: str,
    timeframes: tuple[str, ...],
) -> tuple[BotMarketSnapshot, ...]:
    snapshots = []
    for timeframe in timeframes:
        snapshots.append(
            await snapshot_provider.get_latest_complete_snapshot(
                source=source,
                canonical_symbol=canonical_symbol,
                timeframe=timeframe,
            )
        )
    return tuple(snapshots)


async def _build_market_data_for_instance(
    snapshot_provider: MarketDataSnapshotProvider,
    *,
    source: str,
    config: BotInstanceConfig,
) -> BotMarketDataContext | None:
    if not config.symbols or not config.timeframes:
        return None
    return await _build_market_data_context(
        snapshot_provider,
        source=source,
        canonical_symbol=config.symbols[0],
        primary_timeframe=config.timeframes[0],
        supporting_timeframes=config.timeframes[1:],
    )


async def _build_market_data_context(
    snapshot_provider: MarketDataSnapshotProvider,
    *,
    source: str,
    canonical_symbol: str,
    primary_timeframe: str,
    supporting_timeframes: tuple[str, ...],
) -> BotMarketDataContext:
    try:
        return await snapshot_provider.build_context(
            source=source,
            canonical_symbol=canonical_symbol,
            primary_timeframe=primary_timeframe,
            supporting_timeframes=supporting_timeframes,
        )
    except AttributeError:
        primary_snapshot = await snapshot_provider.get_latest_complete_snapshot(
            source=source,
            canonical_symbol=canonical_symbol,
            timeframe=primary_timeframe,
        )
        supporting_snapshots = await _load_supporting_snapshots(
            snapshot_provider,
            source=source,
            canonical_symbol=canonical_symbol,
            timeframes=supporting_timeframes,
        )
        return BotMarketDataContext(primary_snapshot=primary_snapshot, supporting_snapshots=supporting_snapshots)


async def _persist_signals(
    result: BotRunResult,
    *,
    signal_publisher,
) -> int:
    if result.status is not BotRunStatus.COMPLETE or result.mode is not BotMode.SIGNAL_ONLY:
        return 0
    published = 0
    for signal in result.signals:
        await signal_publisher.publish(signal)
        published += 1
    return published


async def _persist_notifications(
    result: BotRunResult,
    *,
    notification_publisher,
) -> int:
    if result.status is not BotRunStatus.COMPLETE or result.mode is not BotMode.NOTIFICATION_ONLY:
        return 0
    published = 0
    for notification in result.notifications:
        await notification_publisher.publish(
            instance_id=notification.instance_id,
            message_type=notification.message_type,
            payload={
                "notification_id": notification.notification_id,
                "module_id": notification.module_id,
                "status": notification.status.value,
                "channel": notification.channel,
                "error_code": notification.error_code,
            },
        )
        published += 1
    return published


def _permissions_for_mode(mode: BotMode) -> frozenset[BotPermission]:
    permissions = {BotPermission.READ_MARKET_DATA, BotPermission.READ_STATE, BotPermission.WRITE_STATE}
    if mode is BotMode.NOTIFICATION_ONLY:
        permissions.add(BotPermission.SEND_NOTIFICATIONS)
    if mode is BotMode.SIGNAL_ONLY:
        permissions.add(BotPermission.PUBLISH_SIGNALS)
    return frozenset(permissions)


def is_instance_eligible_for_snapshot(config: BotInstanceConfig, *, canonical_symbol: str, timeframe: str) -> bool:
    """Return whether one enabled config should run for a snapshot."""

    return canonical_symbol.upper() in {symbol.upper() for symbol in config.symbols} and timeframe in set(config.timeframes)
