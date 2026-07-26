from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from bot_platform_service.application import BotRunOrchestrationService, BotRuntimeRecoveryService, CandleBatchReadyEvent, RuntimeCapabilities
from bot_platform_service.domain import (
    BotCandle,
    BotHealth,
    BotHealthStatus,
    BotInstanceConfig,
    BotMarketDataContext,
    BotMarketSnapshot,
    BotMode,
    BotRecoverableRunRecord,
    BotRunRequest,
    BotRunResult,
    BotRunStatus,
    BotSignalPublishResult,
    BotStartResult,
    BotStopResult,
    BotTriggerType,
    BotValidationResult,
)


class _InstanceRepository:
    def __init__(self, instances: tuple[BotInstanceConfig, ...]) -> None:
        self.instances = instances

    async def list_enabled_instances_for_snapshot(self, *, source: str, canonical_symbol: str, timeframe: str):
        del source
        return tuple(
            instance
            for instance in self.instances
            if canonical_symbol in instance.symbols and timeframe in instance.timeframes
        )


class _RunRepository:
    def __init__(self, recoverable: tuple[BotRecoverableRunRecord, ...] = ()) -> None:
        self.recoverable = recoverable
        self.keys: set[str] = set()
        self.created = 0
        self.completed: list[BotRunStatus] = []
        self.events: list[str] = []

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
        del run_id, instance_id, module_id, trigger_type, status, trigger_event_id, snapshot_id, correlation_id
        key = str(idempotency_key)
        if key in self.keys:
            return False
        self.keys.add(key)
        self.created += 1
        return True

    async def append_run_event(
        self,
        *,
        event_id: str,
        run_id: str,
        instance_id: str,
        module_id: str,
        event_type: str,
        payload_json: dict[str, object],
        correlation_id: str | None = None,
    ) -> bool:
        del event_id, run_id, instance_id, module_id, payload_json, correlation_id
        self.events.append(event_type)
        return True

    async def complete_run(
        self,
        *,
        run_id: str,
        status: BotRunStatus,
        error_code: str | None = None,
        error_message_redacted: str | None = None,
    ) -> bool:
        del run_id, error_code, error_message_redacted
        self.completed.append(status)
        return True

    async def find_stuck_runs(self, *, stale_before: datetime) -> tuple[str, ...]:
        del stale_before
        return ()

    async def list_recoverable_runs(self, *, stale_before: datetime) -> tuple[BotRecoverableRunRecord, ...]:
        del stale_before
        return self.recoverable


class _AuditRepository:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def append_audit_event(
        self,
        *,
        event_id: str,
        event_type: str,
        actor_type: str,
        actor_id: str,
        payload_json: dict[str, object],
        instance_id: str | None = None,
        module_id: str | None = None,
        correlation_id: str | None = None,
    ) -> bool:
        del event_id, actor_type, actor_id, payload_json, instance_id, module_id, correlation_id
        self.events.append(event_type)
        return True


class _Resolver:
    def __init__(self, failing_instances: set[str] | None = None) -> None:
        self.failing_instances = failing_instances or set()

    async def resolve(self, module_id: str):
        return _Module(module_id=module_id, failing_instances=self.failing_instances)


class _Module:
    def __init__(self, *, module_id: str, failing_instances: set[str]) -> None:
        self.module_id = module_id
        self.failing_instances = failing_instances

    async def validate_config(self, config: BotInstanceConfig) -> BotValidationResult:
        del config
        return BotValidationResult(valid=True)

    async def initialize(self, context) -> None:
        del context

    async def dry_run(self, request: BotRunRequest) -> BotRunResult:
        if request.instance_id in self.failing_instances:
            raise RuntimeError("planned failure")
        return BotRunResult(
            run_id=request.run_id,
            instance_id=request.instance_id,
            module_id=request.module_id,
            mode=request.mode,
            status=BotRunStatus.COMPLETE,
        )

    async def run_once(self, request: BotRunRequest) -> BotRunResult:
        return await self.dry_run(request)

    async def start(self, request) -> BotStartResult:
        return BotStartResult(accepted=True, instance_id=request.instance_id)

    async def stop(self, instance_id: str) -> BotStopResult:
        return BotStopResult(accepted=True, instance_id=instance_id)

    async def health(self, instance_id: str) -> BotHealth:
        return BotHealth(instance_id=instance_id, module_id=self.module_id, status=BotHealthStatus.HEALTHY)


class _MarketData:
    async def get_latest_complete_snapshot(self, **kwargs):
        return _snapshot(kwargs["timeframe"])

    async def build_context(self, **kwargs):
        return BotMarketDataContext(primary_snapshot=_snapshot(kwargs["primary_timeframe"]))


class _SignalPublisher:
    async def publish(self, signal) -> BotSignalPublishResult:
        return BotSignalPublishResult(accepted=True, signal_id=f"signal:{signal.signal_key}")


class _Noop:
    def __getattr__(self, name):
        def _method(*args, **kwargs):
            return None

        return _method


def test_production_readiness_load_duplicate_delivery_and_failure_isolation() -> None:
    instances = tuple(_config(f"instance-{index}") for index in range(20))
    run_repository = _RunRepository()
    service = _service(instances=instances, run_repository=run_repository, failing_instances={"instance-7"})

    first = asyncio.run(service.handle_candle_batch_ready(_event()))
    second = asyncio.run(service.handle_candle_batch_ready(_event()))

    assert len(first) == 20
    assert sum(result.status is BotRunStatus.FAILED for result in first) == 1
    assert sum(result.status is BotRunStatus.COMPLETE for result in first) == 19
    assert all(result.created is False for result in second)
    assert run_repository.created == 20
    assert run_repository.completed.count(BotRunStatus.FAILED) == 1
    assert run_repository.completed.count(BotRunStatus.COMPLETE) == 19


def test_production_readiness_restart_recovery_preserves_history_and_audit() -> None:
    run_repository = _RunRepository(
        recoverable=(
            BotRecoverableRunRecord(
                run_id="run-stale",
                instance_id="instance-1",
                module_id="fixture_bot",
                status=BotRunStatus.RUNNING,
            ),
        )
    )
    audit_repository = _AuditRepository()
    service = BotRuntimeRecoveryService(run_repository=run_repository, audit_repository=audit_repository)

    result = asyncio.run(
        service.recover_stale_running_runs(
            now=datetime(2026, 7, 14, 12, 0, tzinfo=UTC),
            timeout=timedelta(minutes=10),
        )
    )

    assert result[0].recovered is True
    assert run_repository.completed == [BotRunStatus.CANCELLED]
    assert "RECOVERED" in run_repository.events
    assert audit_repository.events == ["RUN_RECOVERED"]


def _service(*, instances: tuple[BotInstanceConfig, ...], run_repository: _RunRepository, failing_instances: set[str]) -> BotRunOrchestrationService:
    return BotRunOrchestrationService(
        instance_repository=_InstanceRepository(instances),
        run_repository=run_repository,
        module_resolver=_Resolver(failing_instances),
        runtime_capabilities=RuntimeCapabilities(
            market_data=_MarketData(),
            signal_publisher=_SignalPublisher(),
            state_store=_Noop(),
            notification_publisher=_Noop(),
            secret_provider=_Noop(),
            logger=_Noop(),
            metrics=_Noop(),
            clock=_Noop(),
        ),
    )


def _config(instance_id: str) -> BotInstanceConfig:
    return BotInstanceConfig(
        instance_id=instance_id,
        module_id="fixture_bot",
        mode=BotMode.DRY_RUN,
        symbols=("ETHUSDT",),
        timeframes=("1h", "4h"),
        config_schema_version=1,
    )


def _event() -> CandleBatchReadyEvent:
    return CandleBatchReadyEvent(
        event_id="event-1",
        snapshot_id="snapshot-1",
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        timeframe="1h",
    )


def _snapshot(timeframe: str) -> BotMarketSnapshot:
    now = datetime(2026, 7, 14, tzinfo=UTC)
    candle = BotCandle(
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        timeframe=timeframe,
        open_time=now,
        close_time=now,
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100.5"),
        volume=Decimal("1000"),
    )
    return BotMarketSnapshot(
        snapshot_id=f"snapshot-{timeframe}",
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        provider_symbol="ETHUSDT",
        timeframe=timeframe,
        last_closed_candle_time=now,
        lookback_start_time=now,
        lookback_end_time=now,
        completeness_status="COMPLETE",
        snapshot_version=1,
        data_hash="hash",
        candles=(candle,),
    )
