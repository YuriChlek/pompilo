from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import MappingProxyType

from bot_platform_service.application import (
    BotRunOrchestrationService,
    CandleBatchReadyEvent,
    RuntimeCapabilities,
    build_trigger_idempotency_key,
    is_instance_eligible_for_snapshot,
)
from bot_platform_service.domain import (
    BotCandle,
    BotHealth,
    BotHealthStatus,
    BotInstanceConfig,
    BotMarketDataContext,
    BotMarketSnapshot,
    BotMode,
    BotNotificationStatus,
    BotRunRequest,
    BotRunResult,
    BotRunStatus,
    BotSignal,
    BotSignalPublishResult,
    BotSignalSide,
    BotSignalType,
    BotStartResult,
    BotStopResult,
    BotTriggerType,
    BotValidationResult,
)


class _InstanceRepository:
    def __init__(self, instances: tuple[BotInstanceConfig, ...]) -> None:
        self.instances = instances

    async def list_enabled_instances_for_snapshot(self, *, source: str, canonical_symbol: str, timeframe: str):
        return tuple(
            instance
            for instance in self.instances
            if is_instance_eligible_for_snapshot(instance, canonical_symbol=canonical_symbol, timeframe=timeframe)
        )


class _RunRepository:
    def __init__(self) -> None:
        self.created_keys: set[str] = set()
        self.created_runs: list[tuple[str, str]] = []
        self.completed: dict[str, BotRunStatus] = {}
        self.events: list[tuple[str, str]] = []

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
        del module_id, trigger_type, status, trigger_event_id, snapshot_id, correlation_id
        if idempotency_key in self.created_keys:
            return False
        self.created_keys.add(str(idempotency_key))
        self.created_runs.append((run_id, instance_id))
        return True

    async def append_run_event(
        self,
        *,
        event_id: str,
        run_id: str,
        instance_id: str,
        module_id: str,
        event_type: str,
        payload_json: MappingProxyType[str, object] | dict[str, object],
        correlation_id: str | None = None,
    ) -> bool:
        del instance_id, module_id, payload_json, correlation_id
        self.events.append((run_id, event_type))
        return True

    async def complete_run(
        self,
        *,
        run_id: str,
        status: BotRunStatus,
        error_code: str | None = None,
        error_message_redacted: str | None = None,
    ) -> bool:
        del error_code, error_message_redacted
        self.completed[run_id] = status
        return True

    async def find_stuck_runs(self, *, stale_before: datetime) -> tuple[str, ...]:
        del stale_before
        return ("run-stuck-1",)


class _ModuleResolver:
    def __init__(self, modules: dict[str, "_Module"]) -> None:
        self.modules = modules

    async def resolve(self, module_id: str):
        return self.modules.get(module_id)


class _Module:
    module_id = "fixture_bot"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.initialized = 0

    async def validate_config(self, config: BotInstanceConfig) -> BotValidationResult:
        del config
        return BotValidationResult(valid=True)

    async def initialize(self, context) -> None:
        del context
        self.initialized += 1

    async def dry_run(self, request: BotRunRequest) -> BotRunResult:
        if self.fail:
            raise RuntimeError("boom")
        return BotRunResult(
            run_id=request.run_id,
            instance_id=request.instance_id,
            module_id=request.module_id,
            mode=request.mode,
            status=BotRunStatus.COMPLETE,
            signals=(_signal(request),),
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
    def __init__(self) -> None:
        self.ids_by_key: dict[str, str] = {}

    async def publish(self, signal: BotSignal) -> BotSignalPublishResult:
        signal_id = self.ids_by_key.setdefault(signal.signal_key, f"signal-{len(self.ids_by_key) + 1}")
        return BotSignalPublishResult(accepted=True, signal_id=signal_id)


class _Noop:
    def __getattr__(self, name):
        def _method(*args, **kwargs):
            return None

        return _method


def test_candle_batch_ready_routes_one_snapshot_to_multiple_instances() -> None:
    run_repository = _RunRepository()
    service = _service(
        instances=(_config("instance-1"), _config("instance-2")),
        run_repository=run_repository,
        modules={"fixture_bot": _Module()},
    )

    results = asyncio.run(service.handle_candle_batch_ready(_event()))

    assert [result.instance_id for result in results] == ["instance-1", "instance-2"]
    assert [result.status for result in results] == [BotRunStatus.COMPLETE, BotRunStatus.COMPLETE]
    assert len(run_repository.created_runs) == 2
    assert set(run_repository.completed.values()) == {BotRunStatus.COMPLETE}


def test_failed_instance_does_not_stop_other_instances() -> None:
    run_repository = _RunRepository()
    service = _service(
        instances=(_config("failed", module_id="bad_bot"), _config("healthy", module_id="fixture_bot")),
        run_repository=run_repository,
        modules={"bad_bot": _Module(fail=True), "fixture_bot": _Module()},
    )

    results = asyncio.run(service.handle_candle_batch_ready(_event()))

    assert [result.status for result in results] == [BotRunStatus.FAILED, BotRunStatus.COMPLETE]
    assert len(run_repository.created_runs) == 2


def test_duplicate_candle_batch_ready_does_not_create_duplicate_runs() -> None:
    run_repository = _RunRepository()
    service = _service(
        instances=(_config("instance-1"),),
        run_repository=run_repository,
        modules={"fixture_bot": _Module()},
    )

    first = asyncio.run(service.handle_candle_batch_ready(_event()))
    second = asyncio.run(service.handle_candle_batch_ready(_event()))

    assert first[0].created is True
    assert second[0].created is False
    assert len(run_repository.created_runs) == 1


def test_stuck_run_detection_delegates_to_repository_threshold() -> None:
    service = _service(
        instances=(),
        run_repository=_RunRepository(),
        modules={},
    )

    stuck = asyncio.run(service.detect_stuck_runs(now=datetime(2026, 7, 14, tzinfo=UTC), timeout=timedelta(minutes=10)))

    assert stuck == ("run-stuck-1",)


def test_trigger_idempotency_key_changes_per_instance() -> None:
    first = build_trigger_idempotency_key(
        instance_id="instance-1",
        trigger_type=BotTriggerType.EVENT,
        trigger_id="event-1",
        snapshot_id="snapshot-1",
    )
    second = build_trigger_idempotency_key(
        instance_id="instance-2",
        trigger_type=BotTriggerType.EVENT,
        trigger_id="event-1",
        snapshot_id="snapshot-1",
    )

    assert first != second


def _service(*, instances: tuple[BotInstanceConfig, ...], run_repository: _RunRepository, modules: dict[str, _Module]) -> BotRunOrchestrationService:
    return BotRunOrchestrationService(
        instance_repository=_InstanceRepository(instances),
        run_repository=run_repository,
        module_resolver=_ModuleResolver(modules),
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


def _config(instance_id: str, *, module_id: str = "fixture_bot") -> BotInstanceConfig:
    return BotInstanceConfig(
        instance_id=instance_id,
        module_id=module_id,
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
        correlation_id="corr-1",
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


def _signal(request: BotRunRequest) -> BotSignal:
    return BotSignal.build(
        instance_id=request.instance_id,
        module_id=request.module_id,
        symbol="ETHUSDT",
        timeframe="1h",
        snapshot_id="snapshot-1",
        signal_type=BotSignalType.ENTRY,
        side=BotSignalSide.BUY,
        confidence=Decimal("0.8"),
        reason="stage_11",
        payload_schema="fixture.stage_11",
        payload_schema_version=1,
        payload={"price": Decimal("100")},
    )
