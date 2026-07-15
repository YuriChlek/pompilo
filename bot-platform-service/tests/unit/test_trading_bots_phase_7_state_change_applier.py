from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

from bot_platform_service.application import BotRunOrchestrationService, CandleBatchReadyEvent, RuntimeCapabilities
from bot_platform_service.domain import (
    BotCandle,
    BotHealth,
    BotHealthStatus,
    BotInstanceConfig,
    BotMarketDataContext,
    BotMarketSnapshot,
    BotMode,
    BotRunRequest,
    BotRunResult,
    BotRunStatus,
    BotStartResult,
    BotStateChange,
    BotStateChangeOperation,
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
    def __init__(self) -> None:
        self.keys: set[str] = set()
        self.completed: list[BotRunStatus] = []
        self.events: list[dict[str, object]] = []

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
        del event_id, run_id, instance_id, module_id, correlation_id
        if event_type == "COMPLETED":
            self.events.append(payload_json)
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


class _Resolver:
    def __init__(self, module: "_Module") -> None:
        self.module = module

    async def resolve(self, module_id: str):
        del module_id
        return self.module


class _Module:
    module_id = "fixture_bot"

    def __init__(
        self,
        *,
        status: BotRunStatus = BotRunStatus.COMPLETE,
        direct_save: bool = False,
        duplicate_returned_changes: bool = False,
    ) -> None:
        self.status = status
        self.direct_save = direct_save
        self.duplicate_returned_changes = duplicate_returned_changes
        self.context = None

    async def validate_config(self, config: BotInstanceConfig) -> BotValidationResult:
        del config
        return BotValidationResult(valid=True)

    async def initialize(self, context) -> None:
        self.context = context

    async def dry_run(self, request: BotRunRequest) -> BotRunResult:
        if self.direct_save:
            await self.context.state_store.save(
                instance_id=request.instance_id,
                namespace="runtime",
                state_key="ETHUSDT",
                value={"source": "direct"},
            )
        changes = (_state_change(request),)
        if self.duplicate_returned_changes:
            changes = changes + (_state_change(request, value={"source": "duplicate"}),)
        return BotRunResult(
            run_id=request.run_id,
            instance_id=request.instance_id,
            module_id=request.module_id,
            mode=request.mode,
            status=self.status,
            state_changes=changes,
        )

    async def run_once(self, request: BotRunRequest) -> BotRunResult:
        return await self.dry_run(request)

    async def start(self, request) -> BotStartResult:
        return BotStartResult(accepted=True, instance_id=request.instance_id)

    async def stop(self, instance_id: str) -> BotStopResult:
        return BotStopResult(accepted=True, instance_id=instance_id)

    async def health(self, instance_id: str) -> BotHealth:
        return BotHealth(instance_id=instance_id, module_id=self.module_id, status=BotHealthStatus.HEALTHY)


class _StateStore:
    def __init__(self) -> None:
        self.saved: list[tuple[str, str, str, object]] = []

    async def load(self, *, instance_id: str, namespace: str, state_key: str) -> object | None:
        del instance_id, namespace, state_key
        return None

    async def save(self, *, instance_id: str, namespace: str, state_key: str, value: object) -> None:
        self.saved.append((instance_id, namespace, state_key, value))


class _MarketData:
    async def get_latest_complete_snapshot(self, **kwargs):
        return _snapshot(kwargs["timeframe"])

    async def build_context(self, **kwargs):
        return BotMarketDataContext(primary_snapshot=_snapshot(kwargs["primary_timeframe"]))


class _Noop:
    def __getattr__(self, name):
        def _method(*args, **kwargs):
            return None

        return _method


def test_orchestration_persists_returned_state_changes_once_after_success() -> None:
    state_store = _StateStore()
    run_repository = _RunRepository()
    service = _service(module=_Module(duplicate_returned_changes=True), state_store=state_store, run_repository=run_repository)

    result = asyncio.run(service.handle_candle_batch_ready(_event()))

    assert result[0].status is BotRunStatus.COMPLETE
    assert len(state_store.saved) == 1
    assert state_store.saved[0] == ("instance-1", "runtime", "ETHUSDT", {"source": "returned"})
    assert run_repository.events[0]["state_changes_applied"] == 1
    assert run_repository.events[0]["state_changes_skipped"] == 1


def test_orchestration_skips_returned_state_change_when_adapter_already_saved_it() -> None:
    state_store = _StateStore()
    run_repository = _RunRepository()
    service = _service(module=_Module(direct_save=True), state_store=state_store, run_repository=run_repository)

    result = asyncio.run(service.handle_candle_batch_ready(_event()))

    assert result[0].status is BotRunStatus.COMPLETE
    assert len(state_store.saved) == 1
    assert state_store.saved[0] == ("instance-1", "runtime", "ETHUSDT", {"source": "direct"})
    assert run_repository.events[0]["state_changes_applied"] == 0
    assert run_repository.events[0]["state_changes_skipped"] == 1


def test_orchestration_does_not_apply_state_changes_for_failed_or_cancelled_results() -> None:
    for status in (BotRunStatus.FAILED, BotRunStatus.CANCELLED):
        state_store = _StateStore()
        run_repository = _RunRepository()
        service = _service(module=_Module(status=status), state_store=state_store, run_repository=run_repository)

        result = asyncio.run(service.handle_candle_batch_ready(_event()))

        assert result[0].status is status
        assert state_store.saved == []
        assert run_repository.completed == [status]
        assert run_repository.events[0]["state_changes_applied"] == 0
        assert run_repository.events[0]["state_changes_skipped"] == 1


def _service(*, module: _Module, state_store: _StateStore, run_repository: _RunRepository) -> BotRunOrchestrationService:
    return BotRunOrchestrationService(
        instance_repository=_InstanceRepository((_config(),)),
        run_repository=run_repository,
        module_resolver=_Resolver(module),
        runtime_capabilities=RuntimeCapabilities(
            market_data=_MarketData(),
            signal_publisher=_Noop(),
            state_store=state_store,
            notification_publisher=_Noop(),
            secret_provider=_Noop(),
            logger=_Noop(),
            metrics=_Noop(),
            clock=_Noop(),
        ),
    )


def _config() -> BotInstanceConfig:
    return BotInstanceConfig(
        instance_id="instance-1",
        module_id="fixture_bot",
        mode=BotMode.DRY_RUN,
        symbols=("ETHUSDT",),
        timeframes=("1h",),
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


def _state_change(request: BotRunRequest, *, value: dict[str, object] | None = None) -> BotStateChange:
    return BotStateChange(
        instance_id=request.instance_id,
        namespace="runtime",
        state_key="ETHUSDT",
        operation=BotStateChangeOperation.UPSERT,
        value=value or {"source": "returned"},
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
