from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from bot_platform_service.application import BotRunOrchestrationService, PollingScheduleTick, RuntimeCapabilities
from bot_platform_service.domain import (
    BotCandle,
    BotInstanceConfig,
    BotMarketDataContext,
    BotMarketSnapshot,
    BotMode,
    BotModuleMetadata,
    BotModuleStatus,
    BotNotificationStatus,
    BotRunRequest,
    BotRunResult,
    BotRunStatus,
    BotSignal,
    BotSignalPublishResult,
    BotSignalSide,
    BotSignalType,
    BotTriggerType,
)
from bot_platform_service.registry import PersistedBotModuleResolver, discover_trading_bot_registrations
from tests.fixtures.spot_grid_indicator_runtime import FakeStockIndicatorsRuntime
from bot_platform_service.trading_bots.spot_grid.application import SpotGridTradingCycleService


INDICATOR_RUNTIME = FakeStockIndicatorsRuntime()


class _InstanceRepository:
    async def list_enabled_instances_for_snapshot(self, *, source: str, canonical_symbol: str, timeframe: str):
        del source, canonical_symbol, timeframe
        return ()


class _RunRepository:
    def __init__(self) -> None:
        self.created_keys: set[str] = set()
        self.created_runs: list[tuple[str, str, str]] = []
        self.completed: dict[str, BotRunStatus] = {}
        self.events: list[tuple[str, str, dict[str, object]]] = []

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
        del trigger_type, status, trigger_event_id, snapshot_id, correlation_id
        key = str(idempotency_key)
        if key in self.created_keys:
            return False
        self.created_keys.add(key)
        self.created_runs.append((run_id, instance_id, module_id))
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
        del event_id, instance_id, module_id, correlation_id
        self.events.append((run_id, event_type, dict(payload_json)))
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
        return ()


class _MetadataRepository:
    def __init__(self) -> None:
        self.metadata = {
            metadata.module_id: metadata
            for metadata in (_metadata_from_registration(registration) for registration in discover_trading_bot_registrations())
        }

    async def get_active_module_metadata(self, module_id: str) -> BotModuleMetadata | None:
        return self.metadata.get(module_id)


class _Resolver:
    def __init__(self, metadata_repository: _MetadataRepository) -> None:
        self.persisted = PersistedBotModuleResolver(metadata_repository)

    async def resolve(self, module_id: str):
        if module_id == "fixture_bot":
            return _FixtureBotModule()
        module = await self.persisted.resolve(module_id)
        if getattr(module, "module_id", None) == "spot_grid":
            module._cycle_service = SpotGridTradingCycleService(indicator_runtime=INDICATOR_RUNTIME)
        return module


class _FixtureBotModule:
    module_id = "fixture_bot"

    async def validate_config(self, config: BotInstanceConfig):
        del config

    async def initialize(self, context) -> None:
        del context

    async def dry_run(self, request: BotRunRequest) -> BotRunResult:
        return BotRunResult(
            run_id=request.run_id,
            instance_id=request.instance_id,
            module_id=request.module_id,
            mode=request.mode,
            status=BotRunStatus.COMPLETE,
            signals=(_fixture_signal(request),),
            diagnostics={"fixture": True},
        )

    async def run_once(self, request: BotRunRequest) -> BotRunResult:
        return await self.dry_run(request)

    async def start(self, request):
        del request

    async def stop(self, instance_id: str):
        del instance_id

    async def health(self, instance_id: str):
        del instance_id


class _MarketData:
    def __init__(self) -> None:
        self.context_calls: list[tuple[str, str, tuple[str, ...]]] = []

    async def get_latest_complete_snapshot(self, **kwargs):
        return _snapshot(kwargs["canonical_symbol"], kwargs["timeframe"])

    async def build_context(self, **kwargs):
        self.context_calls.append((kwargs["canonical_symbol"], kwargs["primary_timeframe"], kwargs["supporting_timeframes"]))
        primary = _snapshot(kwargs["canonical_symbol"], kwargs["primary_timeframe"])
        supporting = tuple(_snapshot(kwargs["canonical_symbol"], timeframe) for timeframe in kwargs["supporting_timeframes"])
        return BotMarketDataContext(primary_snapshot=primary, supporting_snapshots=supporting)


class _SignalPublisher:
    def __init__(self) -> None:
        self.signals: list[BotSignal] = []
        self.ids_by_key: dict[str, str] = {}

    async def publish(self, signal: BotSignal) -> BotSignalPublishResult:
        self.signals.append(signal)
        signal_id = self.ids_by_key.setdefault(signal.signal_key, f"signal-{len(self.ids_by_key) + 1}")
        return BotSignalPublishResult(accepted=True, signal_id=signal_id)


class _NotificationPublisher:
    def __init__(self) -> None:
        self.notifications: list[tuple[str, str, object]] = []

    async def publish(self, *, instance_id: str, message_type: str, payload: object) -> None:
        self.notifications.append((instance_id, message_type, payload))


class _StateStore:
    def __init__(self) -> None:
        self.saved: list[tuple[str, str, str, object]] = []

    async def load(self, *, instance_id: str, namespace: str, state_key: str) -> object | None:
        del instance_id, namespace, state_key
        return None

    async def save(self, *, instance_id: str, namespace: str, state_key: str, value: object) -> None:
        self.saved.append((instance_id, namespace, state_key, value))


class _Noop:
    def __getattr__(self, name):
        def _method(*args, **kwargs):
            return None

        return _method


def test_phase_16_orchestration_runs_fixture_spot_grid_and_spot_greenwich_from_metadata() -> None:
    run_repository = _RunRepository()
    market_data = _MarketData()
    signal_publisher = _SignalPublisher()
    notification_publisher = _NotificationPublisher()
    state_store = _StateStore()
    service = _service(
        run_repository=run_repository,
        market_data=market_data,
        signal_publisher=signal_publisher,
        notification_publisher=notification_publisher,
        state_store=state_store,
    )
    instances = (
        _config("fixture-1", "fixture_bot", BotMode.DRY_RUN, ("1h",)),
        _config("grid-1", "spot_grid", BotMode.SIGNAL_ONLY, ("1h", "4h")),
        _config("greenwich-1", "spot_greenwich", BotMode.NOTIFICATION_ONLY, ("4h", "1d")),
    )

    results = asyncio.run(service.run_polling_tick(PollingScheduleTick("tick-1"), instances))

    assert [result.module_id for result in results] == ["fixture_bot", "spot_grid", "spot_greenwich"]
    assert [result.status for result in results] == [BotRunStatus.COMPLETE, BotRunStatus.COMPLETE, BotRunStatus.COMPLETE]
    assert [call[1:] for call in market_data.context_calls] == [
        ("1h", ()),
        ("1h", ("4h",)),
        ("4h", ("1d",)),
    ]
    assert signal_publisher.signals == []
    assert len(notification_publisher.notifications) == 1
    assert notification_publisher.notifications[0][1] == "spot_greenwich_signal"
    assert notification_publisher.notifications[0][2]["status"] == BotNotificationStatus.SKIPPED.value
    assert {saved[1] for saved in state_store.saved} == {"spot_grid", "spot_greenwich"}
    completed_payloads = [payload for _, event_type, payload in run_repository.events if event_type == "COMPLETED"]
    assert len(completed_payloads) == 3
    assert any(payload["signal_publish_count"] == 0 for payload in completed_payloads)
    assert any(payload["notification_publish_count"] == 1 for payload in completed_payloads)
    assert all("diagnostics" in payload for payload in completed_payloads)


def test_phase_16_failed_platform_native_instance_does_not_stop_other_instances() -> None:
    run_repository = _RunRepository()
    service = _service(
        run_repository=run_repository,
        market_data=_MarketData(),
        signal_publisher=_SignalPublisher(),
        notification_publisher=_NotificationPublisher(),
        state_store=_StateStore(),
    )
    instances = (
        _config("missing-1", "missing_module", BotMode.SIGNAL_ONLY, ("1h",)),
        _config("grid-1", "spot_grid", BotMode.SIGNAL_ONLY, ("1h", "4h")),
    )

    results = asyncio.run(service.run_polling_tick(PollingScheduleTick("tick-2"), instances))

    assert [result.status for result in results] == [BotRunStatus.FAILED, BotRunStatus.COMPLETE]
    assert [result.error_code for result in results] == ["INSTANCE_RUN_FAILED", None]
    assert run_repository.completed[results[0].run_id] is BotRunStatus.FAILED
    assert run_repository.completed[results[1].run_id] is BotRunStatus.COMPLETE


def _service(
    *,
    run_repository: _RunRepository,
    market_data: _MarketData,
    signal_publisher: _SignalPublisher,
    notification_publisher: _NotificationPublisher,
    state_store: _StateStore,
) -> BotRunOrchestrationService:
    return BotRunOrchestrationService(
        instance_repository=_InstanceRepository(),
        run_repository=run_repository,
        module_resolver=_Resolver(_MetadataRepository()),
        runtime_capabilities=RuntimeCapabilities(
            market_data=market_data,
            signal_publisher=signal_publisher,
            state_store=state_store,
            notification_publisher=notification_publisher,
            secret_provider=_Noop(),
            logger=_Noop(),
            metrics=_Noop(),
            clock=_Noop(),
        ),
    )


def _config(
    instance_id: str,
    module_id: str,
    mode: BotMode,
    timeframes: tuple[str, ...],
) -> BotInstanceConfig:
    return BotInstanceConfig(
        instance_id=instance_id,
        module_id=module_id,
        mode=mode,
        symbols=("ETHUSDT",),
        timeframes=timeframes,
        config_schema_version=1,
    )


def _metadata_from_registration(registration) -> BotModuleMetadata:
    manifest = registration.manifest
    return BotModuleMetadata(
        module_id=manifest.module_id,
        display_name=manifest.display_name,
        version=manifest.version,
        adapter_path=registration.adapter_path,
        adapter_class=registration.adapter_class,
        status=BotModuleStatus.ACTIVE,
        manifest={
            "module_id": manifest.module_id,
            "display_name": manifest.display_name,
            "version": manifest.version,
            "supported_modes": [mode.value for mode in manifest.supported_modes],
            "required_timeframes": list(manifest.required_timeframes),
            "required_market_data": list(manifest.required_market_data),
            "supports_multi_symbol": manifest.supports_multi_symbol,
            "config_schema_version": manifest.config_schema_version,
            "status": manifest.status.value,
        },
        config_schema_version=manifest.config_schema_version,
        config_schema=registration.config_schema,
    )


def _fixture_signal(request: BotRunRequest) -> BotSignal:
    assert request.market_data is not None
    return BotSignal.build(
        instance_id=request.instance_id,
        module_id=request.module_id,
        symbol=request.market_data.primary_snapshot.canonical_symbol,
        timeframe=request.market_data.primary_snapshot.timeframe,
        snapshot_id=request.market_data.primary_snapshot.snapshot_id,
        signal_type=BotSignalType.ENTRY,
        side=BotSignalSide.BUY,
        confidence=Decimal("0.80"),
        reason="fixture_entry",
        payload_schema="fixture.entry",
        payload_schema_version=1,
        payload={"price": Decimal("100")},
    )


def _snapshot(symbol: str, timeframe: str) -> BotMarketSnapshot:
    now = datetime(2026, 7, 15, tzinfo=UTC)
    candles = tuple(
        BotCandle(
            source="binance_spot",
            canonical_symbol=symbol,
            timeframe=timeframe,
            open_time=now + timedelta(minutes=index),
            close_time=now + timedelta(minutes=index + 1),
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=Decimal("1000"),
        )
        for index in range(100)
    )
    return BotMarketSnapshot(
        snapshot_id=f"snapshot-{symbol}-{timeframe}",
        source="binance_spot",
        canonical_symbol=symbol,
        provider_symbol=symbol,
        timeframe=timeframe,
        last_closed_candle_time=now,
        lookback_start_time=now,
        lookback_end_time=now,
        completeness_status="COMPLETE",
        snapshot_version=1,
        data_hash=f"hash-{symbol}-{timeframe}",
        candles=candles,
    )
