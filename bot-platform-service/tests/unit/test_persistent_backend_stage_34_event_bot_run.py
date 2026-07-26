from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

from bot_platform_service.application import (
    EventRunCommand,
    ManualBotRunService,
    MarketDataEventConsumerService,
    MarketDataEventRunDispatcherService,
    RuntimeCapabilities,
    build_market_data_event_run_idempotency_key,
)
from bot_platform_service.domain import (
    BotCandle,
    BotHealth,
    BotHealthStatus,
    BotInstanceConfig,
    BotInstanceStatus,
    BotMarketDataContext,
    BotMarketSnapshot,
    BotMode,
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
    MarketDataCandlesCollectedEvent,
    parse_market_data_candles_collected_event,
)


class _Logger:
    def __init__(self) -> None:
        self.info_records: list[tuple[str, dict[str, object]]] = []
        self.warning_records: list[tuple[str, dict[str, object]]] = []

    def info(self, event: str, **fields: object) -> None:
        self.info_records.append((event, dict(fields)))

    def warning(self, event: str, **fields: object) -> None:
        self.warning_records.append((event, dict(fields)))

    def error(self, event: str, **fields: object) -> None:
        return None


class _IdempotencyStore:
    def __init__(self) -> None:
        self.keys: set[str] = set()

    async def record_processed_event(self, *, event, **kwargs) -> bool:
        if event.idempotency_key in self.keys:
            return False
        self.keys.add(event.idempotency_key)
        return True


class _InstanceRepository:
    def __init__(self, instances: tuple[BotInstanceConfig, ...]) -> None:
        self.instances = {instance.instance_id: instance for instance in instances}
        self.statuses = {instance.instance_id: BotInstanceStatus.ENABLED for instance in instances}

    async def get_instance_config(self, instance_id: str) -> BotInstanceConfig | None:
        return self.instances.get(instance_id)

    async def get_instance_status(self, instance_id: str) -> BotInstanceStatus | None:
        return self.statuses.get(instance_id)

    async def list_enabled_instances_for_snapshot(
        self,
        *,
        source: str,
        canonical_symbol: str,
        timeframe: str,
    ) -> tuple[BotInstanceConfig, ...]:
        _ = source
        return tuple(
            instance
            for instance in self.instances.values()
            if instance.mode is BotMode.SIGNAL_ONLY
            and canonical_symbol.upper() in {symbol.upper() for symbol in instance.symbols}
            and timeframe in instance.timeframes
        )


class _RunRepository:
    def __init__(self) -> None:
        self.created_idempotency_keys: set[str] = set()
        self.created_runs: list[dict[str, object]] = []
        self.completed_runs: list[dict[str, object]] = []
        self.events: list[dict[str, object]] = []

    async def acquire_instance_run_lock(self, *, instance_id: str) -> bool:
        return True

    async def create_run(self, **kwargs) -> bool:
        idempotency_key = str(kwargs["idempotency_key"])
        if idempotency_key in self.created_idempotency_keys:
            return False
        self.created_idempotency_keys.add(idempotency_key)
        self.created_runs.append(dict(kwargs))
        return True

    async def complete_run(self, **kwargs) -> bool:
        self.completed_runs.append(dict(kwargs))
        return True

    async def append_run_event(self, **kwargs) -> bool:
        self.events.append(dict(kwargs))
        return True


class _SignalRepository:
    def __init__(self) -> None:
        self.signals: list[dict[str, object]] = []

    async def publish_signal(self, **kwargs) -> str:
        self.signals.append(dict(kwargs))
        return str(kwargs["signal_id"])


class _AuditRepository:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def append_audit_event(self, **kwargs) -> bool:
        self.events.append(dict(kwargs))
        return True


class _SignalEventPublisher:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def publish_persisted_signal(self, **kwargs) -> bool:
        self.events.append(dict(kwargs))
        return True


class _Resolver:
    def __init__(self, module: "_Module") -> None:
        self.module = module

    async def resolve(self, module_id: str):
        return self.module if module_id == self.module.module_id else None


class _Module:
    module_id = "spot_grid"

    def __init__(self) -> None:
        self.requests: list[BotRunRequest] = []

    async def validate_config(self, config: BotInstanceConfig) -> BotValidationResult:
        return BotValidationResult(valid=True)

    async def initialize(self, context) -> None:
        return None

    async def dry_run(self, request: BotRunRequest) -> BotRunResult:
        return await self.run_once(request)

    async def run_once(self, request: BotRunRequest) -> BotRunResult:
        self.requests.append(request)
        return BotRunResult(
            run_id=request.run_id,
            instance_id=request.instance_id,
            module_id=request.module_id,
            mode=request.mode,
            status=BotRunStatus.COMPLETE,
            signals=(
                BotSignal.build(
                    instance_id=request.instance_id,
                    module_id=request.module_id,
                    symbol=request.market_data.primary_snapshot.canonical_symbol,
                    timeframe=request.market_data.primary_snapshot.timeframe,
                    snapshot_id=request.market_data.primary_snapshot.snapshot_id,
                    signal_type=BotSignalType.HOLD,
                    side=BotSignalSide.BUY,
                    confidence=Decimal("0.75"),
                    reason="stage 34 event run",
                    payload_schema="stage34.signal.v1",
                    payload_schema_version=1,
                    payload={"source": "event"},
                ),
            ),
            diagnostics={"event": True},
        )

    async def start(self, request) -> BotStartResult:
        return BotStartResult(accepted=True, instance_id=request.instance_id)

    async def stop(self, instance_id: str) -> BotStopResult:
        return BotStopResult(accepted=True, instance_id=instance_id)

    async def health(self, instance_id: str) -> BotHealth:
        return BotHealth(instance_id=instance_id, module_id=self.module_id, status=BotHealthStatus.HEALTHY)


class _MarketData:
    def __init__(self, *, fail_get_snapshot: bool = False) -> None:
        self.fail_get_snapshot = fail_get_snapshot
        self.snapshot_ids: list[str] = []
        self.latest_calls: list[dict[str, object]] = []

    async def get_snapshot(self, *, snapshot_id: str) -> BotMarketSnapshot:
        self.snapshot_ids.append(snapshot_id)
        if self.fail_get_snapshot:
            raise RuntimeError("snapshot temporarily unavailable")
        return _snapshot(snapshot_id=snapshot_id, timeframe="1h")

    async def get_latest_complete_snapshot(self, **kwargs) -> BotMarketSnapshot:
        self.latest_calls.append(dict(kwargs))
        return _snapshot(snapshot_id=f"latest-{kwargs['timeframe']}", timeframe=str(kwargs["timeframe"]))

    async def build_context(self, **kwargs) -> BotMarketDataContext:
        return BotMarketDataContext(primary_snapshot=_snapshot(snapshot_id="latest-1h", timeframe="1h"))


class _NoopSignalPublisher:
    async def publish(self, signal) -> BotSignalPublishResult:
        return BotSignalPublishResult(accepted=False, signal_id=None, error_code="SIGNAL_FANOUT_DISABLED")


class _NoopStateStore:
    async def load(self, *, instance_id: str, namespace: str, state_key: str) -> object | None:
        return None

    async def save(self, *, instance_id: str, namespace: str, state_key: str, value: object) -> None:
        return None


class _NoopNotificationPublisher:
    async def publish(self, *, instance_id: str, message_type: str, payload: object) -> None:
        return None


class _NoopSecretProvider:
    async def resolve(self, *, instance_id: str, secret_name: str) -> str:
        raise RuntimeError("not configured")


class _Metrics:
    def increment(self, name: str, *, tags: dict[str, str] | None = None) -> None:
        return None

    def observe(self, name: str, value: float, *, tags: dict[str, str] | None = None) -> None:
        return None


class _Clock:
    def now(self):
        return datetime(2026, 7, 16, tzinfo=UTC)


def test_stage_34_event_consumer_dispatches_run_for_each_matching_instance() -> None:
    async def run() -> None:
        instances = (
            _instance("instance-1", timeframes=("1h",)),
            _instance("instance-2", timeframes=("1h", "4h")),
        )
        run_repository = _RunRepository()
        module = _Module()
        service = MarketDataEventConsumerService(
            logger=_Logger(),
            idempotency_store=_IdempotencyStore(),
            instance_repository=_InstanceRepository(instances),
            run_dispatcher=MarketDataEventRunDispatcherService(
                event_run_service=_manual_service(
                    instance_repository=_InstanceRepository(instances),
                    run_repository=run_repository,
                    module=module,
                    market_data=_MarketData(),
                )
            ),
        )

        result = await service.handle_message(message_id="1-0", payload=_event_payload())

        assert result.ack is True
        assert result.matched_instance_count == 2
        assert len(result.dispatch_results) == 2
        assert {record["instance_id"] for record in run_repository.created_runs} == {"instance-1", "instance-2"}
        assert all(record["trigger_type"] is BotTriggerType.EVENT for record in run_repository.created_runs)
        assert all(record["snapshot_id"] == "event-snapshot-1" for record in run_repository.created_runs)
        assert len(module.requests) == 2
        assert all(request.trigger_type is BotTriggerType.EVENT for request in module.requests)

    asyncio.run(run())


def test_stage_34_event_run_uses_event_snapshot_id_and_persists_signal_audit() -> None:
    async def run() -> None:
        instance_repository = _InstanceRepository(
            (
                _instance(
                    "instance-1",
                    timeframes=("1h", "4h"),
                    config={"max_grid_levels": 3, "runtime": {"emit_diagnostics": True}},
                ),
            )
        )
        run_repository = _RunRepository()
        signal_repository = _SignalRepository()
        audit_repository = _AuditRepository()
        market_data = _MarketData()
        module = _Module()
        service = _manual_service(
            instance_repository=instance_repository,
            run_repository=run_repository,
            signal_repository=signal_repository,
            audit_repository=audit_repository,
            module=module,
            market_data=market_data,
        )

        result = await service.run_event_instance(
            EventRunCommand(
                instance_id="instance-1",
                source="BINANCE_SPOT",
                canonical_symbol="BTCUSDT",
                timeframe="1h",
                snapshot_id="event-snapshot-1",
                event_id="BINANCE_SPOT:BTCUSDT:1h:2026-07-16T01:00:00Z",
                idempotency_key="market-data-event:BINANCE_SPOT:BTCUSDT:1h:2026-07-16 01:00:00+00:00:instance-1",
                correlation_id="1-0",
            )
        )

        assert result.accepted is True
        assert market_data.snapshot_ids == ["event-snapshot-1"]
        assert market_data.latest_calls == [
            {
                "source": "BINANCE_SPOT",
                "canonical_symbol": "BTCUSDT",
                "timeframe": "4h",
            }
        ]
        assert run_repository.created_runs[0]["trigger_type"] is BotTriggerType.EVENT
        assert run_repository.created_runs[0]["trigger_event_id"] == "BINANCE_SPOT:BTCUSDT:1h:2026-07-16T01:00:00Z"
        assert run_repository.created_runs[0]["snapshot_id"] == "event-snapshot-1"
        assert signal_repository.signals[0]["signal"].snapshot_id == "event-snapshot-1"
        assert audit_repository.events[0]["actor_id"] == "event_run"
        assert module.requests[0].market_data.primary_snapshot.snapshot_id == "event-snapshot-1"
        assert module.requests[0].config["max_grid_levels"] == 3
        assert module.requests[0].config["runtime"] == {"emit_diagnostics": True}

    asyncio.run(run())


def test_stage_34_repeated_event_does_not_create_duplicate_runs() -> None:
    async def run() -> None:
        event = parse_market_data_candles_collected_event(_event_payload())
        instance = _instance("instance-1", timeframes=("1h",))
        run_repository = _RunRepository()
        dispatcher = MarketDataEventRunDispatcherService(
            event_run_service=_manual_service(
                instance_repository=_InstanceRepository((instance,)),
                run_repository=run_repository,
                module=_Module(),
                market_data=_MarketData(),
            )
        )

        first = await dispatcher.dispatch_event(event=event, instances=(instance,), correlation_id="1-0")
        second = await dispatcher.dispatch_event(event=event, instances=(instance,), correlation_id="1-1")

        assert first[0].accepted is True
        assert second[0].accepted is False
        assert second[0].duplicate is True
        assert len(run_repository.created_runs) == 1
        assert first[0].run_id == second[0].run_id
        assert run_repository.created_runs[0]["idempotency_key"] == build_market_data_event_run_idempotency_key(
            event=event,
            instance_id="instance-1",
        )

    asyncio.run(run())


def test_stage_34_signal_only_safety_rejects_non_signal_only_event_run() -> None:
    async def run() -> None:
        instance_repository = _InstanceRepository(
            (
                BotInstanceConfig(
                    instance_id="instance-1",
                    module_id="spot_grid",
                    mode=BotMode.DRY_RUN,
                    symbols=("BTCUSDT",),
                    timeframes=("1h",),
                    config_schema_version=1,
                    config={},
                ),
            )
        )
        run_repository = _RunRepository()
        service = _manual_service(
            instance_repository=instance_repository,
            run_repository=run_repository,
            module=_Module(),
            market_data=_MarketData(),
        )

        result = await service.run_event_instance(
            EventRunCommand(
                instance_id="instance-1",
                source="BINANCE_SPOT",
                canonical_symbol="BTCUSDT",
                timeframe="1h",
                snapshot_id="event-snapshot-1",
                event_id="event-1",
                idempotency_key="market-data-event:BINANCE_SPOT:BTCUSDT:1h:closed:instance-1",
            )
        )

        assert result.accepted is False
        assert result.error_code == "INSTANCE_MODE_NOT_SIGNAL_ONLY"
        assert run_repository.created_runs == []

    asyncio.run(run())


def test_stage_34_transient_snapshot_error_is_not_acknowledged_for_retry() -> None:
    async def run() -> None:
        instance = _instance("instance-1", timeframes=("1h",))
        logger = _Logger()
        service = MarketDataEventConsumerService(
            logger=logger,
            idempotency_store=_IdempotencyStore(),
            instance_repository=_InstanceRepository((instance,)),
            run_dispatcher=MarketDataEventRunDispatcherService(
                event_run_service=_manual_service(
                    instance_repository=_InstanceRepository((instance,)),
                    run_repository=_RunRepository(),
                    module=_Module(),
                    market_data=_MarketData(fail_get_snapshot=True),
                )
            ),
        )

        result = await service.handle_message(message_id="1-0", payload=_event_payload())

        assert result.ack is False
        assert result.terminal is False
        assert result.error == "MARKET_DATA_SNAPSHOT_ERROR"
        assert logger.warning_records[-1][0] == "bot_platform.market_data_event.dispatch_transient_error"

    asyncio.run(run())


def _manual_service(
    *,
    instance_repository: _InstanceRepository,
    run_repository: _RunRepository,
    module: _Module,
    market_data: _MarketData,
    signal_repository: _SignalRepository | None = None,
    audit_repository: _AuditRepository | None = None,
) -> ManualBotRunService:
    return ManualBotRunService(
        instance_repository=instance_repository,
        run_repository=run_repository,
        signal_repository=signal_repository or _SignalRepository(),
        audit_repository=audit_repository or _AuditRepository(),
        signal_event_publisher=_SignalEventPublisher(),
        module_resolver=_Resolver(module),
        runtime_capabilities=RuntimeCapabilities(
            market_data=market_data,
            signal_publisher=_NoopSignalPublisher(),
            state_store=_NoopStateStore(),
            notification_publisher=_NoopNotificationPublisher(),
            secret_provider=_NoopSecretProvider(),
            logger=_Logger(),
            metrics=_Metrics(),
            clock=_Clock(),
        ),
        market_data_source="BINANCE_SPOT",
    )


def _instance(
    instance_id: str,
    *,
    timeframes: tuple[str, ...],
    config: dict[str, object] | None = None,
) -> BotInstanceConfig:
    return BotInstanceConfig(
        instance_id=instance_id,
        module_id="spot_grid",
        mode=BotMode.SIGNAL_ONLY,
        symbols=("BTCUSDT",),
        timeframes=timeframes,
        config_schema_version=1,
        config=config or {},
    )


def _event_payload() -> dict[str, object]:
    return {
        "event_type": "market_data.candles_collected",
        "contract_version": "market-data-event.v1",
        "source": "BINANCE_SPOT",
        "symbol": "BTCUSDT",
        "provider_symbol": "BTCUSDT",
        "timeframe": "1h",
        "from": "2026-07-16T00:00:00Z",
        "to": "2026-07-16T01:00:00Z",
        "batch_id": "batch-1",
        "snapshot_id": "event-snapshot-1",
        "closed_at": "2026-07-16T01:00:00Z",
        "idempotency_key": "BINANCE_SPOT:BTCUSDT:1h:2026-07-16T01:00:00Z",
    }


def _snapshot(*, snapshot_id: str, timeframe: str) -> BotMarketSnapshot:
    return BotMarketSnapshot(
        snapshot_id=snapshot_id,
        source="BINANCE_SPOT",
        canonical_symbol="BTCUSDT",
        provider_symbol="BTCUSDT",
        timeframe=timeframe,
        last_closed_candle_time=datetime(2026, 7, 16, 1, tzinfo=UTC),
        lookback_start_time=datetime(2026, 7, 16, 0, tzinfo=UTC),
        lookback_end_time=datetime(2026, 7, 16, 1, tzinfo=UTC),
        completeness_status="COMPLETE",
        snapshot_version=1,
        data_hash="hash",
        candles=(
            BotCandle(
                source="BINANCE_SPOT",
                canonical_symbol="BTCUSDT",
                timeframe=timeframe,
                open_time=datetime(2026, 7, 16, 0, tzinfo=UTC),
                close_time=datetime(2026, 7, 16, 1, tzinfo=UTC),
                open=Decimal("100"),
                high=Decimal("110"),
                low=Decimal("90"),
                close=Decimal("105"),
                volume=Decimal("10"),
            ),
        ),
    )
