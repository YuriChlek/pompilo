from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

from bot_platform_service.application import ManualBotRunService, ManualRunCommand, RuntimeCapabilities
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
    BotSignalSide,
    BotSignalType,
    BotStartResult,
    BotStopResult,
    BotValidationResult,
)
from bot_platform_service.observability.metrics import InMemoryMetricsRecorder


class _InstanceRepository:
    config = BotInstanceConfig(
        instance_id="instance-1",
        module_id="spot_grid",
        mode=BotMode.SIGNAL_ONLY,
        symbols=("ETHUSDT",),
        timeframes=("1h",),
        config_schema_version=1,
        config={},
    )

    async def get_instance_config(self, instance_id: str) -> BotInstanceConfig | None:
        return self.config if instance_id == self.config.instance_id else None

    async def get_instance_status(self, instance_id: str) -> BotInstanceStatus | None:
        return BotInstanceStatus.ENABLED if instance_id == self.config.instance_id else None


class _RunRepository:
    def __init__(self) -> None:
        self.created_runs: list[dict[str, object]] = []
        self.completed_runs: list[dict[str, object]] = []
        self.events: list[dict[str, object]] = []

    async def acquire_instance_run_lock(self, *, instance_id: str) -> bool:
        return True

    async def create_run(self, **kwargs) -> bool:
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

    def __init__(self, *, invalid_signal: bool = False) -> None:
        self.invalid_signal = invalid_signal

    async def validate_config(self, config: BotInstanceConfig) -> BotValidationResult:
        return BotValidationResult(valid=True)

    async def initialize(self, context) -> None:
        return None

    async def dry_run(self, request: BotRunRequest) -> BotRunResult:
        return await self.run_once(request)

    async def run_once(self, request: BotRunRequest) -> BotRunResult:
        signals = ("not-a-signal",) if self.invalid_signal else (_signal(request),)
        return BotRunResult(
            run_id=request.run_id,
            instance_id=request.instance_id,
            module_id=request.module_id,
            mode=request.mode,
            status=BotRunStatus.COMPLETE,
            signals=signals,  # type: ignore[arg-type]
        )

    async def start(self, request) -> BotStartResult:
        return BotStartResult(accepted=True, instance_id=request.instance_id)

    async def stop(self, instance_id: str) -> BotStopResult:
        return BotStopResult(accepted=True, instance_id=instance_id)

    async def health(self, instance_id: str) -> BotHealth:
        return BotHealth(instance_id=instance_id, module_id=self.module_id, status=BotHealthStatus.HEALTHY)


def test_stage_23_manual_run_persists_run_events_signals_and_audit() -> None:
    async def run() -> None:
        run_repository = _RunRepository()
        signal_repository = _SignalRepository()
        audit_repository = _AuditRepository()
        signal_event_publisher = _SignalEventPublisher()
        service = _service(
            run_repository=run_repository,
            signal_repository=signal_repository,
            audit_repository=audit_repository,
            signal_event_publisher=signal_event_publisher,
            module=_Module(),
        )

        result = await service.run_instance(ManualRunCommand(instance_id="instance-1", idempotency_key="manual-1"))

        assert result.accepted is True
        assert [event["event_type"] for event in run_repository.events] == ["STARTED", "COMPLETED"]
        assert len(signal_repository.signals) == 1
        assert signal_repository.signals[0]["run_id"] == result.run_id
        assert signal_repository.signals[0]["signal"].signal_key
        assert [event["event_type"] for event in audit_repository.events] == ["SIGNAL_PERSISTED"]
        assert audit_repository.events[0]["payload_json"]["boundary"] == "execution_service_reads_signals_only"
        assert signal_event_publisher.events[0]["signal_id"] == signal_repository.signals[0]["signal_id"]

    asyncio.run(run())


def test_stage_23_invalid_signal_is_not_persisted() -> None:
    async def run() -> None:
        run_repository = _RunRepository()
        signal_repository = _SignalRepository()
        audit_repository = _AuditRepository()
        signal_event_publisher = _SignalEventPublisher()
        service = _service(
            run_repository=run_repository,
            signal_repository=signal_repository,
            audit_repository=audit_repository,
            signal_event_publisher=signal_event_publisher,
            module=_Module(invalid_signal=True),
        )

        result = await service.run_instance(ManualRunCommand(instance_id="instance-1", idempotency_key="manual-1"))

        assert result.accepted is False
        assert result.status is BotRunStatus.FAILED
        assert result.error_code == "INVALID_SIGNAL_CONTRACT"
        assert signal_repository.signals == []
        assert signal_event_publisher.events == []
        assert [event["event_type"] for event in run_repository.events] == ["STARTED", "INVALID_SIGNAL_REJECTED"]
        assert audit_repository.events[0]["event_type"] == "INVALID_SIGNAL_REJECTED"

    asyncio.run(run())


def _service(
    *,
    run_repository: _RunRepository,
    signal_repository: _SignalRepository,
    audit_repository: _AuditRepository,
    signal_event_publisher: _SignalEventPublisher,
    module: _Module,
) -> ManualBotRunService:
    return ManualBotRunService(
        instance_repository=_InstanceRepository(),
        run_repository=run_repository,
        signal_repository=signal_repository,
        audit_repository=audit_repository,
        signal_event_publisher=signal_event_publisher,
        module_resolver=_Resolver(module),
        runtime_capabilities=RuntimeCapabilities(
            market_data=_MarketData(),
            signal_publisher=_SignalPublisher(),
            state_store=_NoopStateStore(),
            notification_publisher=_NoopNotificationPublisher(),
            secret_provider=_NoopSecretProvider(),
            logger=_NoopLogger(),
            metrics=InMemoryMetricsRecorder(),
            clock=_Clock(),
        ),
        market_data_source="BINANCE_SPOT",
    )


class _MarketData:
    async def get_latest_complete_snapshot(self, **kwargs):
        return _snapshot()

    async def build_context(self, **kwargs):
        return BotMarketDataContext(primary_snapshot=_snapshot())


class _SignalPublisher:
    async def publish(self, signal):
        raise AssertionError("Stage 23 must persist signals through manual-run persistence, not context fanout")


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


class _NoopLogger:
    def info(self, event: str, **fields: object) -> None:
        return None

    def warning(self, event: str, **fields: object) -> None:
        return None

    def error(self, event: str, **fields: object) -> None:
        return None


class _Clock:
    def now(self):
        return datetime(2026, 7, 16, tzinfo=UTC)


def _snapshot() -> BotMarketSnapshot:
    return BotMarketSnapshot(
        snapshot_id="snapshot-1",
        source="BINANCE_SPOT",
        canonical_symbol="ETHUSDT",
        provider_symbol="ETHUSDT",
        timeframe="1h",
        last_closed_candle_time=datetime(2026, 7, 16, 10, tzinfo=UTC),
        lookback_start_time=datetime(2026, 7, 16, 9, tzinfo=UTC),
        lookback_end_time=datetime(2026, 7, 16, 10, tzinfo=UTC),
        completeness_status="COMPLETE",
        snapshot_version=1,
        data_hash="hash",
        candles=(
            BotCandle(
                source="BINANCE_SPOT",
                canonical_symbol="ETHUSDT",
                timeframe="1h",
                open_time=datetime(2026, 7, 16, 9, tzinfo=UTC),
                close_time=datetime(2026, 7, 16, 10, tzinfo=UTC),
                open=Decimal("1"),
                high=Decimal("2"),
                low=Decimal("1"),
                close=Decimal("2"),
                volume=Decimal("10"),
            ),
        ),
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
        reason="stage_23",
        payload_schema="fixture.stage_23",
        payload_schema_version=1,
        payload={"price": Decimal("2")},
    )
