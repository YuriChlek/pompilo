from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

from bot_platform_service.application import ManualBotRunService, ManualRunCommand, RuntimeCapabilities
from bot_platform_service.config.settings import BotPlatformSettings
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
from bot_platform_service.infrastructure.signals import DisabledSignalEventPublisher, RedisStreamSignalEventPublisher
from bot_platform_service.observability.metrics import InMemoryMetricsRecorder


class _Redis:
    def __init__(self, *, fail_xadd_times: int = 0) -> None:
        self.keys: set[str] = set()
        self.events: list[tuple[str, dict[str, str]]] = []
        self.deleted: list[str] = []
        self.fail_xadd_times = fail_xadd_times

    async def set(self, key: str, value: str, *, nx: bool = False):
        if nx and key in self.keys:
            return False
        self.keys.add(key)
        return True

    async def xadd(self, stream_name: str, fields: dict[str, str]):
        if self.fail_xadd_times > 0:
            self.fail_xadd_times -= 1
            raise RuntimeError("redis unavailable")
        self.events.append((stream_name, dict(fields)))
        return "1-0"

    async def delete(self, key: str):
        self.deleted.append(key)
        self.keys.discard(key)
        return 1


def test_stage_24_redis_signal_event_publisher_is_duplicate_safe_and_retries() -> None:
    async def run() -> None:
        redis = _Redis(fail_xadd_times=1)
        publisher = RedisStreamSignalEventPublisher(
            redis_client=redis,
            stream_name="bot-platform-signals",
            max_retries=2,
            retry_backoff_seconds=0,
        )
        signal = _signal()

        first = await publisher.publish_persisted_signal(signal_id="sig-1", run_id="run-1", signal=signal)
        second = await publisher.publish_persisted_signal(signal_id="sig-1", run_id="run-1", signal=signal)

        assert first is True
        assert second is False
        assert len(redis.events) == 1
        assert redis.events[0][0] == "bot-platform-signals"
        assert redis.events[0][1]["idempotency_key"] == "bot-platform:signal-event:sig-1"

    asyncio.run(run())


def test_stage_24_redis_failure_releases_idempotency_key_for_retry() -> None:
    async def run() -> None:
        redis = _Redis(fail_xadd_times=2)
        publisher = RedisStreamSignalEventPublisher(
            redis_client=redis,
            stream_name="bot-platform-signals",
            max_retries=1,
            retry_backoff_seconds=0,
        )

        try:
            await publisher.publish_persisted_signal(signal_id="sig-1", run_id="run-1", signal=_signal())
        except RuntimeError:
            pass

        assert redis.deleted == ["bot-platform:signal-event:sig-1"]
        assert "bot-platform:signal-event:sig-1" not in redis.keys
        assert redis.events == []

    asyncio.run(run())


def test_stage_24_disabled_publisher_is_env_controlled(monkeypatch) -> None:
    monkeypatch.setenv("BOT_PLATFORM_SIGNAL_EVENTS_ENABLED", "false")
    settings = BotPlatformSettings.from_env()

    assert settings.signal_events.enabled is False
    assert isinstance(DisabledSignalEventPublisher(), DisabledSignalEventPublisher)


def test_stage_24_downstream_failure_does_not_lose_persisted_signal() -> None:
    async def run() -> None:
        signal_repository = _SignalRepository()
        service = _service(signal_repository=signal_repository, signal_event_publisher=_FailingSignalEventPublisher())

        result = await service.run_instance(ManualRunCommand(instance_id="instance-1", idempotency_key="manual-1"))

        assert result.accepted is True
        assert result.status is BotRunStatus.COMPLETE
        assert len(signal_repository.signals) == 1
        assert signal_repository.signals[0]["signal_id"].startswith("sig_")

    asyncio.run(run())


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
    async def acquire_instance_run_lock(self, *, instance_id: str) -> bool:
        return True

    async def create_run(self, **kwargs) -> bool:
        return True

    async def complete_run(self, **kwargs) -> bool:
        return True

    async def append_run_event(self, **kwargs) -> bool:
        return True


class _SignalRepository:
    def __init__(self) -> None:
        self.signals: list[dict[str, object]] = []

    async def publish_signal(self, **kwargs) -> str:
        self.signals.append(dict(kwargs))
        return str(kwargs["signal_id"])


class _AuditRepository:
    async def append_audit_event(self, **kwargs) -> bool:
        return True


class _FailingSignalEventPublisher:
    async def publish_persisted_signal(self, **kwargs) -> bool:
        raise RuntimeError("redis unavailable")


class _Resolver:
    async def resolve(self, module_id: str):
        return _Module()


class _Module:
    module_id = "spot_grid"

    async def validate_config(self, config: BotInstanceConfig) -> BotValidationResult:
        return BotValidationResult(valid=True)

    async def initialize(self, context) -> None:
        return None

    async def dry_run(self, request: BotRunRequest) -> BotRunResult:
        return await self.run_once(request)

    async def run_once(self, request: BotRunRequest) -> BotRunResult:
        return BotRunResult(
            run_id=request.run_id,
            instance_id=request.instance_id,
            module_id=request.module_id,
            mode=request.mode,
            status=BotRunStatus.COMPLETE,
            signals=(_signal(request),),
        )

    async def start(self, request) -> BotStartResult:
        return BotStartResult(accepted=True, instance_id=request.instance_id)

    async def stop(self, instance_id: str) -> BotStopResult:
        return BotStopResult(accepted=True, instance_id=instance_id)

    async def health(self, instance_id: str) -> BotHealth:
        return BotHealth(instance_id=instance_id, module_id=self.module_id, status=BotHealthStatus.HEALTHY)


def _service(*, signal_repository: _SignalRepository, signal_event_publisher) -> ManualBotRunService:
    return ManualBotRunService(
        instance_repository=_InstanceRepository(),
        run_repository=_RunRepository(),
        signal_repository=signal_repository,
        audit_repository=_AuditRepository(),
        signal_event_publisher=signal_event_publisher,
        module_resolver=_Resolver(),
        runtime_capabilities=RuntimeCapabilities(
            market_data=_MarketData(),
            signal_publisher=_ContextSignalPublisher(),
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


class _ContextSignalPublisher:
    async def publish(self, signal):
        raise AssertionError("Stage 24 must publish downstream only after signal persistence")


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


def _signal(request: BotRunRequest | None = None) -> BotSignal:
    return BotSignal.build(
        instance_id=request.instance_id if request is not None else "instance-1",
        module_id=request.module_id if request is not None else "spot_grid",
        symbol="ETHUSDT",
        timeframe="1h",
        snapshot_id="snapshot-1",
        signal_type=BotSignalType.ENTRY,
        side=BotSignalSide.BUY,
        confidence=Decimal("0.8"),
        reason="stage_24",
        payload_schema="fixture.stage_24",
        payload_schema_version=1,
        payload={"price": Decimal("2")},
    )
