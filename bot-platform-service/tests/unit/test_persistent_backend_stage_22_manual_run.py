from __future__ import annotations

import asyncio
import json
from io import BytesIO
from datetime import UTC, datetime
from decimal import Decimal
from urllib.error import HTTPError

import pytest

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
    BotSignalPublishResult,
    BotStartResult,
    BotStopResult,
    BotTriggerType,
    BotValidationResult,
)
from bot_platform_service.infrastructure.market_data.errors import SnapshotStaleError
from bot_platform_service.observability.metrics import InMemoryMetricsRecorder
from bot_platform_service.runtime.container import BotPlatformRepositories, BotPlatformRuntimeContainer
from bot_platform_service.runtime.http_server import BotPlatformHttpServer
from bot_platform_service.config.settings import BotPlatformSettings
from bot_platform_service.infrastructure.market_data.http_snapshot_client import MarketDataHttpSnapshotClient


class _InstanceRepository:
    def __init__(self, status: BotInstanceStatus = BotInstanceStatus.ENABLED) -> None:
        self.status = status
        self.config = BotInstanceConfig(
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
        return self.status if instance_id == self.config.instance_id else None


class _RunRepository:
    def __init__(self, *, lock_acquired: bool = True) -> None:
        self.lock_acquired = lock_acquired
        self.created_runs: list[dict[str, object]] = []
        self.completed_runs: list[dict[str, object]] = []
        self.events: list[dict[str, object]] = []

    async def acquire_instance_run_lock(self, *, instance_id: str) -> bool:
        return self.lock_acquired

    async def create_run(self, **kwargs) -> bool:
        self.created_runs.append(dict(kwargs))
        return True

    async def complete_run(
        self,
        *,
        run_id: str,
        status: BotRunStatus,
        error_code: str | None = None,
        error_message_redacted: str | None = None,
    ) -> bool:
        self.completed_runs.append(
            {
                "run_id": run_id,
                "status": status,
                "error_code": error_code,
                "error_message_redacted": error_message_redacted,
            }
        )
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
    async def publish_persisted_signal(self, **kwargs) -> bool:
        return False


class _Resolver:
    def __init__(self, module: "_Module") -> None:
        self.module = module

    async def resolve(self, module_id: str):
        return self.module if module_id == self.module.module_id else None


class _Module:
    module_id = "spot_grid"

    def __init__(self) -> None:
        self.initialized = 0
        self.requests: list[BotRunRequest] = []

    async def validate_config(self, config: BotInstanceConfig) -> BotValidationResult:
        return BotValidationResult(valid=True)

    async def initialize(self, context) -> None:
        self.initialized += 1

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
            diagnostics={"manual": True},
        )

    async def start(self, request) -> BotStartResult:
        return BotStartResult(accepted=True, instance_id=request.instance_id)

    async def stop(self, instance_id: str) -> BotStopResult:
        return BotStopResult(accepted=True, instance_id=instance_id)

    async def health(self, instance_id: str) -> BotHealth:
        return BotHealth(instance_id=instance_id, module_id=self.module_id, status=BotHealthStatus.HEALTHY)


class _MarketData:
    def __init__(self, *, stale: bool = False) -> None:
        self.stale = stale

    async def get_latest_complete_snapshot(self, **kwargs):
        return _snapshot()

    async def build_context(self, **kwargs):
        if self.stale:
            raise SnapshotStaleError("snapshot is stale")
        return BotMarketDataContext(primary_snapshot=_snapshot())


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


def test_stage_22_manual_run_creates_bot_run_and_calls_adapter_without_signal_fanout() -> None:
    async def run() -> None:
        run_repository = _RunRepository()
        module = _Module()
        service = _service(run_repository=run_repository, module=module)

        result = await service.run_instance(ManualRunCommand(instance_id="instance-1", idempotency_key="manual-1"))

        assert result.accepted is True
        assert result.status is BotRunStatus.COMPLETE
        assert run_repository.created_runs[0]["trigger_type"] is BotTriggerType.MANUAL
        assert run_repository.created_runs[0]["snapshot_id"] == "snapshot-1"
        assert run_repository.completed_runs[0]["status"] is BotRunStatus.COMPLETE
        assert module.initialized == 1
        assert module.requests[0].market_data is not None

    asyncio.run(run())


def test_stage_22_duplicate_run_lock_prevents_adapter_execution() -> None:
    async def run() -> None:
        run_repository = _RunRepository(lock_acquired=False)
        module = _Module()
        service = _service(run_repository=run_repository, module=module)

        result = await service.run_instance(ManualRunCommand(instance_id="instance-1"))

        assert result.accepted is False
        assert result.duplicate is True
        assert result.error_code == "INSTANCE_RUN_LOCKED"
        assert run_repository.created_runs == []
        assert module.requests == []

    asyncio.run(run())


def test_stage_22_stale_snapshot_returns_controlled_error_without_run_creation() -> None:
    async def run() -> None:
        run_repository = _RunRepository()
        module = _Module()
        service = _service(run_repository=run_repository, module=module, market_data=_MarketData(stale=True))

        result = await service.run_instance(ManualRunCommand(instance_id="instance-1"))

        assert result.accepted is False
        assert result.error_code == "SNAPSHOT_STALE"
        assert run_repository.created_runs == []
        assert module.requests == []

    asyncio.run(run())


def test_stage_22_http_route_exposes_manual_run() -> None:
    async def run() -> None:
        service = _ManualRunService()
        server = BotPlatformHttpServer(container=_container(service))
        response = await server._route(
            method="POST",
            path="/admin/bot-instances/instance-1/run",
            body=json.dumps({"idempotency_key": "manual-1"}).encode("utf-8"),
        )
        body = json.loads(response.body.decode("utf-8"))

        assert response.status_code == 202
        assert body["accepted"] is True
        assert body["run_id"] == "run-1"
        assert service.commands == [ManualRunCommand(instance_id="instance-1", idempotency_key="manual-1", correlation_id=None)]

    asyncio.run(run())


def test_stage_22_http_snapshot_client_maps_stale_contract_to_controlled_error(monkeypatch) -> None:
    payload = json.dumps(
        {
            "contract_version": "market-snapshot.v1",
            "status": "stale",
            "reason": "latest complete snapshot is older than max_age_seconds",
        }
    ).encode("utf-8")

    def stale_response(*args, **kwargs):
        raise HTTPError(args[0], 503, "Service Unavailable", hdrs=None, fp=BytesIO(payload))

    monkeypatch.setattr("bot_platform_service.infrastructure.market_data.http_snapshot_client.urlopen", stale_response)
    client = MarketDataHttpSnapshotClient(base_url="http://market-data.local")

    with pytest.raises(SnapshotStaleError):
        client._get_latest_complete_snapshot_sync("BINANCE_SPOT", "ETHUSDT", "1h")


def _service(
    *,
    run_repository: _RunRepository,
    module: _Module,
    market_data: _MarketData | None = None,
) -> ManualBotRunService:
    return ManualBotRunService(
        instance_repository=_InstanceRepository(),
        run_repository=run_repository,
        signal_repository=_SignalRepository(),
        audit_repository=_AuditRepository(),
        signal_event_publisher=_SignalEventPublisher(),
        module_resolver=_Resolver(module),
        runtime_capabilities=RuntimeCapabilities(
            market_data=market_data or _MarketData(),
            signal_publisher=_NoopSignalPublisher(),
            state_store=_NoopStateStore(),
            notification_publisher=_NoopNotificationPublisher(),
            secret_provider=_NoopSecretProvider(),
            logger=_NoopLogger(),
            metrics=InMemoryMetricsRecorder(),
            clock=_Clock(),
        ),
        market_data_source="BINANCE_SPOT",
    )


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


class _ManualRunService:
    def __init__(self) -> None:
        self.commands: list[ManualRunCommand] = []

    async def run_instance(self, command: ManualRunCommand):
        from bot_platform_service.application import ManualRunResult

        self.commands.append(command)
        return ManualRunResult(True, command.instance_id, run_id="run-1", status=BotRunStatus.COMPLETE)


class _FakeConnection:
    async def commit(self) -> None:
        return None


class _FakeEngine:
    async def dispose(self) -> None:
        return None


def _container(service: _ManualRunService) -> BotPlatformRuntimeContainer:
    repository = object()
    return BotPlatformRuntimeContainer(
        settings=BotPlatformSettings.from_env(),
        engine=_FakeEngine(),
        connection=_FakeConnection(),
        repositories=BotPlatformRepositories(
            bot_modules=repository,
            bot_instances=repository,
            bot_audit_events=repository,
            bot_runs=repository,
            bot_signals=repository,
        ),
        admin_metadata_service=object(),
        admin_instance_service=object(),
        config_validation_service=object(),
        lifecycle_service=object(),
        manual_run_service=service,
        metrics=InMemoryMetricsRecorder(),
    )
