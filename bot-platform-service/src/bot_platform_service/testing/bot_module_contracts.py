from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from bot_platform_service.application import RuntimeCapabilities, RuntimeContextFactory, RuntimeContextRequest
from bot_platform_service.domain import (
    BotCandle,
    BotHealth,
    BotHealthStatus,
    BotInstanceConfig,
    BotManifest,
    BotMarketDataContext,
    BotMarketSnapshot,
    BotMode,
    BotModule,
    BotPermission,
    BotRunRequest,
    BotRunResult,
    BotRunStatus,
    BotSignal,
    BotSignalPublishResult,
    BotSignalSide,
    BotSignalType,
    BotStartRequest,
    BotStartResult,
    BotStopResult,
    BotTriggerType,
    BotValidationResult,
    PermissionDeniedError,
)
from bot_platform_service.infrastructure.signals import PersistentSignalPublisher
from bot_platform_service.registry import build_registration


class ContractViolation(AssertionError):
    """Raised when a bot module violates the platform contract."""


@dataclass(frozen=True, slots=True)
class BotModuleContractCase:
    """Inputs required to validate one bot module contract."""

    manifest: BotManifest
    adapter_path: str
    config: BotInstanceConfig
    module: BotModule
    source_paths: tuple[Path, ...] = ()


class BotModuleContractHarness:
    """Reusable contract harness for future Python bot modules."""

    def assert_contract(self, case: BotModuleContractCase) -> None:
        """Run all contract checks for one bot module."""

        self.assert_manifest(case)
        self.assert_lifecycle(case)
        self.assert_market_data_and_signal_dtos(case)
        self.assert_permission_scoping()
        self.assert_idempotent_signal_publish()
        self.assert_no_forbidden_source_terms(case.source_paths)

    def assert_manifest(self, case: BotModuleContractCase) -> None:
        """Validate manifest metadata and adapter path."""

        registration = build_registration(
            {
                "module_id": case.manifest.module_id,
                "display_name": case.manifest.display_name,
                "version": case.manifest.version,
                "supported_modes": tuple(mode.value for mode in case.manifest.supported_modes),
                "required_timeframes": case.manifest.required_timeframes,
                "required_market_data": case.manifest.required_market_data,
                "supports_multi_symbol": case.manifest.supports_multi_symbol,
                "config_schema_version": case.manifest.config_schema_version,
                "status": case.manifest.status.value,
            },
            adapter_path=case.adapter_path,
        )
        if registration.manifest.module_id != case.config.module_id:
            raise ContractViolation("manifest module_id must match instance config module_id")

    def assert_lifecycle(self, case: BotModuleContractCase) -> None:
        """Validate BotModule lifecycle method conformance."""

        context = _contract_context(case.config.instance_id, permissions=frozenset(BotPermission))
        validation = asyncio.run(case.module.validate_config(case.config))
        if not isinstance(validation, BotValidationResult) or not validation.valid:
            raise ContractViolation("validate_config must return a valid BotValidationResult")
        asyncio.run(case.module.initialize(context))
        start = asyncio.run(case.module.start(BotStartRequest(case.config.instance_id, case.config.module_id, case.config.mode)))
        stop = asyncio.run(case.module.stop(case.config.instance_id))
        health = asyncio.run(case.module.health(case.config.instance_id))
        if not isinstance(start, BotStartResult):
            raise ContractViolation("start must return BotStartResult")
        if not isinstance(stop, BotStopResult):
            raise ContractViolation("stop must return BotStopResult")
        if not isinstance(health, BotHealth) or health.instance_id != case.config.instance_id:
            raise ContractViolation("health must return BotHealth for the instance")

    def assert_market_data_and_signal_dtos(self, case: BotModuleContractCase) -> None:
        """Validate run result, market-data, and signal DTO conformance."""

        request = BotRunRequest(
            run_id="contract-run-1",
            instance_id=case.config.instance_id,
            module_id=case.config.module_id,
            mode=BotMode.DRY_RUN,
            trigger_type=BotTriggerType.MANUAL,
            config=case.config.config,
            market_data=BotMarketDataContext(primary_snapshot=_snapshot(case.config.symbols[0], case.config.timeframes[0])),
        )
        result = asyncio.run(case.module.dry_run(request))
        if not isinstance(result, BotRunResult):
            raise ContractViolation("dry_run must return BotRunResult")
        if result.run_id != request.run_id or result.instance_id != request.instance_id or result.module_id != request.module_id:
            raise ContractViolation("BotRunResult identity fields must match request")
        if not isinstance(result.status, BotRunStatus):
            raise ContractViolation("BotRunResult status must be BotRunStatus")
        for signal in result.signals:
            _assert_signal_contract(signal)

    def assert_permission_scoping(self) -> None:
        """Validate runtime permission wrappers for module-facing capabilities."""

        signal = _signal()
        context = _contract_context("instance-1", permissions=frozenset({BotPermission.SEND_NOTIFICATIONS}))
        asyncio.run(context.notification_publisher.publish(instance_id="instance-1", message_type="contract", payload={}))
        try:
            asyncio.run(context.signal_publisher.publish(signal))
        except PermissionDeniedError as exc:
            if exc.permission is not BotPermission.PUBLISH_SIGNALS:
                raise ContractViolation("signal publisher must require publish_signals permission") from exc
        else:
            raise ContractViolation("signal publisher must reject missing publish_signals permission")

    def assert_idempotent_signal_publish(self) -> None:
        """Validate duplicate signal publish returns the same signal id."""

        repository = _IdempotentSignalRepository()
        audit = _AuditRepository()
        publisher = PersistentSignalPublisher(signal_repository=repository, audit_repository=audit, run_id="contract-run-1")
        signal = _signal()
        first = asyncio.run(publisher.publish(signal))
        second = asyncio.run(publisher.publish(signal))
        if first.signal_id != second.signal_id:
            raise ContractViolation("duplicate signal publish must return existing signal_id")
        if len(audit.events_by_id) != 1:
            raise ContractViolation("persisted signal must have one idempotent audit event")

    def assert_no_forbidden_source_terms(self, source_paths: tuple[Path, ...]) -> None:
        """Reject direct DB, market-data sync, and private exchange construction."""

        forbidden_terms = (
            "asyncpg.connect",
            "create_engine(",
            "BinanceMarketDataSynchronizer",
            "BybitSpot",
            "place_order",
            "cancel_order",
            "create_market_buy_order",
            "create_market_sell_order",
        )
        violations: list[str] = []
        for path in source_paths:
            text = path.read_text(encoding="utf-8")
            found = [term for term in forbidden_terms if term in text]
            if found:
                violations.append(f"{path}: {', '.join(found)}")
        if violations:
            raise ContractViolation("; ".join(violations))


def _assert_signal_contract(signal: BotSignal) -> None:
    if not isinstance(signal, BotSignal):
        raise ContractViolation("signals must contain BotSignal items")
    if len(signal.signal_key) != 64:
        raise ContractViolation("BotSignal signal_key must be deterministic sha256 hex")
    if signal.payload_schema_version <= 0:
        raise ContractViolation("BotSignal payload_schema_version must be positive")
    if signal.payload_hash != signal.payload_hash.lower() or len(signal.payload_hash) != 64:
        raise ContractViolation("BotSignal payload_hash must be sha256 hex")


def _contract_context(instance_id: str, *, permissions: frozenset[BotPermission]):
    capabilities = RuntimeCapabilities(
        market_data=_MarketData(),
        signal_publisher=_SignalPublisher(),
        state_store=_StateStore(),
        notification_publisher=_NotificationPublisher(),
        secret_provider=_SecretProvider(),
        logger=_Logger(),
        metrics=_Metrics(),
        clock=_Clock(),
    )
    return RuntimeContextFactory().build(
        RuntimeContextRequest(
            instance_id=instance_id,
            permissions=permissions,
            capabilities=capabilities,
        )
    )


def _snapshot(symbol: str = "ETHUSDT", timeframe: str = "1h") -> BotMarketSnapshot:
    now = datetime(2026, 7, 14, tzinfo=UTC)
    candle = BotCandle(
        source="binance_spot",
        canonical_symbol=symbol.upper(),
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
        snapshot_id=f"snapshot-{symbol}-{timeframe}",
        source="binance_spot",
        canonical_symbol=symbol.upper(),
        provider_symbol=symbol.upper(),
        timeframe=timeframe,
        last_closed_candle_time=now,
        lookback_start_time=now,
        lookback_end_time=now,
        completeness_status="COMPLETE",
        snapshot_version=1,
        data_hash="hash",
        candles=(candle,),
    )


def _signal() -> BotSignal:
    return BotSignal.build(
        instance_id="instance-1",
        module_id="fixture_bot",
        symbol="ETHUSDT",
        timeframe="1h",
        snapshot_id="snapshot-1",
        signal_type=BotSignalType.ENTRY,
        side=BotSignalSide.BUY,
        confidence=Decimal("0.9"),
        reason="contract_fixture",
        payload_schema="fixture.signal",
        payload_schema_version=1,
        payload={"price": Decimal("100")},
    )


class _MarketData:
    async def get_latest_complete_snapshot(self, *, source, canonical_symbol, timeframe, min_snapshot_version=None):
        return _snapshot(canonical_symbol, timeframe)

    async def build_context(self, *, source, canonical_symbol, primary_timeframe, supporting_timeframes):
        return BotMarketDataContext(primary_snapshot=_snapshot(canonical_symbol, primary_timeframe))


class _SignalPublisher:
    async def publish(self, signal: BotSignal) -> BotSignalPublishResult:
        return BotSignalPublishResult(accepted=True, signal_id="contract-signal-1")


class _StateStore:
    async def load(self, *, instance_id: str, namespace: str, state_key: str) -> object | None:
        return None

    async def save(self, *, instance_id: str, namespace: str, state_key: str, value: object) -> None:
        return None


class _NotificationPublisher:
    async def publish(self, *, instance_id: str, message_type: str, payload: object) -> None:
        return None


class _SecretProvider:
    async def resolve(self, *, instance_id: str, secret_name: str) -> str:
        return f"{instance_id}:{secret_name}"


class _Logger:
    def info(self, event: str, **fields: object) -> None:
        pass

    def warning(self, event: str, **fields: object) -> None:
        pass

    def error(self, event: str, **fields: object) -> None:
        pass


class _Metrics:
    def increment(self, name: str, *, tags: dict[str, str] | None = None) -> None:
        pass

    def observe(self, name: str, value: float, *, tags: dict[str, str] | None = None) -> None:
        pass


class _Clock:
    def now(self) -> object:
        return datetime(2026, 7, 14, tzinfo=UTC)


class _IdempotentSignalRepository:
    def __init__(self) -> None:
        self.signal_ids_by_key: dict[str, str] = {}

    async def publish_signal(self, *, signal_id, run_id, signal, status, correlation_id=None):
        return self.signal_ids_by_key.setdefault(signal.signal_key, signal_id)


class _AuditRepository:
    def __init__(self) -> None:
        self.events_by_id: dict[str, dict[str, object]] = {}

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
        if event_id in self.events_by_id:
            return False
        self.events_by_id[event_id] = payload_json
        return True
