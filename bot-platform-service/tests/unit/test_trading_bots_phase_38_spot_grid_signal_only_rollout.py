from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from bot_platform_service.application import ManualBotRunService, ManualRunCommand, RuntimeCapabilities
from bot_platform_service.domain import (
    BotCandle,
    BotInstanceConfig,
    BotInstanceStatus,
    BotMarketDataContext,
    BotMarketSnapshot,
    BotMode,
    BotRunStatus,
    BotSignal,
    BotSignalPublishStatus,
)
from bot_platform_service.trading_bots.spot_grid.adapter import SpotGridAdapter
from bot_platform_service.trading_bots.spot_grid.application import SpotGridTradingCycleService
from tests.fixtures.spot_grid_indicator_runtime import FakeStockIndicatorsRuntime


SERVICE_ROOT = Path(__file__).resolve().parents[2]
SPOT_GRID_SOURCE_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "trading_bots" / "spot_grid"
MIGRATION_ROLLOUT_DOC = SERVICE_ROOT / "docs" / "migration_rollout.md"


def test_phase_38_signal_only_persists_signals_and_events_for_subscribed_symbols() -> None:
    instance_repository = _InstanceRepository(
        (
            _instance("spot-grid-eth", "ETHUSDT"),
            _instance("spot-grid-btc", "BTCUSDT"),
        )
    )
    run_repository = _RunRepository()
    signal_repository = _SignalRepository()
    audit_repository = _AuditRepository()
    signal_event_publisher = _SignalEventPublisher()
    service = _service(
        instance_repository=instance_repository,
        run_repository=run_repository,
        signal_repository=signal_repository,
        audit_repository=audit_repository,
        signal_event_publisher=signal_event_publisher,
    )

    results = asyncio.run(
        _run_all(
            service,
            (
                ManualRunCommand(instance_id="spot-grid-eth", idempotency_key="phase-38-eth", correlation_id="corr-38"),
                ManualRunCommand(instance_id="spot-grid-btc", idempotency_key="phase-38-btc", correlation_id="corr-38"),
            ),
        )
    )

    assert [result.status for result in results] == [BotRunStatus.COMPLETE, BotRunStatus.COMPLETE]
    assert {call["signal"].symbol for call in signal_repository.calls} == {"ETHUSDT", "BTCUSDT"}
    assert {event["fields"]["symbol"] for event in signal_event_publisher.events} == {"ETHUSDT", "BTCUSDT"}
    assert len(signal_event_publisher.events) == len(signal_repository.persisted_by_key)
    assert len(audit_repository.events_by_id) == len(signal_repository.persisted_by_key)
    assert all(call["status"] is BotSignalPublishStatus.PUBLISHED for call in signal_repository.calls)
    assert all(event["fields"]["event_type"] == "bot_signal.persisted.v1" for event in signal_event_publisher.events)
    assert all(event["fields"]["payload_schema"] == "spot_grid.position_intent" for event in signal_event_publisher.events)
    assert all("payload" not in event["fields"] for event in signal_event_publisher.events)
    assert all("payload_json" not in event["fields"] for event in signal_event_publisher.events)
    assert all("target_price" not in event["fields"] for event in signal_event_publisher.events)
    assert _source_forbidden_execution_terms() == []

    completed_events = run_repository.completed_events
    assert len(completed_events) == 2
    assert all(event["signal_count"] > 0 for event in completed_events)
    assert all(event["persisted_signal_count"] == event["signal_count"] for event in completed_events)
    assert {event["diagnostics"]["subscribed_symbols"] for event in completed_events} == {
        ("ETHUSDT",),
        ("BTCUSDT",),
    }


def test_phase_38_signal_event_stream_is_duplicate_stable_and_execution_disconnected() -> None:
    instance_repository = _InstanceRepository((_instance("spot-grid-eth", "ETHUSDT"),))
    signal_repository = _SignalRepository()
    signal_event_publisher = _SignalEventPublisher()
    service = _service(
        instance_repository=instance_repository,
        run_repository=_RunRepository(),
        signal_repository=signal_repository,
        audit_repository=_AuditRepository(),
        signal_event_publisher=signal_event_publisher,
    )

    results = asyncio.run(
        _run_all(
            service,
            (
                ManualRunCommand(instance_id="spot-grid-eth", idempotency_key="phase-38-eth-a"),
                ManualRunCommand(instance_id="spot-grid-eth", idempotency_key="phase-38-eth-b"),
            ),
        )
    )

    assert [result.status for result in results] == [BotRunStatus.COMPLETE, BotRunStatus.COMPLETE]
    assert len(signal_repository.calls) == len(signal_event_publisher.publish_attempts)
    assert len(signal_event_publisher.events) == len(signal_repository.persisted_by_key)
    assert any(attempt["published"] is False for attempt in signal_event_publisher.publish_attempts)
    assert _source_forbidden_execution_terms() == []


def test_phase_38_signal_only_rollout_is_documented() -> None:
    doc = MIGRATION_ROLLOUT_DOC.read_text(encoding="utf-8")

    assert "`signal_only` persists standardized `BotSignal` records" in doc
    assert "Persisted signal events use `bot_signal.persisted.v1` metadata only" in doc
    assert "execution service remains disconnected" in doc


async def _run_all(service: ManualBotRunService, commands: tuple[ManualRunCommand, ...]):
    results = []
    for command in commands:
        results.append(await service.run_instance(command))
    return tuple(results)


def _service(
    *,
    instance_repository: "_InstanceRepository",
    run_repository: "_RunRepository",
    signal_repository: "_SignalRepository",
    audit_repository: "_AuditRepository",
    signal_event_publisher: "_SignalEventPublisher",
) -> ManualBotRunService:
    return ManualBotRunService(
        instance_repository=instance_repository,
        run_repository=run_repository,
        signal_repository=signal_repository,
        audit_repository=audit_repository,
        signal_event_publisher=signal_event_publisher,
        module_resolver=_Resolver(),
        runtime_capabilities=RuntimeCapabilities(
            market_data=_MarketData(),
            signal_publisher=_ContextSignalPublisher(),
            state_store=_StateStore(),
            notification_publisher=_NotificationPublisher(),
            secret_provider=_NoopSecretProvider(),
            logger=_Logger(),
            metrics=_Metrics(),
            clock=_Clock(),
        ),
        market_data_source="binance_spot",
    )


def _instance(instance_id: str, symbol: str) -> BotInstanceConfig:
    return BotInstanceConfig(
        instance_id=instance_id,
        module_id="spot_grid",
        mode=BotMode.SIGNAL_ONLY,
        symbols=(symbol,),
        timeframes=("1h",),
        config_schema_version=1,
        config={
            "symbols": [symbol],
            "primary_timeframe": "1h",
            "supporting_timeframes": [],
            "max_grid_levels": 2,
            "max_position_fraction": "0.10",
        },
    )


class _InstanceRepository:
    def __init__(self, configs: tuple[BotInstanceConfig, ...]) -> None:
        self.configs = {config.instance_id: config for config in configs}

    async def get_instance_config(self, instance_id: str) -> BotInstanceConfig | None:
        return self.configs.get(instance_id)

    async def get_instance_status(self, instance_id: str) -> BotInstanceStatus | None:
        return BotInstanceStatus.ENABLED if instance_id in self.configs else None


class _RunRepository:
    def __init__(self) -> None:
        self.locks: set[str] = set()
        self.events: list[dict[str, object]] = []

    @property
    def completed_events(self) -> list[dict[str, object]]:
        return [event["payload_json"] for event in self.events if event["event_type"] == "COMPLETED"]

    async def acquire_instance_run_lock(self, *, instance_id: str) -> bool:
        self.locks.add(instance_id)
        return True

    async def create_run(self, **kwargs) -> bool:
        del kwargs
        return True

    async def complete_run(self, **kwargs) -> bool:
        del kwargs
        return True

    async def append_run_event(self, **kwargs) -> bool:
        self.events.append(dict(kwargs))
        return True


class _SignalRepository:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.persisted_by_key: dict[str, str] = {}

    async def publish_signal(
        self,
        *,
        signal_id: str,
        run_id: str,
        signal: BotSignal,
        status: BotSignalPublishStatus = BotSignalPublishStatus.PUBLISHED,
        correlation_id: str | None = None,
    ) -> str:
        self.calls.append(
            {
                "signal_id": signal_id,
                "run_id": run_id,
                "signal": signal,
                "status": status,
                "correlation_id": correlation_id,
            }
        )
        return self.persisted_by_key.setdefault(signal.signal_key, signal_id)


class _AuditRepository:
    def __init__(self) -> None:
        self.events_by_id: dict[str, dict[str, object]] = {}

    async def append_audit_event(self, *, event_id: str, **kwargs) -> bool:
        if event_id in self.events_by_id:
            return False
        self.events_by_id[event_id] = dict(kwargs)
        return True


class _SignalEventPublisher:
    def __init__(self) -> None:
        self.published_signal_ids: set[str] = set()
        self.publish_attempts: list[dict[str, object]] = []
        self.events: list[dict[str, object]] = []

    async def publish_persisted_signal(
        self,
        *,
        signal_id: str,
        run_id: str,
        signal: BotSignal,
        correlation_id: str | None = None,
    ) -> bool:
        published = signal_id not in self.published_signal_ids
        self.publish_attempts.append({"signal_id": signal_id, "published": published})
        if not published:
            return False
        self.published_signal_ids.add(signal_id)
        self.events.append(
            {
                "fields": {
                    "event_type": "bot_signal.persisted.v1",
                    "signal_id": signal_id,
                    "signal_key": signal.signal_key,
                    "run_id": run_id,
                    "instance_id": signal.instance_id,
                    "module_id": signal.module_id,
                    "symbol": signal.symbol,
                    "timeframe": signal.timeframe,
                    "snapshot_id": signal.snapshot_id,
                    "signal_type": signal.signal_type.value,
                    "side": signal.side.value if signal.side is not None else "",
                    "payload_schema": signal.payload_schema,
                    "payload_schema_version": str(signal.payload_schema_version),
                    "payload_hash": signal.payload_hash,
                    "correlation_id": correlation_id or "",
                }
            }
        )
        return True


class _Resolver:
    def __init__(self) -> None:
        self.adapter = SpotGridAdapter(
            cycle_service=SpotGridTradingCycleService(indicator_runtime=_SignalOnlyIndicatorRuntime())
        )

    async def resolve(self, module_id: str):
        return self.adapter if module_id == "spot_grid" else None


class _MarketData:
    async def build_context(self, **kwargs):
        symbol = kwargs["canonical_symbol"]
        primary_timeframe = kwargs["primary_timeframe"]
        supporting_timeframes = tuple(kwargs["supporting_timeframes"])
        return BotMarketDataContext(
            primary_snapshot=_snapshot(symbol=symbol, timeframe=primary_timeframe),
            supporting_snapshots=tuple(_snapshot(symbol=symbol, timeframe=timeframe) for timeframe in supporting_timeframes),
        )


class _ContextSignalPublisher:
    async def publish(self, signal: BotSignal):
        del signal
        raise AssertionError("Manual signal_only rollout persists through ManualBotRunService repositories")


class _StateStore:
    async def load(self, *, instance_id: str, namespace: str, state_key: str) -> object | None:
        del instance_id, namespace, state_key
        return None

    async def save(self, *, instance_id: str, namespace: str, state_key: str, value: object) -> None:
        del instance_id, namespace, state_key, value


class _NotificationPublisher:
    async def publish(self, *, instance_id: str, message_type: str, payload: object) -> None:
        del instance_id, message_type, payload


class _SignalOnlyIndicatorRuntime(FakeStockIndicatorsRuntime):
    def rsi_last(self, quotes, length: int):
        del quotes, length
        return Decimal("30")

    def realized_volatility_last(self, quotes, length: int):
        del quotes, length
        return Decimal("0.01")


class _NoopSecretProvider:
    async def resolve(self, *, instance_id: str, secret_name: str) -> str:
        del instance_id, secret_name
        raise RuntimeError("not configured")


class _Logger:
    def info(self, event: str, **fields: object) -> None:
        del event, fields

    def warning(self, event: str, **fields: object) -> None:
        del event, fields

    def error(self, event: str, **fields: object) -> None:
        del event, fields


class _Metrics:
    def __init__(self) -> None:
        self.counts: list[tuple[str, dict[str, str] | None]] = []

    def increment(self, name: str, value: int = 1, tags: dict[str, str] | None = None) -> None:
        del value
        self.counts.append((name, tags))

    def observe(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        del name, value, tags


class _Clock:
    def now(self):
        return datetime(2026, 7, 26, tzinfo=UTC)


def _snapshot(*, symbol: str, timeframe: str) -> BotMarketSnapshot:
    now = datetime(2026, 7, 26, tzinfo=UTC)
    candles = tuple(_candle(symbol=symbol, timeframe=timeframe, index=index, now=now) for index in range(30))
    return BotMarketSnapshot(
        snapshot_id=f"phase-38-{symbol.lower()}-{timeframe}",
        source="binance_spot",
        canonical_symbol=symbol,
        provider_symbol=symbol,
        timeframe=timeframe,
        last_closed_candle_time=now + timedelta(hours=29),
        lookback_start_time=now,
        lookback_end_time=now + timedelta(hours=29),
        completeness_status="COMPLETE",
        snapshot_version=1,
        data_hash=f"hash-phase-38-{symbol.lower()}-{timeframe}",
        candles=candles,
    )


def _candle(*, symbol: str, timeframe: str, index: int, now: datetime) -> BotCandle:
    opened_at = now + timedelta(hours=index)
    close = Decimal("100")
    return BotCandle(
        source="binance_spot",
        canonical_symbol=symbol,
        timeframe=timeframe,
        open_time=opened_at,
        close_time=opened_at + timedelta(hours=1),
        open=close,
        high=close + Decimal("5"),
        low=close - Decimal("5"),
        close=close,
        volume=Decimal("1000"),
    )


def _source_forbidden_execution_terms() -> list[str]:
    forbidden_terms = (
        "ccxt",
        "pybit",
        "fetch_balance",
        "get_wallet_balance",
        "get_positions",
        "place_order",
        "cancel_order",
        "create_order",
        "create_market_buy_order",
        "create_market_sell_order",
        "BybitSpotExecutionService",
    )
    source = "\n".join(path.read_text(encoding="utf-8") for path in sorted(SPOT_GRID_SOURCE_ROOT.rglob("*.py")))
    return [term for term in forbidden_terms if term in source]
