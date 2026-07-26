from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from bot_platform_service.application import BotRunOrchestrationService, PollingScheduleTick, RuntimeCapabilities
from bot_platform_service.domain import (
    BotCandle,
    BotInstanceConfig,
    BotMarketDataContext,
    BotMarketSnapshot,
    BotMode,
    BotRunStatus,
    BotSignal,
    BotSignalType,
    BotTriggerType,
)
from bot_platform_service.trading_bots.spot_grid.adapter import SpotGridAdapter
from bot_platform_service.trading_bots.spot_grid.application import SpotGridTradingCycleService
from tests.fixtures.spot_grid_indicator_runtime import FakeStockIndicatorsRuntime


SERVICE_ROOT = Path(__file__).resolve().parents[2]
SPOT_GRID_SOURCE_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "trading_bots" / "spot_grid"


def test_phase_36_spot_grid_dry_run_instance_works_with_real_platform_context() -> None:
    run_repository = _RunRepository()
    market_data = _MarketData()
    signal_publisher = _SignalPublisher()
    state_store = _StateStore()
    service = _service(
        run_repository=run_repository,
        market_data=market_data,
        signal_publisher=signal_publisher,
        state_store=state_store,
    )
    instance = BotInstanceConfig(
        instance_id="spot-grid-dry-run",
        module_id="spot_grid",
        mode=BotMode.DRY_RUN,
        symbols=("ETHUSDT",),
        timeframes=("1h",),
        config_schema_version=1,
        config={
            "symbols": ["ETHUSDT"],
            "primary_timeframe": "1h",
            "supporting_timeframes": [],
            "max_grid_levels": 2,
            "max_position_fraction": "0.10",
        },
    )

    results = asyncio.run(service.run_polling_tick(PollingScheduleTick("phase-36-dry-run"), (instance,)))

    assert len(results) == 1
    assert results[0].status is BotRunStatus.COMPLETE
    assert market_data.context_calls == [("ETHUSDT", "1h", ())]
    assert len(run_repository.created_runs) == 1
    assert run_repository.completed[results[0].run_id] is BotRunStatus.COMPLETE
    completed_event = run_repository.completed_events[0]
    diagnostics = completed_event["diagnostics"]
    assert completed_event["status"] == BotRunStatus.COMPLETE.value
    assert completed_event["signal_count"] > 0
    assert completed_event["signal_publish_count"] == 0
    assert completed_event["state_changes_applied"] > 0
    assert diagnostics["mode"] == BotMode.DRY_RUN.value
    assert diagnostics["subscribed_symbols"] == ("ETHUSDT",)
    assert diagnostics["signal_count"] == completed_event["signal_count"]
    assert diagnostics["portfolio_context"]["fallback"] == "empty_conservative"
    assert {saved[1] for saved in state_store.saved} == {"spot_grid"}
    assert {saved[2] for saved in state_store.saved} >= {
        "ETHUSDT:1h:last_plan",
        "ETHUSDT:1h:regime_state",
        "ETHUSDT:1h:cooldown_state",
    }
    assert signal_publisher.published == []


def test_phase_36_dry_run_signals_are_execution_neutral_and_do_not_open_positions() -> None:
    run_repository = _RunRepository()
    signal_publisher = _SignalPublisher()
    state_store = _StateStore()
    service = _service(
        run_repository=run_repository,
        market_data=_MarketData(),
        signal_publisher=signal_publisher,
        state_store=state_store,
    )
    instance = BotInstanceConfig(
        instance_id="spot-grid-dry-run",
        module_id="spot_grid",
        mode=BotMode.DRY_RUN,
        symbols=("ETHUSDT",),
        timeframes=("1h",),
        config_schema_version=1,
        config={"supporting_timeframes": [], "max_grid_levels": 2},
    )

    asyncio.run(service.run_polling_tick(PollingScheduleTick("phase-36-dry-run"), (instance,)))

    diagnostics = run_repository.completed_events[0]["diagnostics"]
    assert diagnostics["signal_count"] > 0
    assert signal_publisher.published == []
    assert _source_forbidden_execution_terms() == []


def _service(
    *,
    run_repository: "_RunRepository",
    market_data: "_MarketData",
    signal_publisher: "_SignalPublisher",
    state_store: "_StateStore",
) -> BotRunOrchestrationService:
    return BotRunOrchestrationService(
        instance_repository=_InstanceRepository(),
        run_repository=run_repository,
        module_resolver=_Resolver(),
        runtime_capabilities=RuntimeCapabilities(
            market_data=market_data,
            signal_publisher=signal_publisher,
            state_store=state_store,
            notification_publisher=_NoopNotificationPublisher(),
            secret_provider=_NoopSecretProvider(),
            logger=_NoopLogger(),
            metrics=_NoopMetrics(),
            clock=_Clock(),
        ),
    )


class _InstanceRepository:
    async def list_enabled_instances_for_snapshot(self, *, source: str, canonical_symbol: str, timeframe: str):
        del source, canonical_symbol, timeframe
        return ()


class _RunRepository:
    def __init__(self) -> None:
        self.created_keys: set[str] = set()
        self.created_runs: list[dict[str, object]] = []
        self.completed: dict[str, BotRunStatus] = {}
        self.events: list[dict[str, object]] = []

    @property
    def completed_events(self) -> list[dict[str, object]]:
        return [event["payload_json"] for event in self.events if event["event_type"] == "COMPLETED"]

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
        del status, trigger_event_id, snapshot_id, correlation_id
        key = str(idempotency_key)
        if key in self.created_keys:
            return False
        self.created_keys.add(key)
        self.created_runs.append(
            {
                "run_id": run_id,
                "instance_id": instance_id,
                "module_id": module_id,
                "trigger_type": trigger_type,
            }
        )
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
        self.events.append({"event_type": event_type, "payload_json": dict(payload_json)})
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


class _Resolver:
    def __init__(self) -> None:
        self.adapter = SpotGridAdapter(
            cycle_service=SpotGridTradingCycleService(indicator_runtime=_DryRunIndicatorRuntime())
        )

    async def resolve(self, module_id: str):
        return self.adapter if module_id == "spot_grid" else None


class _MarketData:
    def __init__(self) -> None:
        self.context_calls: list[tuple[str, str, tuple[str, ...]]] = []

    async def get_latest_complete_snapshot(self, **kwargs):
        return _snapshot(symbol=kwargs["canonical_symbol"], timeframe=kwargs["timeframe"])

    async def build_context(self, **kwargs):
        symbol = kwargs["canonical_symbol"]
        primary_timeframe = kwargs["primary_timeframe"]
        supporting_timeframes = tuple(kwargs["supporting_timeframes"])
        self.context_calls.append((symbol, primary_timeframe, supporting_timeframes))
        return BotMarketDataContext(
            primary_snapshot=_snapshot(symbol=symbol, timeframe=primary_timeframe),
            supporting_snapshots=tuple(_snapshot(symbol=symbol, timeframe=timeframe) for timeframe in supporting_timeframes),
        )


class _SignalPublisher:
    def __init__(self) -> None:
        self.published: list[BotSignal] = []

    async def publish(self, signal: BotSignal):
        self.published.append(signal)
        raise AssertionError("dry_run orchestration must not publish persisted signals")


class _StateStore:
    def __init__(self) -> None:
        self.saved: list[tuple[str, str, str, object]] = []

    async def load(self, *, instance_id: str, namespace: str, state_key: str) -> object | None:
        del instance_id, namespace, state_key
        return None

    async def save(self, *, instance_id: str, namespace: str, state_key: str, value: object) -> None:
        self.saved.append((instance_id, namespace, state_key, value))


class _DryRunIndicatorRuntime(FakeStockIndicatorsRuntime):
    def rsi_last(self, quotes, length: int):
        del quotes, length
        return Decimal("30")

    def realized_volatility_last(self, quotes, length: int):
        del quotes, length
        return Decimal("0.01")


class _NoopNotificationPublisher:
    async def publish(self, *, instance_id: str, message_type: str, payload: object) -> None:
        del instance_id, message_type, payload


class _NoopSecretProvider:
    async def resolve(self, *, instance_id: str, secret_name: str) -> str:
        del instance_id, secret_name
        raise RuntimeError("not configured")


class _NoopLogger:
    def info(self, event: str, **fields: object) -> None:
        del event, fields

    def warning(self, event: str, **fields: object) -> None:
        del event, fields

    def error(self, event: str, **fields: object) -> None:
        del event, fields


class _NoopMetrics:
    def increment(self, name: str, value: int = 1, tags: dict[str, str] | None = None) -> None:
        del name, value, tags

    def observe(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        del name, value, tags


class _Clock:
    def now(self):
        return datetime(2026, 7, 26, tzinfo=UTC)


def _snapshot(*, symbol: str, timeframe: str) -> BotMarketSnapshot:
    now = datetime(2026, 7, 26, tzinfo=UTC)
    candles = tuple(_candle(symbol=symbol, timeframe=timeframe, index=index, now=now) for index in range(30))
    return BotMarketSnapshot(
        snapshot_id=f"phase-36-{symbol.lower()}-{timeframe}",
        source="binance_spot",
        canonical_symbol=symbol,
        provider_symbol=symbol,
        timeframe=timeframe,
        last_closed_candle_time=now + timedelta(hours=29),
        lookback_start_time=now,
        lookback_end_time=now + timedelta(hours=29),
        completeness_status="COMPLETE",
        snapshot_version=1,
        data_hash=f"hash-phase-36-{symbol.lower()}-{timeframe}",
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
