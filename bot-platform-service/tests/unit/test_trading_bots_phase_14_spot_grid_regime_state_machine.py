from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from bot_platform_service.domain import (
    BotCandle,
    BotMarketDataContext,
    BotMarketSnapshot,
    BotMode,
    BotRunRequest,
    BotRunStatus,
    BotRuntimeContext,
    BotTriggerType,
)
from bot_platform_service.trading_bots.spot_grid.adapter import SpotGridAdapter
from bot_platform_service.trading_bots.spot_grid.application import SpotGridTradingCycleService
from bot_platform_service.trading_bots.spot_grid.domain import (
    MarketRegime,
    RegimeSnapshot,
    regime_state_key,
    resolve_regime_state,
)
from bot_platform_service.trading_bots.spot_grid.infrastructure import PlatformSnapshotIndicatorAdapter
from tests.fixtures.spot_grid_indicator_runtime import FakeStockIndicatorsRuntime


SERVICE_ROOT = Path(__file__).resolve().parents[2]
SPOT_GRID_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "trading_bots" / "spot_grid"
INDICATOR_RUNTIME = FakeStockIndicatorsRuntime()


def test_phase_14_regime_state_requires_confirmation_between_close_states() -> None:
    first = resolve_regime_state(
        symbol="ETHUSDT",
        timeframe="1h",
        detected=_detected(MarketRegime.UPTREND),
        snapshot_id="snapshot-2",
        snapshot_version=2,
        data_hash="hash-2",
        previous_state={
            "effective_regime": "range",
            "pending_regime": None,
            "pending_confirmation_count": 0,
            "snapshot_id": "snapshot-1",
        },
    )
    second = resolve_regime_state(
        symbol="ETHUSDT",
        timeframe="1h",
        detected=_detected(MarketRegime.UPTREND),
        snapshot_id="snapshot-3",
        snapshot_version=3,
        data_hash="hash-3",
        previous_state=first.to_payload(),
    )

    assert first.effective_regime is MarketRegime.RANGE
    assert first.pending_regime is MarketRegime.UPTREND
    assert first.pending_confirmation_count == 1
    assert first.reasons == ("regime_transition_pending_confirmation",)
    assert second.effective_regime is MarketRegime.UPTREND
    assert second.pending_regime is None
    assert second.transition_accepted is True
    assert second.reasons == ("regime_transition_confirmed",)


def test_phase_14_repeated_snapshot_preserves_existing_regime_state() -> None:
    previous = {
        "effective_regime": "range",
        "pending_regime": "uptrend",
        "pending_confirmation_count": 1,
        "snapshot_id": "snapshot-2",
    }

    repeated = resolve_regime_state(
        symbol="ETHUSDT",
        timeframe="1h",
        detected=_detected(MarketRegime.UPTREND),
        snapshot_id="snapshot-2",
        snapshot_version=2,
        data_hash="hash-2",
        previous_state=previous,
    )

    assert repeated.effective_regime is MarketRegime.RANGE
    assert repeated.pending_regime is MarketRegime.UPTREND
    assert repeated.pending_confirmation_count == 1
    assert repeated.transition_accepted is False
    assert repeated.reasons == ("repeated_snapshot_state_preserved",)


def test_phase_14_stale_or_malformed_state_does_not_block_new_snapshot() -> None:
    resolved = resolve_regime_state(
        symbol="ETHUSDT",
        timeframe="1h",
        detected=_detected(MarketRegime.DOWNTREND),
        snapshot_id="snapshot-9",
        snapshot_version=9,
        data_hash="hash-9",
        previous_state={"effective_regime": "not-a-regime", "snapshot_id": "snapshot-1"},
    )

    assert resolved.effective_regime is MarketRegime.DOWNTREND
    assert resolved.transition_accepted is True
    assert resolved.reasons == ("initial_regime_state",)


def test_phase_14_adapter_state_changes_contain_regime_state_per_symbol_timeframe() -> None:
    adapter = SpotGridAdapter(
        cycle_service=SpotGridTradingCycleService(
            snapshot_adapter=PlatformSnapshotIndicatorAdapter(required_history=20),
            indicator_runtime=INDICATOR_RUNTIME,
        )
    )

    result = asyncio.run(adapter.run_once(_request(candles=_downtrend_market_candles(), snapshot_id="snapshot-1")))

    assert result.status is BotRunStatus.COMPLETE
    state_by_key = {change.state_key: change for change in result.state_changes}
    key = regime_state_key(symbol="ETHUSDT", timeframe="1h")
    assert key in state_by_key
    assert state_by_key[key].namespace == "spot_grid"
    assert state_by_key[key].value is not None
    assert state_by_key[key].value["effective_regime"] == "downtrend"
    assert state_by_key[key].value["snapshot_id"] == "snapshot-1"


def test_phase_14_adapter_reads_previous_regime_state_and_applies_hysteresis() -> None:
    state_store = _StateStore(
        {
            regime_state_key(symbol="ETHUSDT", timeframe="1h"): {
                "effective_regime": "range",
                "pending_regime": None,
                "pending_confirmation_count": 0,
                "snapshot_id": "snapshot-1",
            }
        }
    )
    adapter = SpotGridAdapter(
        cycle_service=SpotGridTradingCycleService(
            snapshot_adapter=PlatformSnapshotIndicatorAdapter(required_history=20),
            indicator_runtime=INDICATOR_RUNTIME,
        )
    )
    asyncio.run(adapter.initialize(_context(state_store)))

    result = asyncio.run(adapter.run_once(_request(candles=_uptrend_market_candles(), snapshot_id="snapshot-2")))

    assert result.status is BotRunStatus.COMPLETE
    assert result.diagnostics["regime_decision"]["regime"] == "uptrend"
    assert result.diagnostics["regime_state"]["effective_regime"] == "range"
    assert result.diagnostics["regime_state"]["pending_regime"] == "uptrend"
    assert result.diagnostics["regime_state"]["pending_confirmation_count"] == 1
    assert state_store.loads == [
        ("instance-1", "spot_grid", "ETHUSDT:1h:regime_state"),
        ("instance-1", "spot_grid", "ETHUSDT:1h:cooldown_state"),
        ("instance-1", "spot_grid", "ETHUSDT:1h:last_emitted_intent_hash"),
    ]


def test_phase_14_state_machine_source_is_platform_native_and_boundary_safe() -> None:
    source = (SPOT_GRID_ROOT / "domain" / "regime_state_machine.py").read_text(encoding="utf-8")
    forbidden_terms = (
        "spot_grid_bot",
        "sqlalchemy",
        "asyncpg",
        "candle_1h",
        "candle_4h",
        "candle_1d",
        "float",
        "requests",
        "pybit",
    )

    assert [term for term in forbidden_terms if term in source] == []


def _detected(regime: MarketRegime) -> RegimeSnapshot:
    return RegimeSnapshot(
        regime=regime,
        confidence=Decimal("0.75"),
        reasons=(f"{regime.value}_fixture",),
        diagnostics={"fixture": True},
    )


def _request(*, candles: tuple[BotCandle, ...], snapshot_id: str) -> BotRunRequest:
    return BotRunRequest(
        run_id=f"run-{snapshot_id}",
        instance_id="instance-1",
        module_id="spot_grid",
        mode=BotMode.SIGNAL_ONLY,
        trigger_type=BotTriggerType.MANUAL,
        config={"max_grid_levels": 1},
        market_data=BotMarketDataContext(primary_snapshot=_snapshot(candles=candles, snapshot_id=snapshot_id)),
    )


def _snapshot(*, candles: tuple[BotCandle, ...], snapshot_id: str) -> BotMarketSnapshot:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return BotMarketSnapshot(
        snapshot_id=snapshot_id,
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        provider_symbol="ETHUSDT",
        timeframe="1h",
        last_closed_candle_time=now,
        lookback_start_time=now,
        lookback_end_time=now,
        completeness_status="COMPLETE",
        snapshot_version=int(snapshot_id.rsplit("-", 1)[-1]),
        data_hash=f"hash-{snapshot_id}",
        candles=candles,
    )


def _downtrend_market_candles() -> tuple[BotCandle, ...]:
    return tuple(_market_candle(index=index, close=Decimal(130 - index)) for index in range(30))


def _uptrend_market_candles() -> tuple[BotCandle, ...]:
    return tuple(_market_candle(index=index, close=Decimal(100 + index)) for index in range(30))


def _market_candle(*, index: int, close: Decimal) -> BotCandle:
    open_time = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=index)
    return BotCandle(
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        timeframe="1h",
        open_time=open_time,
        close_time=open_time + timedelta(hours=1),
        open=close,
        high=close + Decimal("2"),
        low=close - Decimal("2"),
        close=close,
        volume=Decimal("1000"),
    )


def _context(state_store: "_StateStore") -> BotRuntimeContext:
    return BotRuntimeContext(
        market_data=_Noop(),
        signal_publisher=_Noop(),
        state_store=state_store,
        notification_publisher=_Noop(),
        secret_provider=_Noop(),
        logger=_Noop(),
        metrics=_Noop(),
        clock=_Noop(),
    )


class _StateStore:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values
        self.loads: list[tuple[str, str, str]] = []

    async def load(self, *, instance_id: str, namespace: str, state_key: str) -> object | None:
        self.loads.append((instance_id, namespace, state_key))
        return self.values.get(state_key)

    async def save(self, *, instance_id: str, namespace: str, state_key: str, value: object) -> None:
        self.values[state_key] = value


class _Noop:
    def __getattr__(self, name):
        def _method(*args, **kwargs):
            return None

        return _method
