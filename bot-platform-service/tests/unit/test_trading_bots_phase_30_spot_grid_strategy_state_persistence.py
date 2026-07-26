from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Mapping

from bot_platform_service.application import StateChangeApplierService
from bot_platform_service.domain import (
    BotCandle,
    BotMarketDataContext,
    BotMarketSnapshot,
    BotMode,
    BotRunRequest,
    BotRunResult,
    BotRunStatus,
    BotStateChange,
    BotStateChangeOperation,
    BotTriggerType,
    build_payload_hash,
)
from bot_platform_service.domain.models import BotRuntimeStateRecord
from bot_platform_service.runtime.container import PersistentRuntimeStateStore
from bot_platform_service.trading_bots.spot_grid.adapter import SpotGridAdapter
from bot_platform_service.trading_bots.spot_grid.application import SpotGridTradingCycleService
from tests.fixtures.spot_grid_indicator_runtime import FakeStockIndicatorsRuntime


def test_phase_30_spot_grid_returns_persistable_strategy_state_changes() -> None:
    result = asyncio.run(_adapter().run_once(_request(symbol="eth/usdt")))

    assert result.status is BotRunStatus.COMPLETE
    state_by_key = {change.state_key: change for change in result.state_changes}
    assert set(state_by_key) == {
        "ETHUSDT:1h:last_plan",
        "ETHUSDT:1h:regime_state",
        "ETHUSDT:1h:cooldown_state",
    }
    assert all(change.namespace == "spot_grid" for change in state_by_key.values())
    assert all(change.operation is BotStateChangeOperation.UPSERT for change in state_by_key.values())

    last_plan = _state_value(state_by_key["ETHUSDT:1h:last_plan"])
    assert last_plan["snapshot_id"] == "snapshot-phase-30"
    assert last_plan["data_hash"] == "hash-phase-30"
    assert last_plan["symbol"] == "ETHUSDT"
    assert last_plan["timeframe"] == "1h"
    assert last_plan["mode"] == "signal_only"
    assert isinstance(last_plan["reference_price"], str)

    regime_state = _state_value(state_by_key["ETHUSDT:1h:regime_state"])
    assert regime_state["symbol"] == "ETHUSDT"
    assert regime_state["timeframe"] == "1h"
    assert regime_state["snapshot_id"] == "snapshot-phase-30"
    assert "effective_regime" in regime_state

    cooldown_state = _state_value(state_by_key["ETHUSDT:1h:cooldown_state"])
    assert cooldown_state["snapshot_id"] == "snapshot-phase-30"
    assert cooldown_state["remaining_runs"] == 0
    assert "high_volatility" in cooldown_state


def test_phase_30_state_change_application_is_idempotent_by_deterministic_key() -> None:
    source = asyncio.run(_adapter().run_once(_request(symbol="ETH-USDT")))
    duplicate = _duplicate_last_change(source)
    state_store = _StateStore()

    application = asyncio.run(StateChangeApplierService().apply(duplicate, state_store=state_store))

    assert application.applied == 3
    assert application.skipped == 1
    assert [item[2] for item in state_store.saved] == [
        "ETHUSDT:1h:last_plan",
        "ETHUSDT:1h:regime_state",
        "ETHUSDT:1h:cooldown_state",
    ]


def test_phase_30_runtime_state_store_persists_and_loads_strategy_state() -> None:
    async def run() -> None:
        repository = _RuntimeStateRepository()
        store = PersistentRuntimeStateStore(repository)
        value = {
            "snapshot_id": "snapshot-phase-30",
            "remaining_runs": 2,
            "reference_price": Decimal("100.25"),
            "reason_codes": ("volatility_cooldown_entry_pause",),
        }
        expected_json = {
            "snapshot_id": "snapshot-phase-30",
            "remaining_runs": 2,
            "reference_price": "100.25",
            "reason_codes": ["volatility_cooldown_entry_pause"],
        }

        await store.save(
            instance_id="instance-1",
            namespace="spot_grid",
            state_key="ETHUSDT:1h:cooldown_state",
            value=value,
        )
        loaded = await store.load(
            instance_id="instance-1",
            namespace="spot_grid",
            state_key="ETHUSDT:1h:cooldown_state",
        )

        assert loaded == expected_json
        assert repository.saved[0]["state_id"].startswith("state_")
        assert repository.saved[0]["state_json"] == expected_json
        assert repository.saved[0]["state_hash"] == build_payload_hash(expected_json)
        assert repository.saved[0]["namespace"] == "spot_grid"

    asyncio.run(run())


def _adapter() -> SpotGridAdapter:
    return SpotGridAdapter(cycle_service=SpotGridTradingCycleService(indicator_runtime=FakeStockIndicatorsRuntime()))


def _request(*, symbol: str) -> BotRunRequest:
    return BotRunRequest(
        run_id="run-phase-30",
        instance_id="instance-1",
        module_id="spot_grid",
        mode=BotMode.SIGNAL_ONLY,
        trigger_type=BotTriggerType.MANUAL,
        config={"supporting_timeframes": [], "max_grid_levels": 2},
        market_data=BotMarketDataContext(primary_snapshot=_snapshot(symbol=symbol)),
    )


def _snapshot(*, symbol: str) -> BotMarketSnapshot:
    now = datetime(2026, 7, 25, tzinfo=UTC)
    candles = tuple(
        BotCandle(
            source="binance_spot",
            canonical_symbol=symbol,
            timeframe="1h",
            open_time=now + timedelta(hours=index),
            close_time=now + timedelta(hours=index + 1),
            open=Decimal("100"),
            high=Decimal("102"),
            low=Decimal("98"),
            close=Decimal("100"),
            volume=Decimal("1000"),
        )
        for index in range(30)
    )
    return BotMarketSnapshot(
        snapshot_id="snapshot-phase-30",
        source="binance_spot",
        canonical_symbol=symbol,
        provider_symbol=symbol,
        timeframe="1h",
        last_closed_candle_time=now + timedelta(hours=30),
        lookback_start_time=now,
        lookback_end_time=now + timedelta(hours=30),
        completeness_status="COMPLETE",
        snapshot_version=1,
        data_hash="hash-phase-30",
        candles=candles,
    )


def _state_value(change: BotStateChange) -> Mapping[str, object]:
    assert change.value is not None
    return change.value


def _duplicate_last_change(source: BotRunResult) -> BotRunResult:
    return BotRunResult(
        run_id=source.run_id,
        instance_id=source.instance_id,
        module_id=source.module_id,
        mode=source.mode,
        status=source.status,
        signals=source.signals,
        diagnostics=source.diagnostics,
        state_changes=source.state_changes + (source.state_changes[0],),
    )


class _StateStore:
    def __init__(self) -> None:
        self.saved: list[tuple[str, str, str, object]] = []

    async def load(self, *, instance_id: str, namespace: str, state_key: str) -> object | None:
        del instance_id, namespace, state_key
        return None

    async def save(self, *, instance_id: str, namespace: str, state_key: str, value: object) -> None:
        self.saved.append((instance_id, namespace, state_key, value))


class _RuntimeStateRepository:
    def __init__(self) -> None:
        self.saved: list[dict[str, object]] = []

    async def upsert_runtime_state(self, **kwargs) -> bool:
        self.saved.append(dict(kwargs))
        return True

    async def get_runtime_state(self, *, instance_id: str, namespace: str, state_key: str):
        if not self.saved:
            return None
        state = self.saved[-1]
        return BotRuntimeStateRecord(
            instance_id=instance_id,
            namespace=namespace,
            state_key=state_key,
            state_json=dict(state["state_json"]),
            state_hash=str(state["state_hash"]),
            version=1,
        )
