from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

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
    GridLevelSide,
    MarketRegime,
    PortfolioContext,
    PositionContext,
    PriceBand,
    StrategyGuardSnapshot,
    StrategyRiskLimits,
    TargetExecutionIntent,
    TargetIntent,
    TargetIntentType,
    build_semantic_intent_hash,
)


def test_phase_31_new_snapshot_with_same_semantic_intent_is_blocked_by_state_gate() -> None:
    state_store = _StateStore()
    adapter = _adapter(target_price=Decimal("110"))
    asyncio.run(adapter.initialize(_context(state_store)))

    first = asyncio.run(adapter.run_once(_request(run_id="run-1", snapshot_id="snapshot-1")))
    _persist_state_changes(state_store, first.state_changes)
    second = asyncio.run(adapter.run_once(_request(run_id="run-2", snapshot_id="snapshot-2")))

    assert first.status is BotRunStatus.COMPLETE
    assert second.status is BotRunStatus.COMPLETE
    assert len(first.signals) == 1
    assert second.signals == ()
    assert first.signals[0].signal_key != _signal_key_for_snapshot(first, "snapshot-2")
    dedupe = second.diagnostics["semantic_intent_idempotency"]
    assert dedupe["blocked_count"] == 1
    assert dedupe["emitted_count"] == 0
    assert state_store.loads[-1] == ("instance-1", "spot_grid", "ETHUSDT:1h:last_emitted_intent_hash")


def test_phase_31_intent_hash_is_independent_from_signal_schema_version_and_snapshot() -> None:
    intent = _intent(target_price=Decimal("110"))

    first_hash = build_semantic_intent_hash(intent)
    second_hash = build_semantic_intent_hash(intent)

    assert first_hash == second_hash
    assert len(first_hash) == 64
    assert "payload_schema" not in intent.to_payload()
    assert "snapshot_id" not in intent.to_payload()


def test_phase_31_stale_state_does_not_block_materially_different_intent() -> None:
    state_store = _StateStore(
        {
            "ETHUSDT:1h:last_emitted_intent_hash": {
                "intent_hashes": [build_semantic_intent_hash(_intent(target_price=Decimal("110")))],
                "snapshot_id": "old-snapshot",
            }
        }
    )
    adapter = _adapter(target_price=Decimal("112"))
    asyncio.run(adapter.initialize(_context(state_store)))

    result = asyncio.run(adapter.run_once(_request(run_id="run-3", snapshot_id="snapshot-3")))

    assert result.status is BotRunStatus.COMPLETE
    assert len(result.signals) == 1
    dedupe = result.diagnostics["semantic_intent_idempotency"]
    assert dedupe["blocked_count"] == 0
    assert dedupe["emitted_count"] == 1


def _adapter(*, target_price: Decimal) -> SpotGridAdapter:
    return SpotGridAdapter(cycle_service=SpotGridTradingCycleService(planner=_Planner(target_price=target_price)))


def _request(*, run_id: str, snapshot_id: str) -> BotRunRequest:
    return BotRunRequest(
        run_id=run_id,
        instance_id="instance-1",
        module_id="spot_grid",
        mode=BotMode.SIGNAL_ONLY,
        trigger_type=BotTriggerType.EVENT,
        config={"supporting_timeframes": [], "max_grid_levels": 1},
        market_data=BotMarketDataContext(primary_snapshot=_snapshot(snapshot_id=snapshot_id)),
    )


def _snapshot(*, snapshot_id: str) -> BotMarketSnapshot:
    now = datetime(2026, 7, 25, tzinfo=UTC)
    candle = BotCandle(
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        timeframe="1h",
        open_time=now,
        close_time=now,
        open=Decimal("100"),
        high=Decimal("112"),
        low=Decimal("98"),
        close=Decimal("105"),
        volume=Decimal("1000"),
    )
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
        snapshot_version=1,
        data_hash=f"hash-{snapshot_id}",
        candles=(candle,),
    )


def _intent(*, target_price: Decimal) -> TargetIntent:
    return TargetIntent(
        intent_type=TargetIntentType.CLOSE_POSITION,
        execution_intent=TargetExecutionIntent.LIMIT_EXIT_CANDIDATE,
        symbol="ETHUSDT",
        timeframe="1h",
        regime=MarketRegime.RANGE,
        side=GridLevelSide.SELL,
        target_price=target_price,
        reference_price=Decimal("105"),
        price_band=PriceBand(range_low=Decimal("98"), range_high=Decimal("112")),
        risk=StrategyRiskLimits(max_position_fraction=Decimal("0.10")),
        guards=StrategyGuardSnapshot(
            buy_allowed=False,
            sell_allowed=True,
            no_loss_required=True,
            no_loss_passed=True,
        ),
        reason_codes=("range_take_profit",),
        grid_level_index=1,
        position=PositionContext(
            symbol="ETHUSDT",
            base_quantity=Decimal("1"),
            quote_notional=Decimal("100"),
            cost_basis=Decimal("100"),
        ),
    )


class _Planner:
    def __init__(self, *, target_price: Decimal) -> None:
        self.target_price = target_price

    def plan(self, **kwargs):
        from bot_platform_service.trading_bots.spot_grid.domain import SpotGridPlan

        del kwargs
        intent = _intent(target_price=self.target_price)
        return SpotGridPlan(
            symbol="ETHUSDT",
            timeframe="1h",
            reference_price=Decimal("105"),
            range_low=Decimal("98"),
            range_high=Decimal("112"),
            levels=(),
            diagnostics={"planner": "phase_31_fixture", "intent_count": 1},
            regime=MarketRegime.RANGE,
            intents=(intent,),
        )


class _StateStore:
    def __init__(self, values: dict[str, object] | None = None) -> None:
        self.values = dict(values or {})
        self.loads: list[tuple[str, str, str]] = []

    async def load(self, *, instance_id: str, namespace: str, state_key: str) -> object | None:
        self.loads.append((instance_id, namespace, state_key))
        return self.values.get(state_key)

    async def save(self, *, instance_id: str, namespace: str, state_key: str, value: object) -> None:
        del instance_id, namespace
        self.values[state_key] = value


class _Noop:
    def __getattr__(self, name):
        def _method(*args, **kwargs):
            return None

        return _method


def _context(state_store: _StateStore) -> BotRuntimeContext:
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


def _persist_state_changes(state_store: _StateStore, changes) -> None:
    for change in changes:
        assert change.value is not None
        state_store.values[change.state_key] = change.value


def _signal_key_for_snapshot(result, snapshot_id: str) -> str:
    signal = result.signals[0]
    from bot_platform_service.domain import BotSignal

    rebuilt = BotSignal.build(
        instance_id=signal.instance_id,
        module_id=signal.module_id,
        symbol=signal.symbol,
        timeframe=signal.timeframe,
        snapshot_id=snapshot_id,
        signal_type=signal.signal_type,
        side=signal.side,
        confidence=signal.confidence,
        reason=signal.reason,
        payload_schema=signal.payload_schema,
        payload_schema_version=signal.payload_schema_version,
        payload=signal.payload,
    )
    return rebuilt.signal_key
