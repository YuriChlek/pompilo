from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from bot_platform_service.domain import (
    BotCandle,
    BotMarketDataContext,
    BotMarketSnapshot,
    BotMode,
    BotRunRequest,
    BotRunStatus,
    BotTriggerType,
)
from bot_platform_service.trading_bots.spot_grid.adapter import SpotGridAdapter
from bot_platform_service.trading_bots.spot_grid.application import SpotGridTradingCycleService
from bot_platform_service.trading_bots.spot_grid.domain import (
    GridLevelSide,
    IndicatorSnapshot,
    MarketRegime,
    PositionContext,
    SpotGridCandle,
    SpotGridConfig,
    SpotGridPlanner,
    TargetIntentType,
)
from tests.fixtures.spot_grid_indicator_runtime import FakeStockIndicatorsRuntime


def test_phase_21_price_distance_blocks_levels_too_close_to_reference_price() -> None:
    plan = SpotGridPlanner().plan(
        symbol="ETHUSDT",
        timeframe="1h",
        candles=(_domain_candle("100", "101", "99", "100"),),
        config=SpotGridConfig(
            symbols=("ETHUSDT",),
            primary_timeframe="1h",
            supporting_timeframes=(),
            max_position_fraction=Decimal("0.10"),
            max_grid_levels=2,
            min_price_distance_fraction=Decimal("0.02"),
        ),
        regime=MarketRegime.RANGE,
        indicators=_indicators(rsi14=Decimal("70"), realized_volatility=Decimal("0.01")),
        position_context=PositionContext(symbol="ETHUSDT", cost_basis=Decimal("90")),
    )

    assert plan.intents == ()
    assert plan.diagnostics["price_distance"]["min_price_distance_fraction"] == "0.02"


def test_phase_21_high_volatility_pause_blocks_new_entry_intents() -> None:
    service = SpotGridTradingCycleService(indicator_runtime=_FixedVolatilityRuntime(realized_volatility=Decimal("0.08")))

    result = service.run_once(
        market_data=BotMarketDataContext(primary_snapshot=_snapshot(close_start=Decimal("120"), close_step=Decimal("-1"))),
        config=SpotGridConfig(
            symbols=("ETHUSDT",),
            primary_timeframe="1h",
            supporting_timeframes=(),
            max_position_fraction=Decimal("0.10"),
            max_grid_levels=2,
            high_volatility_pause_threshold=Decimal("0.05"),
            volatility_cooldown_runs=3,
        ),
    )

    assert all(intent.intent_type is not TargetIntentType.OPEN_POSITION for intent in result.plan.intents)
    assert "high_volatility_entry_pause" in result.plan.diagnostics["entry_block_reasons"]
    assert result.diagnostics["volatility_pause"]["active"] is True
    assert result.diagnostics["cooldown_state"]["remaining_runs"] == 3


def test_phase_21_cooldown_state_is_saved_through_bot_state_change() -> None:
    adapter = SpotGridAdapter(
        cycle_service=SpotGridTradingCycleService(
            indicator_runtime=_FixedVolatilityRuntime(realized_volatility=Decimal("0.01"))
        )
    )

    result = asyncio.run(
        adapter.run_once(
            _request(
                config={
                    "supporting_timeframes": [],
                    "max_grid_levels": 2,
                    "high_volatility_pause_threshold": "0.05",
                    "volatility_cooldown_runs": 2,
                }
            )
        )
    )

    assert result.status is BotRunStatus.COMPLETE
    state_by_key = {change.state_key: change for change in result.state_changes}
    cooldown_change = state_by_key["ETHUSDT:1h:cooldown_state"]
    assert cooldown_change.namespace == "spot_grid"
    assert cooldown_change.value is not None
    assert cooldown_change.value["active"] is False
    assert cooldown_change.value["remaining_runs"] == 0


def test_phase_21_previous_cooldown_blocks_entries_and_decrements_state() -> None:
    service = SpotGridTradingCycleService(indicator_runtime=_FixedVolatilityRuntime(realized_volatility=Decimal("0.01")))

    result = service.run_once(
        market_data=BotMarketDataContext(primary_snapshot=_snapshot(close_start=Decimal("120"), close_step=Decimal("-1"))),
        config=SpotGridConfig(
            symbols=("ETHUSDT",),
            primary_timeframe="1h",
            supporting_timeframes=(),
            max_position_fraction=Decimal("0.10"),
            max_grid_levels=2,
            high_volatility_pause_threshold=Decimal("0.05"),
            volatility_cooldown_runs=2,
        ),
        previous_cooldown_state={"remaining_runs": 2},
    )

    assert all(intent.side is not GridLevelSide.BUY for intent in result.plan.intents)
    assert "volatility_cooldown_entry_pause" in result.plan.diagnostics["entry_block_reasons"]
    assert result.diagnostics["cooldown_state"]["remaining_runs"] == 1


class _FixedVolatilityRuntime(FakeStockIndicatorsRuntime):
    def __init__(self, *, realized_volatility: Decimal) -> None:
        self.realized_volatility = realized_volatility

    def rsi_last(self, quotes, length: int):
        del quotes, length
        return Decimal("30")

    def realized_volatility_last(self, quotes, length: int):
        del quotes, length
        return self.realized_volatility


def _request(*, config: dict[str, object]) -> BotRunRequest:
    return BotRunRequest(
        run_id="run-phase-21",
        instance_id="instance-1",
        module_id="spot_grid",
        mode=BotMode.SIGNAL_ONLY,
        trigger_type=BotTriggerType.MANUAL,
        config=config,
        market_data=BotMarketDataContext(primary_snapshot=_snapshot(close_start=Decimal("100"), close_step=Decimal("0"))),
    )


def _snapshot(*, close_start: Decimal, close_step: Decimal) -> BotMarketSnapshot:
    now = datetime(2026, 7, 15, tzinfo=UTC)
    candles = tuple(
        _market_candle(index=index, close=close_start + (Decimal(index) * close_step))
        for index in range(30)
    )
    return BotMarketSnapshot(
        snapshot_id="snapshot-phase-21",
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        provider_symbol="ETHUSDT",
        timeframe="1h",
        last_closed_candle_time=now,
        lookback_start_time=now,
        lookback_end_time=now,
        completeness_status="COMPLETE",
        snapshot_version=1,
        data_hash="hash-phase-21",
        candles=candles,
    )


def _market_candle(*, index: int, close: Decimal) -> BotCandle:
    open_time = datetime(2026, 7, 15, tzinfo=UTC) + timedelta(hours=index)
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


def _indicators(*, rsi14: Decimal, realized_volatility: Decimal) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        ema20=Decimal("100"),
        ema50=Decimal("100"),
        ema200=Decimal("100"),
        atr14=Decimal("2"),
        rsi14=rsi14,
        realized_volatility=realized_volatility,
        realized_volatility_short=realized_volatility,
        current_volume=Decimal("1000"),
        volume_ma20=Decimal("1000"),
        volume_ratio=Decimal("1"),
        candle_count=30,
        has_required_history=True,
        volatility_has_required_history=True,
        volume_has_required_history=True,
    )


def _domain_candle(open_price: str, high: str, low: str, close: str) -> SpotGridCandle:
    return SpotGridCandle(
        timestamp="2026-07-15T00:00:00+00:00",
        open=Decimal(open_price),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal("1000"),
    )
