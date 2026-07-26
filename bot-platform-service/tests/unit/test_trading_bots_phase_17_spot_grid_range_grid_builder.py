from __future__ import annotations

from decimal import Decimal

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


def test_phase_17_range_grid_builder_creates_expected_buy_and_sell_intents() -> None:
    buy_plan = SpotGridPlanner().plan(
        symbol="ethusdt",
        timeframe="1h",
        candles=(
            _candle("100", "106", "94", "100"),
            _candle("100", "110", "90", "100"),
        ),
        config=_config(max_grid_levels=2),
        regime=MarketRegime.RANGE,
        indicators=_indicators(rsi14=Decimal("30")),
    )
    sell_plan = SpotGridPlanner().plan(
        symbol="ethusdt",
        timeframe="1h",
        candles=(
            _candle("100", "106", "94", "100"),
            _candle("100", "110", "90", "100"),
        ),
        config=_config(max_grid_levels=2),
        regime=MarketRegime.RANGE,
        indicators=_indicators(rsi14=Decimal("70")),
        position_context=_position(),
    )

    assert [intent.intent_type for intent in buy_plan.intents] == [
        TargetIntentType.OPEN_POSITION,
        TargetIntentType.OPEN_POSITION,
    ]
    assert [intent.intent_type for intent in sell_plan.intents] == [
        TargetIntentType.CLOSE_POSITION,
        TargetIntentType.CLOSE_POSITION,
    ]
    assert [intent.side for intent in buy_plan.intents] == [
        GridLevelSide.BUY,
        GridLevelSide.BUY,
    ]
    assert [intent.side for intent in sell_plan.intents] == [
        GridLevelSide.SELL,
        GridLevelSide.SELL,
    ]
    assert [intent.target_price for intent in buy_plan.intents] == [
        Decimal("96.66666667"),
        Decimal("93.33333333"),
    ]
    assert [intent.target_price for intent in sell_plan.intents] == [
        Decimal("103.33333333"),
        Decimal("106.66666667"),
    ]
    assert [level.price for level in buy_plan.levels] == [intent.target_price for intent in buy_plan.intents]
    assert [level.price for level in sell_plan.levels] == [intent.target_price for intent in sell_plan.intents]
    assert buy_plan.diagnostics["planner"] == "spot_grid_range_intent_grid"
    assert buy_plan.diagnostics["intent_count"] == 2


def test_phase_17_range_grid_builder_adds_reason_codes_to_every_intent() -> None:
    plan = SpotGridPlanner().plan(
        symbol="ETHUSDT",
        timeframe="1h",
        candles=(_candle("100", "110", "90", "100"),),
        config=_config(max_grid_levels=1),
        regime=MarketRegime.RANGE,
        indicators=_indicators(rsi14=Decimal("30")),
    )

    assert plan.intents
    assert all(intent.reason_codes for intent in plan.intents)
    assert {intent.reason_codes[0] for intent in plan.intents} == {"range_buy"}


def test_phase_17_range_grid_builder_keeps_prices_inside_range_band() -> None:
    plan = SpotGridPlanner().plan(
        symbol="ETHUSDT",
        timeframe="1h",
        candles=(_candle("100", "115", "85", "100"),),
        config=_config(max_grid_levels=4),
        regime=MarketRegime.RANGE,
        indicators=_indicators(rsi14=Decimal("30")),
    )

    for intent in plan.intents:
        assert intent.target_price is not None
        assert plan.range_low < intent.target_price < plan.range_high
        if intent.side is GridLevelSide.BUY:
            assert intent.target_price < plan.reference_price
        if intent.side is GridLevelSide.SELL:
            assert intent.target_price > plan.reference_price


def test_phase_17_range_grid_builder_blocks_buy_intents_when_entries_are_not_allowed() -> None:
    plan = SpotGridPlanner().plan(
        symbol="ETHUSDT",
        timeframe="1h",
        candles=(_candle("100", "110", "90", "100"),),
        config=_config(max_grid_levels=2),
        regime=MarketRegime.DOWNTREND,
        indicators=_indicators(rsi14=Decimal("70")),
        position_context=_position(),
    )

    assert plan.intents
    assert {intent.intent_type for intent in plan.intents} == {TargetIntentType.CLOSE_POSITION}
    assert {intent.side for intent in plan.intents} == {GridLevelSide.SELL}


def _config(*, max_grid_levels: int) -> SpotGridConfig:
    return SpotGridConfig(
        symbols=("ETHUSDT",),
        primary_timeframe="1h",
        supporting_timeframes=(),
        max_position_fraction=Decimal("0.10"),
        max_grid_levels=max_grid_levels,
    )


def _indicators(*, rsi14: Decimal) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        ema20=Decimal("100"),
        ema50=Decimal("100"),
        ema200=Decimal("100"),
        atr14=Decimal("2"),
        rsi14=rsi14,
        realized_volatility=Decimal("0.01"),
        realized_volatility_short=Decimal("0.01"),
        current_volume=Decimal("1000"),
        volume_ma20=Decimal("1000"),
        volume_ratio=Decimal("1"),
        candle_count=30,
        has_required_history=True,
        volatility_has_required_history=True,
        volume_has_required_history=True,
    )


def _position() -> PositionContext:
    return PositionContext(symbol="ETHUSDT", base_quantity=Decimal("1"), quote_notional=Decimal("100"), cost_basis=Decimal("100"))


def _candle(open_price: str, high: str, low: str, close: str) -> SpotGridCandle:
    return SpotGridCandle(
        timestamp="2026-07-15T00:00:00+00:00",
        open=Decimal(open_price),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal("1000"),
    )
