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


def test_phase_19_rsi_buy_rule_allows_entry_only_at_or_below_threshold() -> None:
    allowed = _range_plan(rsi14=Decimal("35"))
    blocked = _range_plan(rsi14=Decimal("35.1"))

    assert {intent.intent_type for intent in allowed.intents} == {TargetIntentType.OPEN_POSITION}
    assert {intent.side for intent in allowed.intents} == {GridLevelSide.BUY}
    assert all("rsi_buy_passed" in intent.reason_codes for intent in allowed.intents)
    assert blocked.intents == ()
    assert blocked.diagnostics["rsi_buy_allowed"] is False
    assert "rsi_buy_block" in blocked.diagnostics["rsi_block_reasons"]


def test_phase_19_rsi_sell_rule_allows_ordinary_exit_only_above_threshold() -> None:
    blocked = _range_plan(rsi14=Decimal("65"))
    allowed = _range_plan(rsi14=Decimal("65.1"))

    assert blocked.intents == ()
    assert blocked.diagnostics["rsi_sell_allowed"] is False
    assert "rsi_sell_block" in blocked.diagnostics["rsi_block_reasons"]
    assert {intent.intent_type for intent in allowed.intents} == {TargetIntentType.CLOSE_POSITION}
    assert {intent.side for intent in allowed.intents} == {GridLevelSide.SELL}
    assert all("rsi_sell_passed" in intent.reason_codes for intent in allowed.intents)


def test_phase_19_blocked_rsi_decisions_are_visible_in_diagnostics() -> None:
    plan = _range_plan(rsi14=Decimal("50"))

    assert plan.intents == ()
    assert plan.diagnostics["rsi14"] == "50"
    assert plan.diagnostics["rsi_buy_allowed"] is False
    assert plan.diagnostics["rsi_sell_allowed"] is False
    assert plan.diagnostics["rsi_block_reasons"] == ("rsi_buy_block", "rsi_sell_block")
    assert plan.diagnostics["intent_count"] == 0


def test_phase_19_de_risk_does_not_bypass_rsi_sell_rule_yet() -> None:
    plan = SpotGridPlanner().plan(
        symbol="ETHUSDT",
        timeframe="1h",
        candles=(_candle("100", "110", "90", "100"),),
        config=_config(max_grid_levels=2),
        regime=MarketRegime.RISK_OFF,
        indicators=_indicators(rsi14=Decimal("30")),
    )

    assert plan.intents == ()
    assert "rsi_sell_block" in plan.diagnostics["rsi_block_reasons"]


def _range_plan(*, rsi14: Decimal):
    return SpotGridPlanner().plan(
        symbol="ETHUSDT",
        timeframe="1h",
        candles=(_candle("100", "110", "90", "100"),),
        config=_config(max_grid_levels=2),
        regime=MarketRegime.RANGE,
        indicators=_indicators(rsi14=rsi14),
        position_context=PositionContext(symbol="ETHUSDT", base_quantity=Decimal("1"), quote_notional=Decimal("100"), cost_basis=Decimal("100")),
    )


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


def _candle(open_price: str, high: str, low: str, close: str) -> SpotGridCandle:
    return SpotGridCandle(
        timestamp="2026-07-15T00:00:00+00:00",
        open=Decimal(open_price),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal("1000"),
    )
