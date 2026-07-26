from __future__ import annotations

from decimal import Decimal

from bot_platform_service.trading_bots.spot_grid.domain import (
    GridLevelSide,
    IndicatorSnapshot,
    MarketRegime,
    PositionContext,
    PortfolioContext,
    SpotGridCandle,
    SpotGridConfig,
    SpotGridPlanner,
    TargetIntentType,
)


def test_phase_20_no_loss_policy_blocks_sell_below_threshold() -> None:
    plan = _range_plan(
        high="110",
        cost_basis=Decimal("109"),
        min_no_loss_exit_price=Decimal("109.50"),
    )

    assert plan.intents == ()
    assert plan.diagnostics["no_loss"]["min_exit_price"] == "109.50000000"


def test_phase_20_no_loss_policy_blocks_sell_when_cost_basis_is_unknown() -> None:
    plan = _range_plan(
        high="110",
        cost_basis=None,
        min_no_loss_exit_price=None,
    )

    assert plan.intents == ()
    assert plan.diagnostics["no_loss"]["passed"] is False
    assert plan.diagnostics["no_loss"]["block_reason"] == "cost_basis_unknown"


def test_phase_20_no_loss_policy_creates_sell_only_above_threshold() -> None:
    plan = _range_plan(
        high="112",
        cost_basis=Decimal("100"),
        min_no_loss_exit_price=None,
    )

    assert plan.intents
    assert {intent.intent_type for intent in plan.intents} == {TargetIntentType.CLOSE_POSITION}
    assert {intent.side for intent in plan.intents} == {GridLevelSide.SELL}
    assert all(intent.target_price is not None and intent.target_price >= Decimal("100.10000000") for intent in plan.intents)
    assert all("no_loss_passed" in intent.reason_codes for intent in plan.intents)
    assert all(intent.position is not None for intent in plan.intents)
    assert {intent.position.min_no_loss_exit_price for intent in plan.intents if intent.position is not None} == {
        Decimal("100.10000000")
    }
    assert plan.diagnostics["no_loss"]["cost_basis"] == "100"
    assert plan.diagnostics["no_loss"]["min_exit_price"] == "100.10000000"
    assert plan.diagnostics["no_loss"]["block_reason"] is None


def test_phase_20_no_loss_policy_blocks_sell_without_position_context() -> None:
    plan = SpotGridPlanner().plan(
        symbol="ETHUSDT",
        timeframe="1h",
        candles=(_candle("100", "110", "90", "100"),),
        config=_config(max_grid_levels=2),
        regime=MarketRegime.RANGE,
        indicators=_indicators(rsi14=Decimal("70")),
    )

    assert plan.intents == ()
    assert plan.diagnostics["no_loss"]["block_reason"] == "position_context_missing"


def test_phase_20_no_loss_policy_uses_cost_basis_from_strategy_context() -> None:
    plan = SpotGridPlanner().plan(
        symbol="ETHUSDT",
        timeframe="1h",
        candles=(_candle("100", "112", "90", "100"),),
        config=_config(max_grid_levels=2),
        regime=MarketRegime.RANGE,
        indicators=_indicators(rsi14=Decimal("70")),
        portfolio_context=PortfolioContext(
            total_equity=Decimal("1000"),
            available_quote=Decimal("900"),
            positions=(
                PositionContext(
                    symbol="ETHUSDT",
                    base_quantity=Decimal("1"),
                    quote_notional=Decimal("100"),
                    cost_basis=Decimal("100"),
                ),
            ),
        ),
    )

    assert plan.intents
    assert plan.diagnostics["no_loss"]["cost_basis"] == "100"
    assert plan.diagnostics["no_loss"]["min_exit_price"] == "100.10000000"


def _range_plan(
    *,
    high: str,
    cost_basis: Decimal | None,
    min_no_loss_exit_price: Decimal | None,
):
    return SpotGridPlanner().plan(
        symbol="ETHUSDT",
        timeframe="1h",
        candles=(_candle("100", high, "90", "100"),),
        config=_config(max_grid_levels=2),
        regime=MarketRegime.RANGE,
        indicators=_indicators(rsi14=Decimal("70")),
        position_context=PositionContext(
            symbol="ETHUSDT",
            base_quantity=Decimal("1"),
            quote_notional=Decimal("100"),
            cost_basis=cost_basis,
            min_no_loss_exit_price=min_no_loss_exit_price,
        ),
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
