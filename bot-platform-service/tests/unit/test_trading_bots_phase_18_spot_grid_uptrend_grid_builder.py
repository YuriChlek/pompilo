from __future__ import annotations

from decimal import Decimal

from bot_platform_service.trading_bots.spot_grid.domain import (
    GridLevelSide,
    IndicatorSnapshot,
    MarketRegime,
    MarketStructureBias,
    MarketStructureSnapshot,
    SpotGridCandle,
    SpotGridConfig,
    SpotGridPlanner,
    SupportResistanceCandidate,
    TargetIntentType,
)


def test_phase_18_uptrend_grid_builder_creates_buy_intents_only_on_pullbacks() -> None:
    plan = SpotGridPlanner().plan(
        symbol="ETHUSDT",
        timeframe="1h",
        candles=(_candle("108", "112", "106", "110"),),
        config=_config(max_grid_levels=3),
        regime=MarketRegime.UPTREND,
        indicators=_indicators(atr14=Decimal("5")),
        market_structure=_structure(),
    )

    assert plan.diagnostics["planner"] == "spot_grid_uptrend_pullback_grid"
    assert plan.intents
    assert {intent.intent_type for intent in plan.intents} == {TargetIntentType.OPEN_POSITION}
    assert {intent.side for intent in plan.intents} == {GridLevelSide.BUY}
    assert all(intent.target_price is not None and intent.target_price < plan.reference_price for intent in plan.intents)
    assert {level.side for level in plan.levels} == {GridLevelSide.BUY}


def test_phase_18_uptrend_grid_builder_uses_atr_step_for_spacing() -> None:
    low_atr = SpotGridPlanner().plan(
        symbol="ETHUSDT",
        timeframe="1h",
        candles=(_candle("108", "112", "106", "110"),),
        config=_config(max_grid_levels=3),
        regime=MarketRegime.UPTREND,
        indicators=_indicators(atr14=Decimal("2")),
        market_structure=_structure(supports=()),
    )
    high_atr = SpotGridPlanner().plan(
        symbol="ETHUSDT",
        timeframe="1h",
        candles=(_candle("108", "112", "106", "110"),),
        config=_config(max_grid_levels=3),
        regime=MarketRegime.UPTREND,
        indicators=_indicators(atr14=Decimal("5")),
        market_structure=_structure(supports=()),
    )

    assert [intent.target_price for intent in low_atr.intents] == [
        Decimal("108.00000000"),
        Decimal("106.00000000"),
        Decimal("104.00000000"),
    ]
    assert [intent.target_price for intent in high_atr.intents] == [
        Decimal("105.00000000"),
        Decimal("100.00000000"),
        Decimal("95.00000000"),
    ]
    assert low_atr.diagnostics["atr14"] == "2"
    assert high_atr.diagnostics["atr14"] == "5"


def test_phase_18_uptrend_grid_builder_aligns_pullbacks_to_local_support() -> None:
    plan = SpotGridPlanner().plan(
        symbol="ETHUSDT",
        timeframe="1h",
        candles=(_candle("108", "112", "106", "110"),),
        config=_config(max_grid_levels=2),
        regime=MarketRegime.UPTREND,
        indicators=_indicators(atr14=Decimal("5")),
        market_structure=_structure(
            supports=(Decimal("104"), Decimal("99")),
            resistances=(Decimal("118"),),
        ),
    )

    assert [intent.target_price for intent in plan.intents] == [Decimal("104.00000000"), Decimal("99.00000000")]
    assert all("support_aligned" in intent.reason_codes for intent in plan.intents)
    assert {intent.price_band.range_high for intent in plan.intents} == {Decimal("118.00000000")}


def test_phase_18_uptrend_grid_builder_blocks_entries_when_guardrails_block_entries() -> None:
    plan = SpotGridPlanner().plan(
        symbol="ETHUSDT",
        timeframe="1h",
        candles=(_candle("108", "112", "106", "110"),),
        config=_config(max_grid_levels=2),
        regime=MarketRegime.UPTREND,
        entry_block_reasons=("supporting_4h_downtrend_block",),
        indicators=_indicators(atr14=Decimal("5")),
        market_structure=_structure(),
    )

    assert plan.intents == ()
    assert plan.levels == ()
    assert plan.diagnostics["entry_allowed"] is False
    assert plan.diagnostics["entry_block_reasons"] == ("supporting_4h_downtrend_block",)


def _config(*, max_grid_levels: int) -> SpotGridConfig:
    return SpotGridConfig(
        symbols=("ETHUSDT",),
        primary_timeframe="1h",
        supporting_timeframes=(),
        max_position_fraction=Decimal("0.10"),
        max_grid_levels=max_grid_levels,
    )


def _indicators(*, atr14: Decimal) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        ema20=Decimal("120"),
        ema50=Decimal("110"),
        ema200=Decimal("100"),
        atr14=atr14,
        rsi14=Decimal("30"),
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


def _structure(
    *,
    supports: tuple[Decimal, ...] = (Decimal("104"), Decimal("100"), Decimal("94")),
    resistances: tuple[Decimal, ...] = (Decimal("118"),),
) -> MarketStructureSnapshot:
    return MarketStructureSnapshot(
        bias=MarketStructureBias.BULLISH,
        candle_count=30,
        has_required_history=True,
        range_low=Decimal("90"),
        range_high=Decimal("120"),
        range_position=Decimal("0.66"),
        swing_highs=(),
        swing_lows=(),
        support_candidates=tuple(
            SupportResistanceCandidate(
                price=price,
                source="fixture_support",
                candle_index=index,
                distance_from_close=Decimal("110") - price,
            )
            for index, price in enumerate(supports)
        ),
        resistance_candidates=tuple(
            SupportResistanceCandidate(
                price=price,
                source="fixture_resistance",
                candle_index=index,
                distance_from_close=price - Decimal("110"),
            )
            for index, price in enumerate(resistances)
        ),
        breakout_direction="up",
        breakout_reference_price=Decimal("112"),
        reasons=("fixture_bullish_structure",),
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
