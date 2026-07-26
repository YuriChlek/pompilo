from __future__ import annotations

from dataclasses import fields
from decimal import Decimal

import pytest

from bot_platform_service.trading_bots.spot_grid.application import parse_spot_grid_config
from bot_platform_service.trading_bots.spot_grid.domain import (
    GridLevel,
    GridLevelSide,
    MarketRegime,
    PortfolioContext,
    PositionContext,
    PriceBand,
    SpotGridCandle,
    SpotGridConfig,
    SpotGridPlan,
    StrategyDecision,
    StrategyGuardSnapshot,
    StrategyRiskLimits,
    TargetExecutionIntent,
    TargetIntent,
    TargetIntentType,
)


def test_phase_7_critical_strategy_dto_annotations_use_decimal_not_float() -> None:
    expected_decimal_fields = {
        SpotGridCandle: {"open", "high", "low", "close", "volume"},
        SpotGridConfig: {"max_position_fraction"},
        GridLevel: {"price"},
        PriceBand: {"range_low", "range_high"},
        StrategyRiskLimits: {"max_position_fraction", "suggested_quote_notional", "max_quote_notional"},
        StrategyGuardSnapshot: {"rsi14", "atr14"},
        PositionContext: {"base_quantity", "quote_notional", "cost_basis", "min_no_loss_exit_price"},
        PortfolioContext: {"total_equity", "available_quote"},
        TargetIntent: {"target_price", "reference_price"},
        SpotGridPlan: {"reference_price", "range_low", "range_high"},
    }

    for model_type, field_names in expected_decimal_fields.items():
        annotations = model_type.__annotations__
        for field_name in field_names:
            assert "float" not in str(annotations[field_name])
            assert "Decimal" in str(annotations[field_name])


def test_phase_7_strategy_dtos_reject_float_trading_values() -> None:
    cases = (
        (SpotGridCandle, {"timestamp": "t", "open": 1.0, "high": Decimal("1"), "low": Decimal("1"), "close": Decimal("1"), "volume": Decimal("1")}),
        (SpotGridConfig, {"symbols": ("ETHUSDT",), "primary_timeframe": "1h", "supporting_timeframes": (), "max_position_fraction": 0.1, "max_grid_levels": 1}),
        (GridLevel, {"side": GridLevelSide.BUY, "price": 1.0, "level_index": 0, "reason": "range_buy"}),
        (PriceBand, {"range_low": 1.0, "range_high": Decimal("2")}),
        (StrategyRiskLimits, {"max_position_fraction": 0.1}),
        (StrategyGuardSnapshot, {"buy_allowed": True, "sell_allowed": False, "rsi14": 31.5}),
        (PositionContext, {"symbol": "ETHUSDT", "base_quantity": 0.1}),
        (PortfolioContext, {"total_equity": 1000.0, "available_quote": Decimal("100")}),
        (
            TargetIntent,
            {
                **_target_intent_kwargs(),
                "target_price": 100.25,
            },
        ),
        (
            SpotGridPlan,
            {
                "symbol": "ETHUSDT",
                "timeframe": "1h",
                "reference_price": 102.5,
                "range_low": Decimal("95"),
                "range_high": Decimal("110"),
                "levels": (),
                "diagnostics": {},
            },
        ),
    )

    for model_type, kwargs in cases:
        with pytest.raises(TypeError, match="must be Decimal"):
            model_type(**kwargs)


def test_phase_7_target_intent_payload_serializes_decimals_as_strings() -> None:
    intent = TargetIntent(
        **_target_intent_kwargs(
            metadata={
                "edge": Decimal("0.000000000000000001"),
                "nested": {"ratio": Decimal("0.123456789123456789")},
            }
        )
    )

    payload = intent.to_payload()

    assert payload["reference_price"] == "102.500000000000000001"
    assert payload["target_price"] == "100.250000000000000001"
    assert payload["price_band"] == {
        "range_low": "95.000000000000000001",
        "range_high": "110.000000000000000001",
    }
    assert payload["risk"] == {
        "max_position_fraction": "0.10",
        "suggested_quote_notional": "25.000000000000000001",
        "max_quote_notional": "50.000000000000000001",
    }
    assert payload["guards"]["rsi14"] == "31.500000000000000001"
    assert payload["guards"]["atr14"] == "2.250000000000000001"
    assert payload["metadata"] == {
        "edge": "0.000000000000000001",
        "nested": {"ratio": "0.123456789123456789"},
    }


def test_phase_7_strategy_decision_diagnostics_convert_decimal_and_reject_float() -> None:
    decision = StrategyDecision(
        symbol="ETHUSDT",
        timeframe="1h",
        regime=MarketRegime.RANGE,
        intents=(TargetIntent(**_target_intent_kwargs()),),
        diagnostics={"score": Decimal("0.987654321987654321")},
    )

    assert decision.to_payload()["diagnostics"] == {"score": "0.987654321987654321"}

    with pytest.raises(TypeError, match="float values are not allowed"):
        StrategyDecision(
            symbol="ETHUSDT",
            timeframe="1h",
            regime=MarketRegime.RANGE,
            intents=(),
            diagnostics={"score": 0.98},
        )


def test_phase_7_persisted_config_rejects_float_decimal_overrides() -> None:
    with pytest.raises(TypeError, match="max_position_fraction must not be float"):
        parse_spot_grid_config(
            {"max_position_fraction": 0.1},
            fallback_symbols=("ETHUSDT",),
            fallback_timeframes=("1h",),
        )


def test_phase_7_strategy_model_field_names_do_not_reintroduce_float_aliases() -> None:
    model_types = (
        SpotGridCandle,
        SpotGridConfig,
        GridLevel,
        PriceBand,
        StrategyRiskLimits,
        StrategyGuardSnapshot,
        PositionContext,
        PortfolioContext,
        TargetIntent,
        SpotGridPlan,
    )

    violations = {
        model_type.__name__: sorted(field.name for field in fields(model_type) if field.name.endswith("_float"))
        for model_type in model_types
    }

    assert {name: names for name, names in violations.items() if names} == {}


def _target_intent_kwargs(*, metadata: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "intent_type": TargetIntentType.OPEN_POSITION,
        "execution_intent": TargetExecutionIntent.LIMIT_ENTRY_CANDIDATE,
        "symbol": "ETHUSDT",
        "timeframe": "1h",
        "regime": MarketRegime.RANGE,
        "side": GridLevelSide.BUY,
        "target_price": Decimal("100.250000000000000001"),
        "reference_price": Decimal("102.500000000000000001"),
        "price_band": PriceBand(
            range_low=Decimal("95.000000000000000001"),
            range_high=Decimal("110.000000000000000001"),
        ),
        "risk": StrategyRiskLimits(
            max_position_fraction=Decimal("0.10"),
            suggested_quote_notional=Decimal("25.000000000000000001"),
            max_quote_notional=Decimal("50.000000000000000001"),
        ),
        "guards": StrategyGuardSnapshot(
            buy_allowed=True,
            sell_allowed=False,
            rsi14=Decimal("31.500000000000000001"),
            atr14=Decimal("2.250000000000000001"),
        ),
        "reason_codes": ("range_buy",),
        "grid_level_index": 1,
        "metadata": metadata or {},
    }
