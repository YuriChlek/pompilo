from __future__ import annotations

from dataclasses import fields
from decimal import Decimal
from pathlib import Path

import pytest

from bot_platform_service.trading_bots.spot_grid.domain import (
    GridLevelSide,
    MarketRegime,
    PortfolioContext,
    PositionContext,
    PriceBand,
    StrategyDecision,
    StrategyGuardSnapshot,
    StrategyRiskLimits,
    TargetExecutionIntent,
    TargetIntent,
    TargetIntentType,
    normalize_spot_grid_symbol,
)


SERVICE_ROOT = Path(__file__).resolve().parents[2]
SPOT_GRID_DOMAIN_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "trading_bots" / "spot_grid" / "domain"


def test_phase_6_strategy_models_normalize_symbols_to_platform_format() -> None:
    assert normalize_spot_grid_symbol("btc/usdt") == "BTCUSDT"
    assert normalize_spot_grid_symbol(" eth-usdt ") == "ETHUSDT"
    assert PositionContext(symbol="sol_usdt").symbol == "SOLUSDT"
    assert _target_intent(symbol="ada/usdt").symbol == "ADAUSDT"

    with pytest.raises(ValueError, match="symbol must not be empty"):
        normalize_spot_grid_symbol(" / - _ ")


def test_phase_6_target_intent_serializes_to_json_safe_payload() -> None:
    intent = _target_intent()

    payload = intent.to_payload()

    assert payload == {
        "intent_type": "open_position",
        "execution_intent": "limit_entry_candidate",
        "strategy": "spot_grid",
        "regime": "range",
        "symbol": "ETHUSDT",
        "timeframe": "1h",
        "reference_price": "102.50",
        "target_price": "100.25",
        "grid_level_index": 1,
        "side": "buy",
        "price_band": {
            "range_low": "95",
            "range_high": "110",
        },
        "risk": {
            "max_position_fraction": "0.10",
            "suggested_quote_notional": "25",
            "max_quote_notional": "50",
        },
        "guards": {
            "rsi14": "31.5",
            "atr14": "2.25",
            "buy_allowed": True,
            "sell_allowed": False,
            "no_loss_required": False,
            "no_loss_passed": None,
        },
        "reason_codes": ("range_buy", "rsi_oversold"),
    }


def test_phase_6_strategy_decision_serializes_without_execution_specific_fields() -> None:
    decision = StrategyDecision(
        symbol="eth/usdt",
        timeframe="1h",
        regime=MarketRegime.RANGE,
        intents=(_target_intent(),),
        diagnostics={"planner": "fixture"},
    )

    payload = decision.to_payload()

    assert payload["symbol"] == "ETHUSDT"
    assert payload["regime"] == "range"
    assert payload["diagnostics"] == {"planner": "fixture"}
    serialized = repr(payload)
    forbidden_terms = (
        "client_order_id",
        "order_id",
        "exchange_order_id",
        "venue_order_id",
        "fill_id",
        "api_key",
        "api_secret",
        "private_key",
    )
    assert [term for term in forbidden_terms if term in serialized] == []


def test_phase_6_portfolio_and_position_context_are_strategy_safe() -> None:
    context = PortfolioContext(
        total_equity=Decimal("1000"),
        available_quote=Decimal("250"),
        positions=(
            PositionContext(
                symbol="eth/usdt",
                base_quantity=Decimal("0.5"),
                quote_notional=Decimal("100"),
                cost_basis=Decimal("1800"),
                min_no_loss_exit_price=Decimal("1810"),
            ),
        ),
    )

    assert context.to_payload() == {
        "total_equity": "1000",
        "available_quote": "250",
        "positions": [
            {
                "symbol": "ETHUSDT",
                "base_quantity": "0.5",
                "quote_notional": "100",
                "cost_basis": "1800",
                "min_no_loss_exit_price": "1810",
            }
        ],
    }


def test_phase_6_strategy_models_do_not_expose_order_or_exchange_fields() -> None:
    forbidden_field_names = {
        "client_order_id",
        "order_id",
        "exchange_order_id",
        "venue_order_id",
        "fill_id",
        "filled_size",
        "api_key",
        "api_secret",
        "private_key",
        "secret_ref",
    }
    model_types = (
        PositionContext,
        PortfolioContext,
        StrategyDecision,
        StrategyGuardSnapshot,
        StrategyRiskLimits,
        TargetIntent,
    )

    violations = {
        model_type.__name__: sorted(forbidden_field_names & {field.name for field in fields(model_type)})
        for model_type in model_types
    }

    assert {name: names for name, names in violations.items() if names} == {}


def test_phase_6_domain_models_have_no_forbidden_boundary_imports() -> None:
    forbidden_terms = (
        "spot_grid_bot",
        "asyncpg",
        "sqlalchemy",
        "redis",
        "requests",
        "httpx",
        "pybit",
        "Bybit",
        "Binance",
    )
    violations: list[str] = []
    for path in sorted(SPOT_GRID_DOMAIN_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        found = [term for term in forbidden_terms if term in text]
        if found:
            violations.append(f"{path.relative_to(SERVICE_ROOT)}: {', '.join(found)}")

    assert violations == []


def _target_intent(*, symbol: str = "eth/usdt") -> TargetIntent:
    return TargetIntent(
        intent_type=TargetIntentType.OPEN_POSITION,
        execution_intent=TargetExecutionIntent.LIMIT_ENTRY_CANDIDATE,
        symbol=symbol,
        timeframe="1h",
        regime=MarketRegime.RANGE,
        side=GridLevelSide.BUY,
        target_price=Decimal("100.25"),
        reference_price=Decimal("102.50"),
        price_band=PriceBand(range_low=Decimal("95"), range_high=Decimal("110")),
        risk=StrategyRiskLimits(
            max_position_fraction=Decimal("0.10"),
            suggested_quote_notional=Decimal("25"),
            max_quote_notional=Decimal("50"),
        ),
        guards=StrategyGuardSnapshot(
            buy_allowed=True,
            sell_allowed=False,
            rsi14=Decimal("31.5"),
            atr14=Decimal("2.25"),
        ),
        reason_codes=("range_buy", "rsi_oversold"),
        grid_level_index=1,
    )
