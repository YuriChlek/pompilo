from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from bot_platform_service.trading_bots.spot_grid.domain import (
    GridLevelSide,
    IndicatorSnapshot,
    MarketRegime,
    PortfolioContext,
    PositionContext,
    SpotGridCandle,
    SpotGridConfig,
    SpotGridPlanner,
)


SERVICE_ROOT = Path(__file__).resolve().parents[2]
RECOVERY_SOURCE = (
    SERVICE_ROOT
    / "src"
    / "bot_platform_service"
    / "trading_bots"
    / "spot_grid"
    / "domain"
    / "recovery.py"
)


def test_phase_26_recovery_averaging_buy_intent_has_reasons_and_budget_fields() -> None:
    plan = _plan(
        regime=MarketRegime.RANGE,
        indicators=_indicators(rsi14=Decimal("30")),
        portfolio_context=_portfolio_context(
            total_equity=Decimal("1000"),
            available_quote=Decimal("300"),
            quote_notional=Decimal("120"),
            cost_basis=Decimal("120"),
        ),
    )

    recovery_intent = _single_recovery_intent(plan.intents)
    payload = recovery_intent.to_payload()

    assert recovery_intent.side is GridLevelSide.BUY
    assert recovery_intent.target_price == Decimal("100.00000000")
    assert payload["risk"]["suggested_quote_notional"] == "80.00"
    assert payload["risk"]["max_quote_notional"] == "80.00"
    assert payload["reason_codes"] == (
        "recovery_averaging_buy",
        "budget_available",
        "position_underwater",
        "recovery_eligible",
    )
    assert payload["position"] == {
        "symbol": "ETHUSDT",
        "base_quantity": "1",
        "quote_notional": "120",
        "cost_basis": "120",
        "min_no_loss_exit_price": None,
    }
    assert plan.diagnostics["recovery_averaging"]["intent_count"] == 1


@pytest.mark.parametrize(
    "regime",
    (MarketRegime.DOWNTREND, MarketRegime.HIGH_VOLATILITY, MarketRegime.RISK_OFF),
)
def test_phase_26_recovery_averaging_is_blocked_in_forbidden_regimes(regime: MarketRegime) -> None:
    plan = _plan(
        regime=regime,
        indicators=_indicators(rsi14=Decimal("30")),
        portfolio_context=_portfolio_context(
            total_equity=Decimal("1000"),
            available_quote=Decimal("300"),
            quote_notional=Decimal("120"),
            cost_basis=Decimal("120"),
        ),
    )

    assert _recovery_intents(plan.intents) == ()
    assert plan.diagnostics["recovery_averaging"]["blocked_regime"] is True
    assert "recovery_averaging_regime_block" in plan.diagnostics["recovery_averaging"]["reason_codes"]


def test_phase_26_recovery_averaging_requires_buy_guardrails_and_budget() -> None:
    rsi_blocked = _plan(
        regime=MarketRegime.RANGE,
        indicators=_indicators(rsi14=Decimal("55")),
        portfolio_context=_portfolio_context(
            total_equity=Decimal("1000"),
            available_quote=Decimal("300"),
            quote_notional=Decimal("120"),
            cost_basis=Decimal("120"),
        ),
    )
    budget_blocked = _plan(
        regime=MarketRegime.RANGE,
        indicators=_indicators(rsi14=Decimal("30")),
        portfolio_context=_portfolio_context(
            total_equity=Decimal("1000"),
            available_quote=Decimal("300"),
            quote_notional=Decimal("200"),
            cost_basis=Decimal("120"),
        ),
    )

    assert _recovery_intents(rsi_blocked.intents) == ()
    assert "recovery_averaging_rsi_buy_block" in rsi_blocked.diagnostics["recovery_averaging"]["reason_codes"]
    assert _recovery_intents(budget_blocked.intents) == ()
    assert budget_blocked.diagnostics["recovery_averaging"]["budget_available"] is False
    assert "recovery_averaging_budget_block" in budget_blocked.diagnostics["recovery_averaging"]["reason_codes"]


def test_phase_26_recovery_averaging_intent_payload_has_no_private_execution_details() -> None:
    plan = _plan(
        regime=MarketRegime.RANGE,
        indicators=_indicators(rsi14=Decimal("30")),
        portfolio_context=_portfolio_context(
            total_equity=Decimal("1000"),
            available_quote=Decimal("300"),
            quote_notional=Decimal("120"),
            cost_basis=Decimal("120"),
        ),
    )

    payload = _single_recovery_intent(plan.intents).to_payload()
    private_terms = (
        "client_order_id",
        "order_id",
        "exchange_order_id",
        "venue_order_id",
        "fill_id",
        "fill_state",
        "filled_size",
        "api_key",
        "api_secret",
        "private_key",
    )

    assert _private_payload_terms(payload, private_terms) == []


def test_phase_26_recovery_source_has_no_exchange_or_persistence_reads() -> None:
    text = RECOVERY_SOURCE.read_text(encoding="utf-8")
    forbidden_terms = (
        "asyncpg",
        "sqlalchemy",
        "requests",
        "httpx",
        "ccxt",
        "pybit",
        "fetch_balance",
        "get_wallet_balance",
        "get_positions",
        "place_order",
        "create_order",
    )

    assert [term for term in forbidden_terms if term in text] == []


def _plan(
    *,
    regime: MarketRegime,
    indicators: IndicatorSnapshot,
    portfolio_context: PortfolioContext,
):
    return SpotGridPlanner().plan(
        symbol="ETHUSDT",
        timeframe="1h",
        candles=(_candle("100", "110", "90", "100"),),
        config=SpotGridConfig(
            symbols=("ETHUSDT",),
            primary_timeframe="1h",
            supporting_timeframes=(),
            max_position_fraction=Decimal("0.20"),
            max_grid_levels=1,
        ),
        regime=regime,
        indicators=indicators,
        portfolio_context=portfolio_context,
    )


def _recovery_intents(intents):
    return tuple(intent for intent in intents if "recovery_averaging_buy" in intent.reason_codes)


def _single_recovery_intent(intents):
    recovery_intents = _recovery_intents(intents)
    assert len(recovery_intents) == 1
    return recovery_intents[0]


def _portfolio_context(
    *,
    total_equity: Decimal,
    available_quote: Decimal,
    quote_notional: Decimal,
    cost_basis: Decimal,
) -> PortfolioContext:
    return PortfolioContext(
        total_equity=total_equity,
        available_quote=available_quote,
        positions=(
            PositionContext(
                symbol="ETHUSDT",
                base_quantity=Decimal("1"),
                quote_notional=quote_notional,
                cost_basis=cost_basis,
            ),
        ),
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


def _private_payload_terms(value, private_terms: tuple[str, ...]) -> list[str]:
    if isinstance(value, dict):
        matches = [key for key in value if key in private_terms]
        for item in value.values():
            matches.extend(_private_payload_terms(item, private_terms))
        return matches
    if isinstance(value, list | tuple):
        matches: list[str] = []
        for item in value:
            matches.extend(_private_payload_terms(item, private_terms))
        return matches
    return []
