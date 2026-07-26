from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from bot_platform_service.trading_bots.spot_grid.domain import (
    GridLevelSide,
    IndicatorSnapshot,
    MarketRegime,
    PortfolioContext,
    PositionContext,
    SpotGridCandle,
    SpotGridConfig,
    SpotGridPlanner,
    allocate_portfolio_budget,
    target_intent_to_bot_signal,
)


SERVICE_ROOT = Path(__file__).resolve().parents[2]
ALLOCATOR_SOURCE = (
    SERVICE_ROOT
    / "src"
    / "bot_platform_service"
    / "trading_bots"
    / "spot_grid"
    / "domain"
    / "portfolio_allocator.py"
)


def test_phase_24_allocator_calculates_per_symbol_and_portfolio_budget_caps() -> None:
    allocation = allocate_portfolio_budget(
        portfolio_context=_portfolio_context(),
        symbol="ETHUSDT",
        max_position_fraction=Decimal("0.20"),
        candidate_count=2,
    )

    assert allocation.to_payload() == {
        "symbol": "ETHUSDT",
        "max_position_fraction": "0.20",
        "symbol_current_quote_notional": "100",
        "symbol_max_quote_notional": "200.00",
        "symbol_remaining_quote_notional": "100.00",
        "portfolio_available_quote": "500",
        "portfolio_remaining_quote_notional": "500",
        "max_quote_notional": "100.00",
        "suggested_quote_notional": "50.00",
        "candidate_count": 2,
    }


def test_phase_24_max_position_fraction_changes_suggested_notional() -> None:
    small = allocate_portfolio_budget(
        portfolio_context=_portfolio_context(),
        symbol="ETHUSDT",
        max_position_fraction=Decimal("0.15"),
        candidate_count=2,
    )
    large = allocate_portfolio_budget(
        portfolio_context=_portfolio_context(),
        symbol="ETHUSDT",
        max_position_fraction=Decimal("0.30"),
        candidate_count=2,
    )

    assert small.suggested_quote_notional == Decimal("25.00")
    assert large.suggested_quote_notional == Decimal("100.00")
    assert small.suggested_quote_notional < large.suggested_quote_notional


def test_phase_24_entry_intent_payload_contains_suggested_and_max_quote_notional() -> None:
    plan = SpotGridPlanner().plan(
        symbol="ETHUSDT",
        timeframe="1h",
        candles=(_candle("100", "110", "90", "100"),),
        config=_config(max_position_fraction=Decimal("0.20")),
        regime=MarketRegime.RANGE,
        indicators=_indicators(rsi14=Decimal("30")),
        portfolio_context=_portfolio_context(),
    )

    buy_intents = tuple(intent for intent in plan.intents if intent.side is GridLevelSide.BUY)

    assert buy_intents
    assert {intent.risk.suggested_quote_notional for intent in buy_intents} == {Decimal("50.00")}
    assert {intent.risk.max_quote_notional for intent in buy_intents} == {Decimal("100.00")}
    assert plan.diagnostics["allocation"]["suggested_quote_notional"] == "50.00"
    assert plan.diagnostics["allocation"]["max_quote_notional"] == "100.00"

    signal = target_intent_to_bot_signal(
        buy_intents[0],
        instance_id="instance-1",
        module_id="spot_grid",
        snapshot_id="snapshot-phase-24",
        confidence=Decimal("0.60"),
    )
    assert signal.payload["risk"]["suggested_quote_notional"] == "50.00"
    assert signal.payload["risk"]["max_quote_notional"] == "100.00"
    assert "required_quote_notional" not in signal.payload["risk"]


def test_phase_24_allocator_source_has_no_exchange_or_order_terms() -> None:
    text = ALLOCATOR_SOURCE.read_text(encoding="utf-8")
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


def _portfolio_context() -> PortfolioContext:
    return PortfolioContext(
        total_equity=Decimal("1000"),
        available_quote=Decimal("500"),
        positions=(
            PositionContext(
                symbol="ETHUSDT",
                base_quantity=Decimal("1"),
                quote_notional=Decimal("100"),
                cost_basis=Decimal("100"),
            ),
            PositionContext(
                symbol="BTCUSDT",
                base_quantity=Decimal("0.01"),
                quote_notional=Decimal("300"),
                cost_basis=Decimal("30000"),
            ),
        ),
    )


def _config(*, max_position_fraction: Decimal) -> SpotGridConfig:
    return SpotGridConfig(
        symbols=("ETHUSDT",),
        primary_timeframe="1h",
        supporting_timeframes=(),
        max_position_fraction=max_position_fraction,
        max_grid_levels=2,
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
