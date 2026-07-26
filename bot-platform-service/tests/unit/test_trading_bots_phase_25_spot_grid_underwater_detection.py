from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from bot_platform_service.domain import BotCandle, BotMarketDataContext, BotMarketSnapshot
from bot_platform_service.trading_bots.spot_grid.application import SpotGridTradingCycleService
from bot_platform_service.trading_bots.spot_grid.domain import (
    MarketRegime,
    PortfolioContext,
    PositionContext,
    SpotGridConfig,
    detect_underwater_state,
)
from tests.fixtures.spot_grid_indicator_runtime import FakeStockIndicatorsRuntime


SERVICE_ROOT = Path(__file__).resolve().parents[2]
UNDERWATER_SOURCE = (
    SERVICE_ROOT
    / "src"
    / "bot_platform_service"
    / "trading_bots"
    / "spot_grid"
    / "domain"
    / "underwater.py"
)


def test_phase_25_underwater_state_is_deterministic_for_fixture_position() -> None:
    state = detect_underwater_state(
        portfolio_context=_portfolio_context(cost_basis=Decimal("120")),
        symbol="ETHUSDT",
        reference_price=Decimal("100"),
        regime=MarketRegime.RANGE,
        entry_block_reasons=(),
    )

    assert state.to_payload() == {
        "reference_price": "100",
        "regime": "range",
        "underwater_count": 1,
        "recovery_eligible_count": 1,
        "reason_codes": ("underwater_positions_detected", "recovery_eligible"),
        "positions": [
            {
                "symbol": "ETHUSDT",
                "base_quantity": "1",
                "quote_notional": "120",
                "cost_basis": "120",
                "reference_price": "100",
                "unrealized_pnl_fraction": "-0.1666666666666666666666666666666667",
                "underwater": True,
                "recovery_eligible": True,
                "reason_codes": ("position_underwater", "recovery_eligible"),
            }
        ],
    }


def test_phase_25_recovery_eligibility_is_blocked_by_guardrails_without_creating_intents() -> None:
    state = detect_underwater_state(
        portfolio_context=_portfolio_context(cost_basis=Decimal("120")),
        symbol="ETHUSDT",
        reference_price=Decimal("100"),
        regime=MarketRegime.RISK_OFF,
        entry_block_reasons=("high_volatility_entry_pause",),
    )

    assert state.underwater_count == 1
    assert state.recovery_eligible_count == 0
    assert state.reason_codes == ("underwater_positions_detected", "recovery_not_eligible")
    assert state.positions[0].reason_codes == (
        "position_underwater",
        "recovery_regime_block",
        "recovery_entry_guard_block",
        "recovery_not_eligible",
    )


def test_phase_25_trading_cycle_diagnostics_explain_underwater_decision() -> None:
    service = SpotGridTradingCycleService(indicator_runtime=FakeStockIndicatorsRuntime())

    result = service.run_once(
        market_data=BotMarketDataContext(primary_snapshot=_snapshot(close=Decimal("100"))),
        config=_config(),
        portfolio_context=_portfolio_context(cost_basis=Decimal("120")),
    )

    assert result.plan.intents == ()
    assert result.diagnostics["underwater"]["underwater_count"] == 1
    assert result.diagnostics["underwater"]["recovery_eligible_count"] == 1
    assert result.diagnostics["underwater"]["positions"][0]["reason_codes"] == (
        "position_underwater",
        "recovery_eligible",
    )


def test_phase_25_missing_cost_basis_explains_unknown_underwater_decision() -> None:
    state = detect_underwater_state(
        portfolio_context=_portfolio_context(cost_basis=None),
        symbol="ETHUSDT",
        reference_price=Decimal("100"),
        regime=MarketRegime.RANGE,
        entry_block_reasons=(),
    )

    assert state.underwater_count == 0
    assert state.recovery_eligible_count == 0
    assert state.positions[0].reason_codes == (
        "cost_basis_unknown",
        "position_not_underwater",
        "recovery_not_eligible",
    )


def test_phase_25_underwater_source_has_no_exchange_or_persistence_reads() -> None:
    text = UNDERWATER_SOURCE.read_text(encoding="utf-8")
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


def _portfolio_context(*, cost_basis: Decimal | None) -> PortfolioContext:
    return PortfolioContext(
        total_equity=Decimal("1000"),
        available_quote=Decimal("300"),
        positions=(
            PositionContext(
                symbol="ETHUSDT",
                base_quantity=Decimal("1"),
                quote_notional=Decimal("120"),
                cost_basis=cost_basis,
            ),
        ),
    )


def _config() -> SpotGridConfig:
    return SpotGridConfig(
        symbols=("ETHUSDT",),
        primary_timeframe="1h",
        supporting_timeframes=(),
        max_position_fraction=Decimal("0.10"),
        max_grid_levels=2,
    )


def _snapshot(*, close: Decimal) -> BotMarketSnapshot:
    now = datetime(2026, 7, 15, tzinfo=UTC)
    candle = BotCandle(
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        timeframe="1h",
        open_time=now,
        close_time=now,
        open=close,
        high=close + Decimal("2"),
        low=close - Decimal("2"),
        close=close,
        volume=Decimal("1000"),
    )
    return BotMarketSnapshot(
        snapshot_id="snapshot-phase-25",
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        provider_symbol="ETHUSDT",
        timeframe="1h",
        last_closed_candle_time=now,
        lookback_start_time=now,
        lookback_end_time=now,
        completeness_status="COMPLETE",
        snapshot_version=1,
        data_hash="hash-phase-25",
        candles=(candle,),
    )
