from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from bot_platform_service.domain import BotCandle, BotMarketDataContext, BotMarketSnapshot
from bot_platform_service.trading_bots.spot_grid.application import SpotGridTradingCycleService
from bot_platform_service.trading_bots.spot_grid.domain import (
    PortfolioContext,
    PositionContext,
    SpotGridConfig,
    compute_exposure,
)
from tests.fixtures.spot_grid_indicator_runtime import FakeStockIndicatorsRuntime


SERVICE_ROOT = Path(__file__).resolve().parents[2]
EXPOSURE_SOURCE = (
    SERVICE_ROOT
    / "src"
    / "bot_platform_service"
    / "trading_bots"
    / "spot_grid"
    / "domain"
    / "exposure.py"
)


def test_phase_23_exposure_is_deterministic_for_fixture_portfolio() -> None:
    exposure = compute_exposure(_portfolio_context())

    payload = exposure.to_payload()

    assert payload["portfolio"] == {
        "total_equity": "1000",
        "available_quote": "300",
        "gross_quote_notional": "400",
        "gross_exposure_fraction": "0.4",
        "available_quote_fraction": "0.3",
    }
    assert payload["per_symbol"]["BTCUSDT"] == {
        "symbol": "BTCUSDT",
        "position_count": 1,
        "base_quantity": "0.01",
        "quote_notional": "250",
        "exposure_fraction": "0.25",
    }
    assert payload["per_symbol"]["ETHUSDT"] == {
        "symbol": "ETHUSDT",
        "position_count": 2,
        "base_quantity": "1.5",
        "quote_notional": "150",
        "exposure_fraction": "0.15",
    }


def test_phase_23_trading_cycle_diagnostics_include_portfolio_and_per_symbol_exposure() -> None:
    service = SpotGridTradingCycleService(indicator_runtime=FakeStockIndicatorsRuntime())

    result = service.run_once(
        market_data=BotMarketDataContext(primary_snapshot=_snapshot()),
        config=_config(),
        portfolio_context=_portfolio_context(),
    )

    assert result.diagnostics["exposure"]["portfolio"]["gross_quote_notional"] == "400"
    assert result.diagnostics["exposure"]["portfolio"]["gross_exposure_fraction"] == "0.4"
    assert result.diagnostics["exposure"]["per_symbol"]["ETHUSDT"]["quote_notional"] == "150"
    assert result.diagnostics["exposure"]["per_symbol"]["BTCUSDT"]["exposure_fraction"] == "0.25"


def test_phase_23_empty_portfolio_context_has_zero_exposure_without_exchange_reads() -> None:
    service = SpotGridTradingCycleService(indicator_runtime=FakeStockIndicatorsRuntime())

    result = service.run_once(
        market_data=BotMarketDataContext(primary_snapshot=_snapshot()),
        config=_config(),
    )

    assert result.diagnostics["portfolio_context"]["fallback"] == "empty_conservative"
    assert result.diagnostics["exposure"] == {
        "portfolio": {
            "total_equity": "0",
            "available_quote": "0",
            "gross_quote_notional": "0",
            "gross_exposure_fraction": None,
            "available_quote_fraction": None,
        },
        "per_symbol": {},
    }


def test_phase_23_exposure_source_has_no_exchange_or_persistence_reads() -> None:
    text = EXPOSURE_SOURCE.read_text(encoding="utf-8")
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
        available_quote=Decimal("300"),
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
                quote_notional=Decimal("250"),
                cost_basis=Decimal("25000"),
            ),
            PositionContext(
                symbol="ETHUSDT",
                base_quantity=Decimal("0.5"),
                quote_notional=Decimal("50"),
                cost_basis=Decimal("100"),
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


def _snapshot() -> BotMarketSnapshot:
    now = datetime(2026, 7, 15, tzinfo=UTC)
    candle = BotCandle(
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        timeframe="1h",
        open_time=now,
        close_time=now,
        open=Decimal("100"),
        high=Decimal("112"),
        low=Decimal("90"),
        close=Decimal("100"),
        volume=Decimal("1000"),
    )
    return BotMarketSnapshot(
        snapshot_id="snapshot-phase-23",
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        provider_symbol="ETHUSDT",
        timeframe="1h",
        last_closed_candle_time=now,
        lookback_start_time=now,
        lookback_end_time=now,
        completeness_status="COMPLETE",
        snapshot_version=1,
        data_hash="hash-phase-23",
        candles=(candle,),
    )
