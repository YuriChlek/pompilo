from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from bot_platform_service.domain import BotCandle, BotMarketSnapshot
from bot_platform_service.trading_bots.spot_grid.application import (
    SpotGridTradingCycleService,
    parse_spot_grid_config,
)
from bot_platform_service.trading_bots.spot_grid.domain import (
    GridLevelSide,
    SpotGridCandle,
    SpotGridConfig,
    SpotGridPlanner,
)


SERVICE_ROOT = Path(__file__).resolve().parents[2]
SPOT_GRID_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "trading_bots" / "spot_grid"


def test_phase_11_spot_grid_domain_planner_builds_decimal_grid_levels() -> None:
    planner = SpotGridPlanner()
    config = SpotGridConfig(
        symbols=("ETHUSDT",),
        primary_timeframe="1h",
        supporting_timeframes=("4h",),
        max_position_fraction=Decimal("0.10"),
        max_grid_levels=2,
    )

    plan = planner.plan(
        symbol="ethusdt",
        timeframe="1h",
        candles=(
            _domain_candle("100", "105", "95", "101"),
            _domain_candle("101", "106", "96", "102"),
        ),
        config=config,
    )

    assert plan.symbol == "ETHUSDT"
    assert plan.reference_price == Decimal("102.00000000")
    assert plan.range_low == Decimal("95.00000000")
    assert plan.range_high == Decimal("106.00000000")
    assert [level.side for level in plan.levels] == [
        GridLevelSide.BUY,
        GridLevelSide.SELL,
        GridLevelSide.BUY,
        GridLevelSide.SELL,
    ]
    assert all(isinstance(level.price, Decimal) for level in plan.levels)
    assert plan.diagnostics["planner"] == "spot_grid_decimal_grid"


def test_phase_11_spot_grid_application_runs_one_cycle_from_platform_snapshot() -> None:
    service = SpotGridTradingCycleService()
    config = parse_spot_grid_config(
        {
            "symbols": ["ETHUSDT"],
            "primary_timeframe": "1h",
            "supporting_timeframes": ["4h"],
            "max_position_fraction": "0.15",
            "max_grid_levels": 3,
        },
        fallback_symbols=("BTCUSDT",),
        fallback_timeframes=("1h", "4h"),
    )

    result = service.run_once(snapshot=_snapshot(), config=config)

    assert result.plan.symbol == "ETHUSDT"
    assert result.plan.timeframe == "1h"
    assert len(result.plan.levels) == 6
    assert result.diagnostics["snapshot_id"] == "snapshot-1"
    assert result.diagnostics["max_position_fraction"] == "0.15"


def test_phase_11_spot_grid_config_parser_uses_platform_fallbacks() -> None:
    config = parse_spot_grid_config(
        {},
        fallback_symbols=("ETHUSDT",),
        fallback_timeframes=("1h", "4h"),
    )

    assert config.symbols == ("ETHUSDT",)
    assert config.primary_timeframe == "1h"
    assert config.supporting_timeframes == ("4h",)
    assert config.max_position_fraction == Decimal("0.10")
    assert config.max_grid_levels == 6


def test_phase_11_spot_grid_source_boundary_has_no_private_exchange_or_sync_imports() -> None:
    forbidden_terms = (
        "spot_grid_bot",
        "asyncpg",
        "sqlalchemy",
        "Bybit",
        "Binance",
        "ensure_candle_tables",
        "MarketDataSynchronizer",
        "scheduler",
        "place_order",
        "cancel_order",
        "create_market_buy_order",
        "create_market_sell_order",
    )
    violations: list[str] = []
    for path in sorted(SPOT_GRID_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        found = [term for term in forbidden_terms if term in text]
        if found:
            violations.append(f"{path.relative_to(SERVICE_ROOT)}: {', '.join(found)}")

    assert violations == []


def _domain_candle(open_price: str, high: str, low: str, close: str) -> SpotGridCandle:
    return SpotGridCandle(
        timestamp="2026-07-15T00:00:00+00:00",
        open=Decimal(open_price),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal("1000"),
    )


def _snapshot() -> BotMarketSnapshot:
    now = datetime(2026, 7, 15, tzinfo=UTC)
    candles = (
        BotCandle(
            source="binance_spot",
            canonical_symbol="ETHUSDT",
            timeframe="1h",
            open_time=now,
            close_time=now,
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("101"),
            volume=Decimal("1000"),
        ),
        BotCandle(
            source="binance_spot",
            canonical_symbol="ETHUSDT",
            timeframe="1h",
            open_time=now,
            close_time=now,
            open=Decimal("101"),
            high=Decimal("106"),
            low=Decimal("96"),
            close=Decimal("102"),
            volume=Decimal("1200"),
        ),
    )
    return BotMarketSnapshot(
        snapshot_id="snapshot-1",
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        provider_symbol="ETHUSDT",
        timeframe="1h",
        last_closed_candle_time=now,
        lookback_start_time=now,
        lookback_end_time=now,
        completeness_status="COMPLETE",
        snapshot_version=1,
        data_hash="hash-1",
        candles=candles,
    )
