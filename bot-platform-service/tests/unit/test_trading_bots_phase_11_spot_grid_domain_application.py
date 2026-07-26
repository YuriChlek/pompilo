from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from bot_platform_service.domain import BotCandle, BotMarketDataContext, BotMarketSnapshot
from bot_platform_service.trading_bots.spot_grid.bot_config import (
    DEFAULT_SPOT_GRID_CONFIG,
    HIGH_VOLATILITY_PAUSE_THRESHOLD_BOUNDS,
    MAX_GRID_LEVELS_BOUNDS,
    MAX_POSITION_FRACTION_BOUNDS,
    MIN_PRICE_DISTANCE_FRACTION_BOUNDS,
    VOLATILITY_COOLDOWN_RUNS_BOUNDS,
)
from bot_platform_service.trading_bots.spot_grid.application import (
    SpotGridTradingCycleService,
    parse_spot_grid_config,
)
from bot_platform_service.trading_bots.spot_grid.config_schema import CONFIG_SCHEMA
from bot_platform_service.trading_bots.spot_grid.domain import (
    GridLevelSide,
    IndicatorSnapshot,
    PositionContext,
    SpotGridCandle,
    SpotGridConfig,
    SpotGridPlanner,
)
from tests.fixtures.spot_grid_indicator_runtime import FakeStockIndicatorsRuntime


SERVICE_ROOT = Path(__file__).resolve().parents[2]
SPOT_GRID_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "trading_bots" / "spot_grid"
INDICATOR_RUNTIME = FakeStockIndicatorsRuntime()


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
        indicators=_indicators(rsi14=Decimal("70")),
        position_context=PositionContext(symbol="ETHUSDT", cost_basis=Decimal("100")),
    )

    assert plan.symbol == "ETHUSDT"
    assert plan.reference_price == Decimal("102.00000000")
    assert plan.range_low == Decimal("95.00000000")
    assert plan.range_high == Decimal("106.00000000")
    assert [level.side for level in plan.levels] == [
        GridLevelSide.SELL,
        GridLevelSide.SELL,
    ]
    assert all(isinstance(level.price, Decimal) for level in plan.levels)
    assert plan.diagnostics["planner"] == "spot_grid_range_intent_grid"


def test_phase_11_spot_grid_application_runs_one_cycle_from_platform_snapshot() -> None:
    service = SpotGridTradingCycleService(indicator_runtime=INDICATOR_RUNTIME)
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

    result = service.run_once(
        market_data=BotMarketDataContext(
            primary_snapshot=_snapshot(),
            supporting_snapshots=(_snapshot(snapshot_id="snapshot-4h", timeframe="4h"),),
        ),
        config=config,
    )

    assert result.plan.symbol == "ETHUSDT"
    assert result.plan.timeframe == "1h"
    assert len(result.plan.levels) == 0
    assert result.diagnostics["no_loss"]["block_reason"] == "position_context_missing"
    assert result.diagnostics["snapshot_id"] == "snapshot-1"
    assert result.diagnostics["primary_timeframe"] == "1h"
    assert result.diagnostics["supporting_timeframes"] == ("4h",)
    assert result.diagnostics["supporting_snapshots"] == (
        {
            "snapshot_id": "snapshot-4h",
            "snapshot_version": 1,
            "timeframe": "4h",
            "data_hash": "hash-snapshot-4h",
        },
    )
    assert result.diagnostics["max_position_fraction"] == "0.15"


def test_phase_11_spot_grid_application_keeps_single_timeframe_context_working() -> None:
    service = SpotGridTradingCycleService(indicator_runtime=INDICATOR_RUNTIME)
    config = parse_spot_grid_config(
        {"max_grid_levels": 1},
        fallback_symbols=("ETHUSDT",),
        fallback_timeframes=("1h",),
    )

    result = service.run_once(
        market_data=BotMarketDataContext(primary_snapshot=_snapshot()),
        config=config,
    )

    assert result.plan.timeframe == "1h"
    assert len(result.plan.levels) == 0
    assert result.diagnostics["no_loss"]["block_reason"] == "position_context_missing"
    assert result.diagnostics["supporting_timeframes"] == ()
    assert result.diagnostics["supporting_snapshots"] == ()


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


def test_phase_11_spot_grid_bot_config_merges_defaults_and_overrides() -> None:
    config = parse_spot_grid_config(
        {
            "max_position_fraction": "0.25",
            "max_grid_levels": 3,
            "emit_diagnostics": False,
        },
        fallback_symbols=("SOLUSDT",),
        fallback_timeframes=("4h", "1h"),
    )

    assert config.symbols == ("SOLUSDT",)
    assert config.primary_timeframe == "4h"
    assert config.supporting_timeframes == ("1h",)
    assert config.max_position_fraction == Decimal("0.25")
    assert config.max_grid_levels == 3
    assert config.emit_diagnostics is False


def test_phase_11_spot_grid_config_schema_mirrors_bot_config_defaults_and_bounds() -> None:
    fields = {
        field["key"]: field
        for section in CONFIG_SCHEMA["sections"]
        for field in section["fields"]
    }

    assert fields["symbols"]["default"] == list(DEFAULT_SPOT_GRID_CONFIG.symbols)
    assert fields["primary_timeframe"]["default"] == DEFAULT_SPOT_GRID_CONFIG.primary_timeframe
    assert fields["supporting_timeframes"]["default"] == list(DEFAULT_SPOT_GRID_CONFIG.supporting_timeframes)
    assert fields["max_position_fraction"]["default"] == str(DEFAULT_SPOT_GRID_CONFIG.max_position_fraction)
    assert fields["max_position_fraction"]["min"] == str(MAX_POSITION_FRACTION_BOUNDS.min)
    assert fields["max_position_fraction"]["max"] == str(MAX_POSITION_FRACTION_BOUNDS.max)
    assert fields["max_grid_levels"]["default"] == DEFAULT_SPOT_GRID_CONFIG.max_grid_levels
    assert fields["max_grid_levels"]["min"] == MAX_GRID_LEVELS_BOUNDS.min
    assert fields["max_grid_levels"]["max"] == MAX_GRID_LEVELS_BOUNDS.max
    assert fields["min_price_distance_fraction"]["default"] == str(DEFAULT_SPOT_GRID_CONFIG.min_price_distance_fraction)
    assert fields["min_price_distance_fraction"]["min"] == str(MIN_PRICE_DISTANCE_FRACTION_BOUNDS.min)
    assert fields["min_price_distance_fraction"]["max"] == str(MIN_PRICE_DISTANCE_FRACTION_BOUNDS.max)
    assert fields["high_volatility_pause_threshold"]["default"] == str(
        DEFAULT_SPOT_GRID_CONFIG.high_volatility_pause_threshold
    )
    assert fields["high_volatility_pause_threshold"]["min"] == str(HIGH_VOLATILITY_PAUSE_THRESHOLD_BOUNDS.min)
    assert fields["high_volatility_pause_threshold"]["max"] == str(HIGH_VOLATILITY_PAUSE_THRESHOLD_BOUNDS.max)
    assert fields["volatility_cooldown_runs"]["default"] == DEFAULT_SPOT_GRID_CONFIG.volatility_cooldown_runs
    assert fields["volatility_cooldown_runs"]["min"] == VOLATILITY_COOLDOWN_RUNS_BOUNDS.min
    assert fields["volatility_cooldown_runs"]["max"] == VOLATILITY_COOLDOWN_RUNS_BOUNDS.max
    assert fields["emit_diagnostics"]["default"] == DEFAULT_SPOT_GRID_CONFIG.emit_diagnostics


def test_phase_11_spot_grid_bot_config_has_no_env_or_runtime_side_effect_imports() -> None:
    source = (SPOT_GRID_ROOT / "bot_config.py").read_text(encoding="utf-8")
    forbidden_terms = (
        "os.getenv",
        "environ",
        ".env",
        "asyncpg",
        "sqlalchemy",
        "redis",
        "requests",
        "httpx",
        "pybit",
        "SecretProvider",
        "spot_grid_bot",
    )

    assert [term for term in forbidden_terms if term in source] == []


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


def _snapshot(*, snapshot_id: str = "snapshot-1", timeframe: str = "1h") -> BotMarketSnapshot:
    now = datetime(2026, 7, 15, tzinfo=UTC)
    candles = (
        BotCandle(
            source="binance_spot",
            canonical_symbol="ETHUSDT",
            timeframe=timeframe,
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
            timeframe=timeframe,
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
        snapshot_id=snapshot_id,
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        provider_symbol="ETHUSDT",
        timeframe=timeframe,
        last_closed_candle_time=now,
        lookback_start_time=now,
        lookback_end_time=now,
        completeness_status="COMPLETE",
        snapshot_version=1,
        data_hash=f"hash-{snapshot_id}",
        candles=candles,
    )
