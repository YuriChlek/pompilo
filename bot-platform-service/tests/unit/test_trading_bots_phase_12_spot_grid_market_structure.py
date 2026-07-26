from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from bot_platform_service.domain import BotCandle, BotMarketDataContext, BotMarketSnapshot
from bot_platform_service.trading_bots.spot_grid.application import (
    SpotGridTradingCycleService,
    parse_spot_grid_config,
)
from bot_platform_service.trading_bots.spot_grid.domain import (
    IndicatorCandle,
    MarketStructureBias,
    compute_market_structure,
)
from bot_platform_service.trading_bots.spot_grid.infrastructure import PlatformSnapshotIndicatorAdapter
from tests.fixtures.spot_grid_indicator_runtime import FakeStockIndicatorsRuntime


SERVICE_ROOT = Path(__file__).resolve().parents[2]
SPOT_GRID_DOMAIN_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "trading_bots" / "spot_grid" / "domain"
INDICATOR_RUNTIME = FakeStockIndicatorsRuntime()


def test_phase_12_range_fixture_produces_stable_swings_range_position_and_candidates() -> None:
    snapshot = compute_market_structure(_range_indicator_candles(), swing_window=1)

    assert snapshot.bias == MarketStructureBias.RANGE
    assert snapshot.range_low == Decimal("98")
    assert snapshot.range_high == Decimal("106")
    assert snapshot.range_position == Decimal("0.5")
    assert [swing.price for swing in snapshot.swing_highs] == [
        Decimal("106"),
        Decimal("106"),
        Decimal("106"),
        Decimal("106"),
    ]
    assert [swing.price for swing in snapshot.swing_lows] == [Decimal("98"), Decimal("98"), Decimal("98")]
    assert [candidate.to_payload() for candidate in snapshot.support_candidates] == [
        {"price": "98", "source": "swing_low", "candle_index": 6, "distance_from_close": "4"},
        {"price": "98", "source": "swing_low", "candle_index": 4, "distance_from_close": "4"},
        {"price": "98", "source": "swing_low", "candle_index": 2, "distance_from_close": "4"},
    ]
    assert [candidate.to_payload() for candidate in snapshot.resistance_candidates] == [
        {"price": "106", "source": "swing_high", "candle_index": 7, "distance_from_close": "4"},
        {"price": "106", "source": "swing_high", "candle_index": 5, "distance_from_close": "4"},
        {"price": "106", "source": "swing_high", "candle_index": 3, "distance_from_close": "4"},
    ]
    assert snapshot.breakout_direction is None


def test_phase_12_breakout_fixture_marks_bullish_structure_without_legacy_tables() -> None:
    snapshot = compute_market_structure(_breakout_indicator_candles(), swing_window=1)

    assert snapshot.bias == MarketStructureBias.BULLISH
    assert snapshot.range_low == Decimal("98")
    assert snapshot.range_high == Decimal("110")
    assert snapshot.range_position == Decimal("0.9166666666666666666666666666666667")
    assert snapshot.resistance_candidates == ()
    assert snapshot.breakout_direction == "up"
    assert snapshot.breakout_reference_price == Decimal("106")
    assert snapshot.reasons == ("up_breakout",)


def test_phase_12_short_history_has_documented_structure_fallback() -> None:
    snapshot = compute_market_structure(_range_indicator_candles()[:3], swing_window=1)

    assert snapshot.bias == MarketStructureBias.NEUTRAL
    assert snapshot.has_required_history is False
    assert snapshot.range_low == Decimal("98")
    assert snapshot.range_high == Decimal("106")
    assert snapshot.range_position == Decimal("0.375")
    assert snapshot.swing_highs == ()
    assert snapshot.swing_lows == ()
    assert snapshot.support_candidates == ()
    assert snapshot.resistance_candidates == ()
    assert snapshot.reasons == ("insufficient_candles",)


def test_phase_12_trading_cycle_diagnostics_include_grid_ready_market_structure() -> None:
    service = SpotGridTradingCycleService(
        snapshot_adapter=PlatformSnapshotIndicatorAdapter(required_history=5),
        indicator_runtime=INDICATOR_RUNTIME,
    )
    config = parse_spot_grid_config(
        {"max_grid_levels": 1},
        fallback_symbols=("ETHUSDT",),
        fallback_timeframes=("1h",),
    )

    result = service.run_once(
        market_data=BotMarketDataContext(primary_snapshot=_market_snapshot(candles=_range_market_candles())),
        config=config,
    )

    structure = result.diagnostics["market_structure"]
    assert structure["bias"] == "range"
    assert structure["range_low"] == "98"
    assert structure["range_high"] == "106"
    assert structure["range_position"] == "0.5"
    assert structure["support_candidates"][0]["price"] == "99"
    assert structure["resistance_candidates"][0]["price"] == "106"


def test_phase_12_market_structure_source_is_platform_native_and_boundary_safe() -> None:
    source = (SPOT_GRID_DOMAIN_ROOT / "market_structure.py").read_text(encoding="utf-8")
    forbidden_terms = (
        "spot_grid_bot",
        "sqlalchemy",
        "asyncpg",
        "candle_1h",
        "candle_4h",
        "candle_1d",
        "float",
    )

    assert [term for term in forbidden_terms if term in source] == []


def _range_indicator_candles() -> tuple[IndicatorCandle, ...]:
    return (
        _indicator_candle(index=0, high=Decimal("101"), low=Decimal("99"), close=Decimal("100")),
        _indicator_candle(index=1, high=Decimal("106"), low=Decimal("102"), close=Decimal("103")),
        _indicator_candle(index=2, high=Decimal("102"), low=Decimal("98"), close=Decimal("101")),
        _indicator_candle(index=3, high=Decimal("106"), low=Decimal("103"), close=Decimal("104")),
        _indicator_candle(index=4, high=Decimal("103"), low=Decimal("98"), close=Decimal("102")),
        _indicator_candle(index=5, high=Decimal("106"), low=Decimal("104"), close=Decimal("105")),
        _indicator_candle(index=6, high=Decimal("102"), low=Decimal("98"), close=Decimal("101")),
        _indicator_candle(index=7, high=Decimal("106"), low=Decimal("103"), close=Decimal("104")),
        _indicator_candle(index=8, high=Decimal("104"), low=Decimal("102"), close=Decimal("102")),
    )


def _breakout_indicator_candles() -> tuple[IndicatorCandle, ...]:
    return (
        *_range_indicator_candles()[:-1],
        _indicator_candle(index=8, high=Decimal("110"), low=Decimal("107"), close=Decimal("109")),
    )


def _indicator_candle(*, index: int, high: Decimal, low: Decimal, close: Decimal) -> IndicatorCandle:
    return IndicatorCandle(
        timestamp=(datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=index)).isoformat(),
        open=close,
        high=high,
        low=low,
        close=close,
        volume=Decimal("100"),
    )


def _market_snapshot(*, candles: tuple[BotCandle, ...]) -> BotMarketSnapshot:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return BotMarketSnapshot(
        snapshot_id="snapshot-phase-12",
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        provider_symbol="ETHUSDT",
        timeframe="1h",
        last_closed_candle_time=now,
        lookback_start_time=now,
        lookback_end_time=now,
        completeness_status="COMPLETE",
        snapshot_version=1,
        data_hash="hash-phase-12",
        candles=candles,
    )


def _range_market_candles() -> tuple[BotCandle, ...]:
    return tuple(
        _market_candle(
            index=index,
            high=candle.high,
            low=candle.low,
            close=candle.close,
        )
        for index, candle in enumerate(_range_indicator_candles())
    )


def _market_candle(*, index: int, high: Decimal, low: Decimal, close: Decimal) -> BotCandle:
    open_time = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=index)
    return BotCandle(
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        timeframe="1h",
        open_time=open_time,
        close_time=open_time + timedelta(hours=1),
        open=close,
        high=high,
        low=low,
        close=close,
        volume=Decimal("100"),
    )
