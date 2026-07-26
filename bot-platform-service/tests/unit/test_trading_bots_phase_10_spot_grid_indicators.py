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
    IndicatorInput,
    IndicatorSnapshot,
    compute_core_indicators,
)
from bot_platform_service.trading_bots.spot_grid.infrastructure import PlatformSnapshotIndicatorAdapter
from tests.fixtures.spot_grid_indicator_runtime import FakeStockIndicatorsRuntime


SERVICE_ROOT = Path(__file__).resolve().parents[2]
SPOT_GRID_DOMAIN_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "trading_bots" / "spot_grid" / "domain"
INDICATOR_RUNTIME = FakeStockIndicatorsRuntime()


def test_phase_10_empty_indicator_input_returns_empty_decimal_snapshot() -> None:
    snapshot = compute_core_indicators(_indicator_input(candles=()))

    assert snapshot == IndicatorSnapshot(
        ema20=None,
        ema50=None,
        ema200=None,
        atr14=None,
        rsi14=None,
        realized_volatility=None,
        realized_volatility_short=None,
        current_volume=None,
        volume_ma20=None,
        volume_ratio=None,
        candle_count=0,
        has_required_history=False,
        volatility_has_required_history=False,
        volume_has_required_history=False,
    )
    assert snapshot.to_payload() == {
        "ema20": None,
        "ema50": None,
        "ema200": None,
        "atr14": None,
        "rsi14": None,
        "realized_volatility": None,
        "realized_volatility_short": None,
        "current_volume": None,
        "volume_ma20": None,
        "volume_ratio": None,
        "candle_count": 0,
        "has_required_history": False,
        "volatility_has_required_history": False,
        "volume_has_required_history": False,
    }


def test_phase_10_single_candle_indicator_snapshot_has_documented_fallbacks() -> None:
    snapshot = compute_core_indicators(
        _indicator_input(
            candles=(
                _indicator_candle(index=0, close=Decimal("100"), high=Decimal("103"), low=Decimal("97")),
            )
        ),
        runtime=INDICATOR_RUNTIME,
    )

    assert snapshot.to_payload() == {
        "ema20": "100",
        "ema50": "100",
        "ema200": "100",
        "atr14": "6",
        "rsi14": "50",
        "realized_volatility": "0",
        "realized_volatility_short": "0",
        "current_volume": "1000",
        "volume_ma20": "1000",
        "volume_ratio": "1",
        "candle_count": 1,
        "has_required_history": False,
        "volatility_has_required_history": False,
        "volume_has_required_history": False,
    }


def test_phase_10_fixture_candles_produce_stable_ema_atr_rsi_values() -> None:
    snapshot = compute_core_indicators(_indicator_input(candles=_ascending_indicator_candles()), runtime=INDICATOR_RUNTIME)

    assert snapshot.to_payload() == {
        "ema20": "106.8399355325756241434511009750154",
        "ema50": "103.4935769206884094201392933841419",
        "ema200": "101.0010435027244266965270240435157",
        "atr14": "4",
        "rsi14": "100",
        "realized_volatility": "0.0003565229064740196326951069018486654",
        "realized_volatility_short": "0.0001148189533804514266094031406021149",
        "current_volume": "1014",
        "volume_ma20": "1007",
        "volume_ratio": "1.006951340615690168818272095332671",
        "candle_count": 15,
        "has_required_history": False,
        "volatility_has_required_history": False,
        "volume_has_required_history": False,
    }
    assert isinstance(snapshot.ema20, Decimal)
    assert isinstance(snapshot.ema50, Decimal)
    assert isinstance(snapshot.ema200, Decimal)
    assert isinstance(snapshot.atr14, Decimal)
    assert isinstance(snapshot.rsi14, Decimal)


def test_phase_10_trading_cycle_includes_indicator_snapshot_diagnostics() -> None:
    service = SpotGridTradingCycleService(
        snapshot_adapter=PlatformSnapshotIndicatorAdapter(required_history=10),
        indicator_runtime=INDICATOR_RUNTIME,
    )
    config = parse_spot_grid_config(
        {"max_grid_levels": 1},
        fallback_symbols=("ETHUSDT",),
        fallback_timeframes=("1h",),
    )

    result = service.run_once(
        market_data=BotMarketDataContext(primary_snapshot=_market_snapshot(candles=_ascending_market_candles())),
        config=config,
    )

    assert result.diagnostics["indicator_candle_count"] == 15
    assert result.diagnostics["indicator_required_history"] == 10
    assert result.diagnostics["indicator_has_required_history"] is True
    assert result.diagnostics["indicator_volatility_has_required_history"] is False
    assert result.diagnostics["indicator_volume_has_required_history"] is False
    assert result.diagnostics["indicators"] == {
        "ema20": "106.8399355325756241434511009750154",
        "ema50": "103.4935769206884094201392933841419",
        "ema200": "101.0010435027244266965270240435157",
        "atr14": "4",
        "rsi14": "100",
        "realized_volatility": "0.0003565229064740196326951069018486654",
        "realized_volatility_short": "0.0001148189533804514266094031406021149",
        "current_volume": "1014",
        "volume_ma20": "1007",
        "volume_ratio": "1.006951340615690168818272095332671",
        "candle_count": 15,
        "has_required_history": True,
        "volatility_has_required_history": False,
        "volume_has_required_history": False,
    }


def test_phase_10_indicator_source_is_decimal_only_and_boundary_safe() -> None:
    source = (SPOT_GRID_DOMAIN_ROOT / "indicators.py").read_text(encoding="utf-8")
    forbidden_terms = (
        "math.",
        "spot_grid_bot",
        "sqlalchemy",
        "asyncpg",
        "pandas",
        "numpy",
    )

    assert [term for term in forbidden_terms if term in source] == []
    assert "load_stock_indicators_runtime" in source
    assert "from stock_indicators import indicators" in source
    assert "get_ema" in source
    assert "get_atr" in source
    assert "get_rsi" in source
    assert "_ema_last" not in source
    assert "_atr_last" not in source
    assert "_rsi_last" not in source


def _indicator_input(*, candles: tuple[IndicatorCandle, ...]) -> IndicatorInput:
    return IndicatorInput(
        source="binance_spot",
        symbol="eth/usdt",
        timeframe="1h",
        snapshot_id="snapshot-phase-10",
        snapshot_version=1,
        data_hash="hash-phase-10",
        candles=candles,
        required_history=200,
    )


def _ascending_indicator_candles() -> tuple[IndicatorCandle, ...]:
    return tuple(
        _indicator_candle(
            index=index,
            close=Decimal(100 + index),
            high=Decimal(100 + index + 2),
            low=Decimal(100 + index - 2),
        )
        for index in range(15)
    )


def _indicator_candle(*, index: int, close: Decimal, high: Decimal, low: Decimal) -> IndicatorCandle:
    return IndicatorCandle(
        timestamp=(datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=index)).isoformat(),
        open=close - Decimal("1"),
        high=high,
        low=low,
        close=close,
        volume=Decimal(1000 + index),
    )


def _market_snapshot(*, candles: tuple[BotCandle, ...]) -> BotMarketSnapshot:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return BotMarketSnapshot(
        snapshot_id="snapshot-phase-10",
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        provider_symbol="ETHUSDT",
        timeframe="1h",
        last_closed_candle_time=now,
        lookback_start_time=now,
        lookback_end_time=now,
        completeness_status="COMPLETE",
        snapshot_version=1,
        data_hash="hash-phase-10",
        candles=candles,
    )


def _ascending_market_candles() -> tuple[BotCandle, ...]:
    return tuple(
        _market_candle(
            index=index,
            close=Decimal(100 + index),
            high=Decimal(100 + index + 2),
            low=Decimal(100 + index - 2),
        )
        for index in range(15)
    )


def _market_candle(*, index: int, close: Decimal, high: Decimal, low: Decimal) -> BotCandle:
    open_time = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=index)
    return BotCandle(
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        timeframe="1h",
        open_time=open_time,
        close_time=open_time + timedelta(hours=1),
        open=close - Decimal("1"),
        high=high,
        low=low,
        close=close,
        volume=Decimal(1000 + index),
    )
