from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from bot_platform_service.domain import BotCandle, BotMarketDataContext, BotMarketSnapshot
from bot_platform_service.trading_bots.spot_grid.application import (
    SpotGridTradingCycleService,
    parse_spot_grid_config,
)
from bot_platform_service.trading_bots.spot_grid.domain import (
    IndicatorCandle,
    IndicatorInput,
    compute_core_indicators,
)
from bot_platform_service.trading_bots.spot_grid.infrastructure import PlatformSnapshotIndicatorAdapter
from tests.fixtures.spot_grid_indicator_runtime import FakeStockIndicatorsRuntime


INDICATOR_RUNTIME = FakeStockIndicatorsRuntime()


def test_phase_11_fixture_candles_produce_deterministic_volatility_and_volume_metrics() -> None:
    snapshot = compute_core_indicators(
        _indicator_input(
            candles=(
                _indicator_candle(index=0, close=Decimal("100"), volume=Decimal("1000")),
                _indicator_candle(index=1, close=Decimal("110"), volume=Decimal("1200")),
                _indicator_candle(index=2, close=Decimal("99"), volume=Decimal("1300")),
            )
        ),
        runtime=INDICATOR_RUNTIME,
    )

    assert snapshot.to_payload() == {
        "ema20": "100.7664399092970521541950113378685",
        "ema50": "100.3375624759707804690503652441369",
        "ema200": "100.0885621643028637905002351426945",
        "atr14": "8.333333333333333333333333333333333",
        "rsi14": "47.61904761904761904761904761904762",
        "realized_volatility": "0.1",
        "realized_volatility_short": "0.1",
        "current_volume": "1300",
        "volume_ma20": "1166.666666666666666666666666666667",
        "volume_ratio": "1.114285714285714285714285714285714",
        "candle_count": 3,
        "has_required_history": False,
        "volatility_has_required_history": False,
        "volume_has_required_history": False,
    }


def test_phase_11_single_candle_has_documented_short_history_fallback_metrics() -> None:
    snapshot = compute_core_indicators(
        _indicator_input(candles=(_indicator_candle(index=0, close=Decimal("100"), volume=Decimal("1250")),)),
        runtime=INDICATOR_RUNTIME,
    )

    assert snapshot.realized_volatility == Decimal("0")
    assert snapshot.realized_volatility_short == Decimal("0")
    assert snapshot.current_volume == Decimal("1250")
    assert snapshot.volume_ma20 == Decimal("1250")
    assert snapshot.volume_ratio == Decimal("1")
    assert snapshot.volatility_has_required_history is False
    assert snapshot.volume_has_required_history is False


def test_phase_11_diagnostics_show_metric_history_sufficiency() -> None:
    service = SpotGridTradingCycleService(
        snapshot_adapter=PlatformSnapshotIndicatorAdapter(required_history=3),
        indicator_runtime=INDICATOR_RUNTIME,
    )
    config = parse_spot_grid_config(
        {"max_grid_levels": 1},
        fallback_symbols=("ETHUSDT",),
        fallback_timeframes=("1h",),
    )

    result = service.run_once(
        market_data=BotMarketDataContext(primary_snapshot=_market_snapshot(candles=_volume_ready_market_candles())),
        config=config,
    )

    assert result.diagnostics["indicator_has_required_history"] is True
    assert result.diagnostics["indicator_volatility_has_required_history"] is False
    assert result.diagnostics["indicator_volume_has_required_history"] is True
    assert result.diagnostics["indicators"]["realized_volatility"] == "0"
    assert result.diagnostics["indicators"]["current_volume"] == "100"
    assert result.diagnostics["indicators"]["volume_ma20"] == "100"
    assert result.diagnostics["indicators"]["volume_ratio"] == "1"
    assert result.diagnostics["indicators"]["volatility_has_required_history"] is False
    assert result.diagnostics["indicators"]["volume_has_required_history"] is True


def _indicator_input(*, candles: tuple[IndicatorCandle, ...]) -> IndicatorInput:
    return IndicatorInput(
        source="binance_spot",
        symbol="eth/usdt",
        timeframe="1h",
        snapshot_id="snapshot-phase-11",
        snapshot_version=1,
        data_hash="hash-phase-11",
        candles=candles,
        required_history=200,
    )


def _indicator_candle(*, index: int, close: Decimal, volume: Decimal) -> IndicatorCandle:
    return IndicatorCandle(
        timestamp=(datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=index)).isoformat(),
        open=close,
        high=close + Decimal("1"),
        low=close - Decimal("1"),
        close=close,
        volume=volume,
    )


def _market_snapshot(*, candles: tuple[BotCandle, ...]) -> BotMarketSnapshot:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return BotMarketSnapshot(
        snapshot_id="snapshot-phase-11",
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        provider_symbol="ETHUSDT",
        timeframe="1h",
        last_closed_candle_time=now,
        lookback_start_time=now,
        lookback_end_time=now,
        completeness_status="COMPLETE",
        snapshot_version=1,
        data_hash="hash-phase-11",
        candles=candles,
    )


def _volume_ready_market_candles() -> tuple[BotCandle, ...]:
    return tuple(_market_candle(index=index, close=Decimal("100"), volume=Decimal("100")) for index in range(20))


def _market_candle(*, index: int, close: Decimal, volume: Decimal) -> BotCandle:
    open_time = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=index)
    return BotCandle(
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        timeframe="1h",
        open_time=open_time,
        close_time=open_time + timedelta(hours=1),
        open=close,
        high=close + Decimal("1"),
        low=close - Decimal("1"),
        close=close,
        volume=volume,
    )
