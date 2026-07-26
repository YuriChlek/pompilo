from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from bot_platform_service.domain import (
    BotCandle,
    BotMarketDataContext,
    BotMarketSnapshot,
    BotMode,
    BotRunRequest,
    BotRunStatus,
    BotSignalSide,
    BotSignalType,
    BotTriggerType,
)
from bot_platform_service.trading_bots.spot_grid.adapter import SpotGridAdapter
from bot_platform_service.trading_bots.spot_grid.application import SpotGridTradingCycleService
from bot_platform_service.trading_bots.spot_grid.domain import (
    IndicatorSnapshot,
    MarketRegime,
    MarketStructureBias,
    MarketStructureSnapshot,
    detect_single_timeframe_regime,
)
from bot_platform_service.trading_bots.spot_grid.infrastructure import PlatformSnapshotIndicatorAdapter
from tests.fixtures.spot_grid_indicator_runtime import FakeStockIndicatorsRuntime


SERVICE_ROOT = Path(__file__).resolve().parents[2]
SPOT_GRID_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "trading_bots" / "spot_grid"
INDICATOR_RUNTIME = FakeStockIndicatorsRuntime()


def test_phase_13_range_regime_is_deterministic_and_explained() -> None:
    regime = detect_single_timeframe_regime(
        indicators=_indicators(
            ema20=Decimal("100"),
            ema50=Decimal("100"),
            ema200=Decimal("100"),
            rsi14=Decimal("50"),
            realized_volatility=Decimal("0.01"),
            atr14=Decimal("2"),
        ),
        market_structure=_structure(MarketStructureBias.RANGE),
    )

    assert regime.regime is MarketRegime.RANGE
    assert regime.confidence == Decimal("0.60")
    assert regime.reasons == ("range_single_timeframe",)
    assert regime.to_payload()["diagnostics"]["ema_alignment"] == "mixed"


def test_phase_13_uptrend_downtrend_high_volatility_and_risk_off_fixtures() -> None:
    uptrend = detect_single_timeframe_regime(
        indicators=_indicators(ema20=Decimal("120"), ema50=Decimal("110"), ema200=Decimal("100"), rsi14=Decimal("58")),
        market_structure=_structure(MarketStructureBias.BULLISH),
    )
    downtrend = detect_single_timeframe_regime(
        indicators=_indicators(ema20=Decimal("90"), ema50=Decimal("100"), ema200=Decimal("110"), rsi14=Decimal("39")),
        market_structure=_structure(MarketStructureBias.BEARISH),
    )
    high_volatility = detect_single_timeframe_regime(
        indicators=_indicators(
            ema20=Decimal("101"),
            ema50=Decimal("100"),
            ema200=Decimal("99"),
            rsi14=Decimal("50"),
            realized_volatility=Decimal("0.05"),
        ),
        market_structure=_structure(MarketStructureBias.RANGE),
    )
    risk_off = detect_single_timeframe_regime(
        indicators=_indicators(
            ema20=Decimal("90"),
            ema50=Decimal("100"),
            ema200=Decimal("110"),
            rsi14=Decimal("30"),
            realized_volatility=Decimal("0.05"),
        ),
        market_structure=_structure(MarketStructureBias.BEARISH),
    )

    assert uptrend.regime is MarketRegime.UPTREND
    assert uptrend.reasons == ("bullish_single_timeframe",)
    assert downtrend.regime is MarketRegime.DOWNTREND
    assert downtrend.reasons == ("bearish_single_timeframe",)
    assert high_volatility.regime is MarketRegime.HIGH_VOLATILITY
    assert high_volatility.reasons == ("high_volatility",)
    assert risk_off.regime is MarketRegime.RISK_OFF
    assert risk_off.reasons == ("risk_off_bearish_high_volatility",)


def test_phase_13_short_history_falls_back_to_range_with_diagnostics() -> None:
    regime = detect_single_timeframe_regime(
        indicators=_indicators(has_required_history=False),
        market_structure=_structure(MarketStructureBias.BULLISH, has_required_history=False),
    )

    assert regime.regime is MarketRegime.RANGE
    assert regime.confidence == Decimal("0.50")
    assert regime.reasons == (
        "insufficient_indicator_history",
        "insufficient_market_structure_history",
        "range_short_history_fallback",
    )
    assert regime.to_payload()["diagnostics"]["indicator_has_required_history"] is False
    assert regime.to_payload()["diagnostics"]["market_structure_has_required_history"] is False


def test_phase_13_trading_cycle_diagnostics_include_regime_decision() -> None:
    service = SpotGridTradingCycleService(
        snapshot_adapter=PlatformSnapshotIndicatorAdapter(required_history=20),
        indicator_runtime=INDICATOR_RUNTIME,
    )
    config = _config(max_grid_levels=1)

    result = service.run_once(
        market_data=BotMarketDataContext(primary_snapshot=_snapshot(candles=_downtrend_market_candles())),
        config=config,
    )

    assert result.plan.regime is MarketRegime.DOWNTREND
    assert result.diagnostics["regime"] == "downtrend"
    assert result.diagnostics["regime_decision"]["regime"] == "downtrend"
    assert "bearish_single_timeframe" in result.diagnostics["regime_decision"]["reasons"]
    assert result.diagnostics["entry_allowed"] is False


def test_phase_13_downtrend_does_not_create_entry_signals() -> None:
    adapter = SpotGridAdapter(
        cycle_service=SpotGridTradingCycleService(
            snapshot_adapter=PlatformSnapshotIndicatorAdapter(required_history=20),
            indicator_runtime=INDICATOR_RUNTIME,
        )
    )

    result = asyncio.run(
        adapter.run_once(
            _request(
                market_data=BotMarketDataContext(primary_snapshot=_snapshot(candles=_downtrend_market_candles())),
                config={"max_grid_levels": 2},
            )
        )
    )

    assert result.status is BotRunStatus.COMPLETE
    assert result.signals == ()
    assert result.diagnostics["signal_count"] == 0
    assert "rsi_sell_block" in result.diagnostics["rsi_block_reasons"]


def test_phase_13_regime_detector_source_is_platform_native_and_boundary_safe() -> None:
    source = (SPOT_GRID_ROOT / "domain" / "regime_detector.py").read_text(encoding="utf-8")
    forbidden_terms = (
        "spot_grid_bot",
        "sqlalchemy",
        "asyncpg",
        "candle_1h",
        "candle_4h",
        "candle_1d",
        "float",
        "requests",
        "pybit",
    )

    assert [term for term in forbidden_terms if term in source] == []


def _indicators(
    *,
    ema20: Decimal = Decimal("100"),
    ema50: Decimal = Decimal("100"),
    ema200: Decimal = Decimal("100"),
    atr14: Decimal = Decimal("2"),
    rsi14: Decimal = Decimal("50"),
    realized_volatility: Decimal = Decimal("0.01"),
    has_required_history: bool = True,
) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        ema20=ema20,
        ema50=ema50,
        ema200=ema200,
        atr14=atr14,
        rsi14=rsi14,
        realized_volatility=realized_volatility,
        realized_volatility_short=realized_volatility,
        current_volume=Decimal("1000"),
        volume_ma20=Decimal("1000"),
        volume_ratio=Decimal("1"),
        candle_count=30,
        has_required_history=has_required_history,
        volatility_has_required_history=True,
        volume_has_required_history=True,
    )


def _structure(
    bias: MarketStructureBias,
    *,
    has_required_history: bool = True,
) -> MarketStructureSnapshot:
    return MarketStructureSnapshot(
        bias=bias,
        candle_count=30,
        has_required_history=has_required_history,
        range_low=Decimal("90"),
        range_high=Decimal("110"),
        range_position=Decimal("0.5"),
        swing_highs=(),
        swing_lows=(),
        support_candidates=(),
        resistance_candidates=(),
        breakout_direction=None,
        breakout_reference_price=None,
        reasons=("test_fixture",),
    )


def _config(*, max_grid_levels: int):
    from bot_platform_service.trading_bots.spot_grid.application import parse_spot_grid_config

    return parse_spot_grid_config(
        {"max_grid_levels": max_grid_levels},
        fallback_symbols=("ETHUSDT",),
        fallback_timeframes=("1h",),
    )


def _request(
    *,
    market_data: BotMarketDataContext,
    config: dict[str, object],
) -> BotRunRequest:
    return BotRunRequest(
        run_id="run-phase-13",
        instance_id="instance-1",
        module_id="spot_grid",
        mode=BotMode.SIGNAL_ONLY,
        trigger_type=BotTriggerType.MANUAL,
        config=config,
        market_data=market_data,
    )


def _snapshot(*, candles: tuple[BotCandle, ...]) -> BotMarketSnapshot:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return BotMarketSnapshot(
        snapshot_id="snapshot-phase-13",
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        provider_symbol="ETHUSDT",
        timeframe="1h",
        last_closed_candle_time=now,
        lookback_start_time=now,
        lookback_end_time=now,
        completeness_status="COMPLETE",
        snapshot_version=1,
        data_hash="hash-phase-13",
        candles=candles,
    )


def _downtrend_market_candles() -> tuple[BotCandle, ...]:
    return tuple(_market_candle(index=index, close=Decimal(130 - index)) for index in range(30))


def _market_candle(*, index: int, close: Decimal) -> BotCandle:
    open_time = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=index)
    return BotCandle(
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        timeframe="1h",
        open_time=open_time,
        close_time=open_time + timedelta(hours=1),
        open=close + Decimal("1"),
        high=close + Decimal("2"),
        low=close - Decimal("2"),
        close=close,
        volume=Decimal("1000"),
    )
