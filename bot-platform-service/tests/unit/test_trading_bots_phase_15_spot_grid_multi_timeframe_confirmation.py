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
from bot_platform_service.trading_bots.spot_grid.application import (
    SpotGridTradingCycleService,
    parse_spot_grid_config,
)
from bot_platform_service.trading_bots.spot_grid.infrastructure import PlatformSnapshotIndicatorAdapter
from tests.fixtures.spot_grid_indicator_runtime import FakeStockIndicatorsRuntime


SERVICE_ROOT = Path(__file__).resolve().parents[2]
SPOT_GRID_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "trading_bots" / "spot_grid"
INDICATOR_RUNTIME = FakeStockIndicatorsRuntime()


def test_phase_15_supporting_4h_downtrend_blocks_new_buy_intents() -> None:
    adapter = _adapter()

    result = asyncio.run(
        adapter.run_once(
            _request(
                config={"max_grid_levels": 2, "supporting_timeframes": ["4h"]},
                market_data=BotMarketDataContext(
                    primary_snapshot=_snapshot(
                        snapshot_id="snapshot-1h-up",
                        timeframe="1h",
                        candles=_trend_market_candles(timeframe="1h", start=Decimal("100"), step=Decimal("1")),
                    ),
                    supporting_snapshots=(
                        _snapshot(
                            snapshot_id="snapshot-4h-down",
                            timeframe="4h",
                            candles=_trend_market_candles(timeframe="4h", start=Decimal("130"), step=Decimal("-1")),
                        ),
                    ),
                ),
            )
        )
    )

    assert result.status is BotRunStatus.COMPLETE
    assert result.signals == ()
    assert result.diagnostics["regime_decision"]["regime"] == "uptrend"
    assert result.diagnostics["multi_timeframe_confirmation"]["new_buy_allowed"] is False
    assert result.diagnostics["multi_timeframe_confirmation"]["reason_codes"] == ("supporting_4h_downtrend_block",)
    assert result.diagnostics["multi_timeframe_confirmation"]["supporting_regimes"][0]["regime"] == "downtrend"
    assert result.diagnostics["entry_block_reasons"] == ("supporting_4h_downtrend_block",)


def test_phase_15_supporting_4h_uptrend_allows_primary_buy_intents() -> None:
    adapter = _adapter()

    result = asyncio.run(
        adapter.run_once(
            _request(
                config={"max_grid_levels": 2, "supporting_timeframes": ["4h"]},
                market_data=BotMarketDataContext(
                    primary_snapshot=_snapshot(
                        snapshot_id="snapshot-1h-up",
                        timeframe="1h",
                        candles=_trend_market_candles(timeframe="1h", start=Decimal("100"), step=Decimal("1")),
                    ),
                    supporting_snapshots=(
                        _snapshot(
                            snapshot_id="snapshot-4h-up",
                            timeframe="4h",
                            candles=_trend_market_candles(timeframe="4h", start=Decimal("100"), step=Decimal("1")),
                        ),
                    ),
                ),
            )
        )
    )

    assert result.status is BotRunStatus.COMPLETE
    assert result.signals == ()
    assert result.diagnostics["multi_timeframe_confirmation"]["new_buy_allowed"] is True
    assert result.diagnostics["multi_timeframe_confirmation"]["reason_codes"] == ()
    assert result.diagnostics["multi_timeframe_confirmation"]["supporting_regimes"][0]["regime"] == "uptrend"
    assert "rsi_buy_block" in result.diagnostics["rsi_block_reasons"]


def test_phase_15_missing_configured_supporting_4h_has_conservative_fallback() -> None:
    service = SpotGridTradingCycleService(
        snapshot_adapter=PlatformSnapshotIndicatorAdapter(required_history=20),
        indicator_runtime=INDICATOR_RUNTIME,
    )
    config = parse_spot_grid_config(
        {"max_grid_levels": 2, "supporting_timeframes": ["4h"]},
        fallback_symbols=("ETHUSDT",),
        fallback_timeframes=("1h", "4h"),
    )

    result = service.run_once(
        market_data=BotMarketDataContext(
            primary_snapshot=_snapshot(
                snapshot_id="snapshot-1h-up",
                timeframe="1h",
                candles=_trend_market_candles(timeframe="1h", start=Decimal("100"), step=Decimal("1")),
            )
        ),
        config=config,
    )

    confirmation = result.diagnostics["multi_timeframe_confirmation"]
    assert confirmation["status"] == "missing"
    assert confirmation["new_buy_allowed"] is False
    assert confirmation["reason_codes"] == ("supporting_4h_missing",)
    assert result.diagnostics["entry_block_reasons"] == ("supporting_4h_missing",)
    assert result.plan.levels == ()
    assert result.plan.intents == ()


def test_phase_15_single_timeframe_config_does_not_require_supporting_4h() -> None:
    service = SpotGridTradingCycleService(
        snapshot_adapter=PlatformSnapshotIndicatorAdapter(required_history=20),
        indicator_runtime=INDICATOR_RUNTIME,
    )
    config = parse_spot_grid_config(
        {"max_grid_levels": 1},
        fallback_symbols=("ETHUSDT",),
        fallback_timeframes=("1h",),
    )

    result = service.run_once(
        market_data=BotMarketDataContext(
            primary_snapshot=_snapshot(
                snapshot_id="snapshot-1h-up",
                timeframe="1h",
                candles=_trend_market_candles(timeframe="1h", start=Decimal("100"), step=Decimal("1")),
            )
        ),
        config=config,
    )

    confirmation = result.diagnostics["multi_timeframe_confirmation"]
    assert confirmation["status"] == "not_configured"
    assert confirmation["new_buy_allowed"] is True
    assert confirmation["reason_codes"] == ()
    assert result.plan.levels == ()
    assert result.plan.intents == ()
    assert "rsi_buy_block" in result.diagnostics["rsi_block_reasons"]


def test_phase_15_primary_and_supporting_regimes_are_visible_in_diagnostics() -> None:
    service = SpotGridTradingCycleService(
        snapshot_adapter=PlatformSnapshotIndicatorAdapter(required_history=20),
        indicator_runtime=INDICATOR_RUNTIME,
    )
    config = parse_spot_grid_config(
        {"max_grid_levels": 1, "supporting_timeframes": ["4h"]},
        fallback_symbols=("ETHUSDT",),
        fallback_timeframes=("1h", "4h"),
    )

    result = service.run_once(
        market_data=BotMarketDataContext(
            primary_snapshot=_snapshot(
                snapshot_id="snapshot-1h-up",
                timeframe="1h",
                candles=_trend_market_candles(timeframe="1h", start=Decimal("100"), step=Decimal("1")),
            ),
            supporting_snapshots=(
                _snapshot(
                    snapshot_id="snapshot-4h-down",
                    timeframe="4h",
                    candles=_trend_market_candles(timeframe="4h", start=Decimal("130"), step=Decimal("-1")),
                ),
            ),
        ),
        config=config,
    )

    assert result.diagnostics["regime_decision"]["regime"] == "uptrend"
    supporting = result.diagnostics["multi_timeframe_confirmation"]["supporting_regimes"][0]
    assert supporting["timeframe"] == "4h"
    assert supporting["snapshot_id"] == "snapshot-4h-down"
    assert supporting["regime"] == "downtrend"
    assert supporting["regime_decision"]["reasons"] == ("bearish_single_timeframe",)


def test_phase_15_application_source_is_platform_native_and_boundary_safe() -> None:
    source = (
        SPOT_GRID_ROOT / "application" / "trading_cycle_service.py"
    ).read_text(encoding="utf-8")
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


def _adapter() -> SpotGridAdapter:
    return SpotGridAdapter(
        cycle_service=SpotGridTradingCycleService(
            snapshot_adapter=PlatformSnapshotIndicatorAdapter(required_history=20),
            indicator_runtime=INDICATOR_RUNTIME,
        )
    )


def _request(
    *,
    config: dict[str, object],
    market_data: BotMarketDataContext,
) -> BotRunRequest:
    return BotRunRequest(
        run_id="run-phase-15",
        instance_id="instance-1",
        module_id="spot_grid",
        mode=BotMode.SIGNAL_ONLY,
        trigger_type=BotTriggerType.MANUAL,
        config=config,
        market_data=market_data,
    )


def _snapshot(*, snapshot_id: str, timeframe: str, candles: tuple[BotCandle, ...]) -> BotMarketSnapshot:
    now = datetime(2026, 1, 1, tzinfo=UTC)
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


def _trend_market_candles(*, timeframe: str, start: Decimal, step: Decimal) -> tuple[BotCandle, ...]:
    return tuple(
        _market_candle(timeframe=timeframe, index=index, close=start + (step * Decimal(index))) for index in range(30)
    )


def _market_candle(*, timeframe: str, index: int, close: Decimal) -> BotCandle:
    open_time = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=index)
    return BotCandle(
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        timeframe=timeframe,
        open_time=open_time,
        close_time=open_time + timedelta(hours=1),
        open=close,
        high=close + Decimal("2"),
        low=close - Decimal("2"),
        close=close,
        volume=Decimal("1000"),
    )
