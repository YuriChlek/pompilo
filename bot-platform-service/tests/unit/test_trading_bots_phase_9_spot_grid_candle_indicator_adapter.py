from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from bot_platform_service.domain import BotCandle, BotMarketDataContext, BotMarketSnapshot
from bot_platform_service.trading_bots.spot_grid.application import (
    SpotGridTradingCycleService,
    parse_spot_grid_config,
)
from bot_platform_service.trading_bots.spot_grid.domain import IndicatorCandle, IndicatorInput, StockIndicatorsRuntime
from bot_platform_service.trading_bots.spot_grid.infrastructure import PlatformSnapshotIndicatorAdapter
from tests.fixtures.spot_grid_indicator_runtime import FakeStockIndicatorsRuntime


SERVICE_ROOT = Path(__file__).resolve().parents[2]
SPOT_GRID_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "trading_bots" / "spot_grid"
INDICATOR_RUNTIME = FakeStockIndicatorsRuntime()


@dataclass(frozen=True, slots=True)
class _FakeQuote:
    date: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


def test_phase_9_platform_snapshot_adapter_orders_and_normalizes_candles() -> None:
    adapter = PlatformSnapshotIndicatorAdapter(required_history=3)
    snapshot = _snapshot(
        candles=(
            _candle(offset_minutes=2, open_price="102", high="105", low="101", close="104", volume="12"),
            _candle(offset_minutes=0, open_price="100", high="103", low="99", close="101", volume="10"),
            _candle(offset_minutes=1, open_price="101", high="104", low="100", close="102", volume="11"),
        )
    )

    result = adapter.adapt(snapshot)

    assert isinstance(result, IndicatorInput)
    assert result.source == "binance_spot"
    assert result.symbol == "BTCUSDT"
    assert result.timeframe == "1h"
    assert result.snapshot_id == "snapshot-phase-9"
    assert result.snapshot_version == 7
    assert result.data_hash == "hash-phase-9"
    assert result.candle_count == 3
    assert result.required_history == 3
    assert result.has_required_history is True
    assert [candle.close for candle in result.candles] == [Decimal("101"), Decimal("102"), Decimal("104")]
    assert [candle.timestamp for candle in result.candles] == [
        "2026-07-15T00:01:00+00:00",
        "2026-07-15T00:02:00+00:00",
        "2026-07-15T00:03:00+00:00",
    ]
    assert all(isinstance(candle, IndicatorCandle) for candle in result.candles)


def test_phase_9_indicator_input_payload_is_json_safe_and_decimal_precise() -> None:
    adapter = PlatformSnapshotIndicatorAdapter(required_history=1)
    result = adapter.adapt(
        _snapshot(
            candles=(
                _candle(
                    offset_minutes=0,
                    open_price="100.000000000000000001",
                    high="101.000000000000000001",
                    low="99.000000000000000001",
                    close="100.500000000000000001",
                    volume="12.000000000000000001",
                ),
            )
        )
    )

    assert result.to_payload() == {
        "source": "binance_spot",
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "snapshot_id": "snapshot-phase-9",
        "snapshot_version": 7,
        "data_hash": "hash-phase-9",
        "candle_count": 1,
        "required_history": 1,
        "has_required_history": True,
        "candles": [
            {
                "timestamp": "2026-07-15T00:01:00+00:00",
                "open": "100.000000000000000001",
                "high": "101.000000000000000001",
                "low": "99.000000000000000001",
                "close": "100.500000000000000001",
                "volume": "12.000000000000000001",
            }
        ],
    }


def test_phase_9_indicator_input_builds_stock_indicators_quotes_without_precision_loss() -> None:
    adapter = PlatformSnapshotIndicatorAdapter(required_history=1)
    indicator_input = adapter.adapt(
        _snapshot(
            candles=(
                _candle(
                    offset_minutes=0,
                    open_price="100.000000000000000001",
                    high="101.000000000000000001",
                    low="99.000000000000000001",
                    close="100.500000000000000001",
                    volume="12.000000000000000001",
                ),
            )
        )
    )
    runtime = StockIndicatorsRuntime(indicators=object(), quote_type=_FakeQuote, candle_part=object())

    quotes = runtime.build_quotes(indicator_input.candles)

    assert quotes == (
        _FakeQuote(
            date=datetime(2026, 7, 15, 0, 1, tzinfo=UTC),
            open=Decimal("100.000000000000000001"),
            high=Decimal("101.000000000000000001"),
            low=Decimal("99.000000000000000001"),
            close=Decimal("100.500000000000000001"),
            volume=Decimal("12.000000000000000001"),
        ),
    )


def test_phase_9_empty_snapshot_returns_empty_indicator_input() -> None:
    adapter = PlatformSnapshotIndicatorAdapter(required_history=2)

    result = adapter.adapt(_snapshot(candles=()))

    assert result.candles == ()
    assert result.candle_count == 0
    assert result.has_required_history is False
    assert result.to_payload()["candles"] == []


def test_phase_9_short_history_is_reported_in_trading_cycle_diagnostics() -> None:
    service = SpotGridTradingCycleService(
        snapshot_adapter=PlatformSnapshotIndicatorAdapter(required_history=5),
        indicator_runtime=INDICATOR_RUNTIME,
    )
    config = parse_spot_grid_config(
        {"max_grid_levels": 1},
        fallback_symbols=("BTCUSDT",),
        fallback_timeframes=("1h",),
    )

    result = service.run_once(
        market_data=BotMarketDataContext(
            primary_snapshot=_snapshot(
                candles=(
                    _candle(offset_minutes=0, open_price="100", high="101", low="99", close="100", volume="10"),
                    _candle(offset_minutes=1, open_price="100", high="102", low="99", close="101", volume="11"),
                )
            )
        ),
        config=config,
    )

    assert result.diagnostics["indicator_candle_count"] == 2
    assert result.diagnostics["indicator_required_history"] == 5
    assert result.diagnostics["indicator_has_required_history"] is False


def test_phase_9_platform_snapshot_adapter_does_not_read_legacy_candle_tables() -> None:
    source = (SPOT_GRID_ROOT / "infrastructure" / "platform_snapshot_adapter.py").read_text(encoding="utf-8")
    forbidden_terms = (
        "spot_grid_bot",
        "candle_store",
        "ensure_candle_tables",
        "DatabaseMarketDataProvider",
        "select(",
        "sqlalchemy",
        "asyncpg",
        "postgres",
    )

    assert [term for term in forbidden_terms if term in source] == []


def _snapshot(*, candles: tuple[BotCandle, ...]) -> BotMarketSnapshot:
    now = datetime(2026, 7, 15, tzinfo=UTC)
    return BotMarketSnapshot(
        snapshot_id="snapshot-phase-9",
        source="binance_spot",
        canonical_symbol="btc/usdt",
        provider_symbol="BTCUSDT",
        timeframe="1h",
        last_closed_candle_time=now,
        lookback_start_time=now,
        lookback_end_time=now,
        completeness_status="COMPLETE",
        snapshot_version=7,
        data_hash="hash-phase-9",
        candles=candles,
    )


def _candle(
    *,
    offset_minutes: int,
    open_price: str,
    high: str,
    low: str,
    close: str,
    volume: str,
) -> BotCandle:
    open_time = datetime(2026, 7, 15, tzinfo=UTC) + timedelta(minutes=offset_minutes)
    close_time = open_time + timedelta(minutes=1)
    return BotCandle(
        source="binance_spot",
        canonical_symbol="BTCUSDT",
        timeframe="1h",
        open_time=open_time,
        close_time=close_time,
        open=Decimal(open_price),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal(volume),
    )
