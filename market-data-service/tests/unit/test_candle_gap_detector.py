from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
import unittest

from market_data_service.domain.candle_gap_detector import detect_candle_range_status
from market_data_service.domain.candle_models import CanonicalCandle
from market_data_service.domain.enums import CandleRangeStatus, MarketDataSource


def _candle(open_time: datetime, *, timeframe: str = "1h", close_offset: timedelta = timedelta(hours=1)) -> CanonicalCandle:
    return CanonicalCandle(
        candle_id=open_time.isoformat(),
        source=MarketDataSource.BINANCE_SPOT,
        canonical_symbol="ETH/USDT",
        provider_symbol="ETHUSDT",
        timeframe=timeframe,
        open_time=open_time,
        close_time=open_time + close_offset,
        open=Decimal("100"),
        high=Decimal("110"),
        low=Decimal("90"),
        close=Decimal("105"),
        volume=Decimal("10"),
        quote_volume=Decimal("1050"),
        taker_buy_base_volume=None,
        taker_buy_quote_volume=None,
        taker_sell_base_volume=None,
        taker_sell_quote_volume=None,
        trades_count=None,
        is_closed=True,
        provider_payload_hash="hash",
    )


class CandleGapDetectorTests(unittest.TestCase):
    def test_complete_range_returns_complete_status(self) -> None:
        result = detect_candle_range_status(
            [
                _candle(datetime(2026, 7, 14, 8, tzinfo=UTC)),
                _candle(datetime(2026, 7, 14, 9, tzinfo=UTC)),
            ],
            timeframe="1h",
            expected_from=datetime(2026, 7, 14, 8, tzinfo=UTC),
            expected_to=datetime(2026, 7, 14, 10, tzinfo=UTC),
        )

        self.assertEqual(result.status, CandleRangeStatus.COMPLETE)
        self.assertEqual(result.expected_count, 2)
        self.assertEqual(result.gap_count, 0)

    def test_middle_gap_returns_gap_detected_status(self) -> None:
        result = detect_candle_range_status(
            [
                _candle(datetime(2026, 7, 14, 8, tzinfo=UTC)),
                _candle(datetime(2026, 7, 14, 10, tzinfo=UTC)),
            ],
            timeframe="1h",
            expected_from=datetime(2026, 7, 14, 8, tzinfo=UTC),
            expected_to=datetime(2026, 7, 14, 11, tzinfo=UTC),
        )

        self.assertEqual(result.status, CandleRangeStatus.GAP_DETECTED)
        self.assertEqual(result.gap_count, 1)
        self.assertEqual(result.missing_intervals[0].open_time, datetime(2026, 7, 14, 9, tzinfo=UTC))

    def test_missing_last_candle_returns_incomplete_status(self) -> None:
        result = detect_candle_range_status(
            [_candle(datetime(2026, 7, 14, 8, tzinfo=UTC))],
            timeframe="1h",
            expected_from=datetime(2026, 7, 14, 8, tzinfo=UTC),
            expected_to=datetime(2026, 7, 14, 10, tzinfo=UTC),
        )

        self.assertEqual(result.status, CandleRangeStatus.INCOMPLETE)
        self.assertEqual(result.missing_intervals[0].open_time, datetime(2026, 7, 14, 9, tzinfo=UTC))

    def test_duplicate_interval_returns_gap_detected_status(self) -> None:
        duplicated = _candle(datetime(2026, 7, 14, 8, tzinfo=UTC))

        result = detect_candle_range_status(
            [duplicated, duplicated],
            timeframe="1h",
            expected_from=datetime(2026, 7, 14, 8, tzinfo=UTC),
            expected_to=datetime(2026, 7, 14, 9, tzinfo=UTC),
        )

        self.assertEqual(result.status, CandleRangeStatus.GAP_DETECTED)
        self.assertEqual(result.duplicate_count, 1)

    def test_wrong_duration_returns_gap_detected_status(self) -> None:
        result = detect_candle_range_status(
            [_candle(datetime(2026, 7, 14, 8, tzinfo=UTC), close_offset=timedelta(minutes=30))],
            timeframe="1h",
            expected_from=datetime(2026, 7, 14, 8, tzinfo=UTC),
            expected_to=datetime(2026, 7, 14, 9, tzinfo=UTC),
        )

        self.assertEqual(result.status, CandleRangeStatus.GAP_DETECTED)
        self.assertEqual(result.wrong_duration_count, 1)

    def test_stale_complete_range_returns_stale_status(self) -> None:
        result = detect_candle_range_status(
            [_candle(datetime(2026, 7, 14, 8, tzinfo=UTC))],
            timeframe="1h",
            expected_from=datetime(2026, 7, 14, 8, tzinfo=UTC),
            expected_to=datetime(2026, 7, 14, 9, tzinfo=UTC),
            now=datetime(2026, 7, 14, 12, tzinfo=UTC),
            max_allowed_lag=timedelta(hours=1),
        )

        self.assertEqual(result.status, CandleRangeStatus.STALE)
