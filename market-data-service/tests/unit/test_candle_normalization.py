from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
import unittest

from market_data_service.domain.candle_normalizer import normalize_closed_candle
from market_data_service.domain.candle_validation import (
    CandleValidationError,
    validate_candle_sequence,
    validate_closed_candle,
)
from market_data_service.domain.enums import MarketDataSource


def _payload(open_time: datetime) -> dict[str, object]:
    return {
        "open_time": open_time,
        "open": "100.1",
        "high": "110.2",
        "low": "99.9",
        "close": "105.5",
        "volume": "10.0",
        "quote_volume": "1055.0",
        "taker_buy_base_volume": "6.5",
        "taker_buy_quote_volume": "685.75",
        "trades_count": 42,
    }


class CandleNormalizationTests(unittest.TestCase):
    def test_normalizer_coerces_market_values_to_decimal(self) -> None:
        candle = normalize_closed_candle(
            _payload(datetime(2026, 7, 14, 8, tzinfo=UTC)),
            source=MarketDataSource.BINANCE_SPOT,
            canonical_symbol="ETH/USDT",
            provider_symbol="ETHUSDT",
            timeframe="1h",
        )

        self.assertEqual(candle.open, Decimal("100.1"))
        self.assertEqual(candle.volume, Decimal("10.0"))
        self.assertEqual(candle.taker_sell_base_volume, Decimal("3.5"))
        self.assertEqual(candle.taker_sell_quote_volume, Decimal("369.25"))
        self.assertEqual(candle.close_time, datetime(2026, 7, 14, 9, tzinfo=UTC))

    def test_normalizer_rejects_float_market_values(self) -> None:
        payload = _payload(datetime(2026, 7, 14, 8, tzinfo=UTC))
        payload["open"] = 100.1

        with self.assertRaises(TypeError):
            normalize_closed_candle(
                payload,
                source=MarketDataSource.BINANCE_SPOT,
                canonical_symbol="ETH/USDT",
                provider_symbol="ETHUSDT",
                timeframe="1h",
            )

    def test_closed_candle_validator_accepts_closed_candle_after_safety_delay(self) -> None:
        candle = normalize_closed_candle(
            _payload(datetime(2026, 7, 14, 8, tzinfo=UTC)),
            source=MarketDataSource.BINANCE_SPOT,
            canonical_symbol="ETH/USDT",
            provider_symbol="ETHUSDT",
            timeframe="1h",
        )

        validate_closed_candle(
            candle,
            now=datetime(2026, 7, 14, 9, 1, tzinfo=UTC),
            safety_delay=timedelta(seconds=30),
        )

    def test_closed_candle_validator_blocks_not_closed_candle(self) -> None:
        candle = normalize_closed_candle(
            _payload(datetime(2026, 7, 14, 8, tzinfo=UTC)),
            source=MarketDataSource.BINANCE_SPOT,
            canonical_symbol="ETH/USDT",
            provider_symbol="ETHUSDT",
            timeframe="1h",
        )

        with self.assertRaisesRegex(CandleValidationError, "not closed"):
            validate_closed_candle(
                candle,
                now=datetime(2026, 7, 14, 9, 0, 20, tzinfo=UTC),
                safety_delay=timedelta(seconds=30),
            )

    def test_sequence_validator_requires_monotonic_continuous_ordering(self) -> None:
        first = normalize_closed_candle(
            _payload(datetime(2026, 7, 14, 8, tzinfo=UTC)),
            source=MarketDataSource.BINANCE_SPOT,
            canonical_symbol="ETH/USDT",
            provider_symbol="ETHUSDT",
            timeframe="1h",
        )
        second = normalize_closed_candle(
            _payload(datetime(2026, 7, 14, 9, tzinfo=UTC)),
            source=MarketDataSource.BINANCE_SPOT,
            canonical_symbol="ETH/USDT",
            provider_symbol="ETHUSDT",
            timeframe="1h",
        )

        validate_candle_sequence((first, second))

        with self.assertRaisesRegex(CandleValidationError, "continuous"):
            validate_candle_sequence((second, first))
