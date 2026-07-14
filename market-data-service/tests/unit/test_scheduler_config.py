from __future__ import annotations

from datetime import timedelta
import os
import unittest
from unittest.mock import patch

from market_data_service.config.scheduler_config import get_scheduler_config
from market_data_service.domain.enums import MarketDataSource


class SchedulerConfigTests(unittest.TestCase):
    def test_scheduler_config_uses_production_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = get_scheduler_config()

        self.assertEqual(config.source, MarketDataSource.BINANCE_SPOT)
        self.assertEqual(config.provider_symbols, ("ETHUSDT",))
        self.assertEqual(config.timeframes, ("1h", "4h", "1d"))
        self.assertEqual(config.safety_delay_by_timeframe["1h"], timedelta(seconds=30))
        self.assertEqual(config.safety_delay_by_timeframe["4h"], timedelta(seconds=45))
        self.assertEqual(config.safety_delay_by_timeframe["1d"], timedelta(seconds=90))
        self.assertEqual(config.jitter_seconds, 30)
        self.assertEqual(config.poll_interval_seconds, 30.0)

    def test_scheduler_config_reads_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MARKET_DATA_PROVIDER_SYMBOLS": "ethusdt, btcusdt",
                "MARKET_DATA_TIMEFRAMES": "1H,4H",
                "MARKET_DATA_1H_SAFETY_DELAY_SECONDS": "10",
                "MARKET_DATA_4H_SAFETY_DELAY_SECONDS": "20",
                "MARKET_DATA_1D_SAFETY_DELAY_SECONDS": "30",
                "MARKET_DATA_SCHEDULER_JITTER_SECONDS": "7",
                "MARKET_DATA_SCHEDULER_POLL_INTERVAL_SECONDS": "5.5",
            },
            clear=True,
        ):
            config = get_scheduler_config()

        self.assertEqual(config.provider_symbols, ("ETHUSDT", "BTCUSDT"))
        self.assertEqual(config.timeframes, ("1h", "4h"))
        self.assertEqual(config.safety_delay_by_timeframe["1h"], timedelta(seconds=10))
        self.assertEqual(config.jitter_seconds, 7)
        self.assertEqual(config.poll_interval_seconds, 5.5)

    def test_scheduler_config_rejects_unsupported_timeframe(self) -> None:
        with patch.dict(os.environ, {"MARKET_DATA_TIMEFRAMES": "15m"}, clear=True):
            with self.assertRaises(ValueError):
                get_scheduler_config()

    def test_scheduler_config_rejects_negative_jitter(self) -> None:
        with patch.dict(os.environ, {"MARKET_DATA_SCHEDULER_JITTER_SECONDS": "-1"}, clear=True):
            with self.assertRaises(ValueError):
                get_scheduler_config()
