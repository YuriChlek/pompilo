from __future__ import annotations

from datetime import timedelta
import os
import unittest
from unittest.mock import patch

from market_data_service.config.provider_config import get_binance_spot_provider_config


class ProviderConfigTests(unittest.TestCase):
    def test_binance_provider_config_reads_hardening_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = get_binance_spot_provider_config()

        self.assertEqual(config.max_concurrent_requests, 8)
        self.assertEqual(config.circuit_breaker_failure_threshold, 5)
        self.assertEqual(config.circuit_breaker_recovery_timeout_seconds, 60.0)
        self.assertEqual(config.safety_delay_by_timeframe["1h"], timedelta(seconds=30))

    def test_binance_provider_config_reads_hardening_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BINANCE_MAX_CONCURRENT_REQUESTS": "4",
                "BINANCE_CIRCUIT_BREAKER_FAILURE_THRESHOLD": "3",
                "BINANCE_CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS": "15.5",
            },
            clear=True,
        ):
            config = get_binance_spot_provider_config()

        self.assertEqual(config.max_concurrent_requests, 4)
        self.assertEqual(config.circuit_breaker_failure_threshold, 3)
        self.assertEqual(config.circuit_breaker_recovery_timeout_seconds, 15.5)

    def test_binance_provider_config_rejects_invalid_concurrency(self) -> None:
        with patch.dict(os.environ, {"BINANCE_MAX_CONCURRENT_REQUESTS": "0"}, clear=True):
            with self.assertRaises(ValueError):
                get_binance_spot_provider_config()
