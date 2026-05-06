from __future__ import annotations

from decimal import Decimal
import unittest

from trading.application.bootstrap import build_strategy_config
from utils.config import APP_CONFIG, build_app_config


class ConfigTests(unittest.TestCase):
    def test_build_app_config_exposes_typed_orderflow_thresholds(self) -> None:
        config = build_app_config()

        self.assertIsInstance(config.orderflow.max_wall_distance_bps, Decimal)
        self.assertIsInstance(config.orderflow.min_wall_notional_usdt, Decimal)
        self.assertIsInstance(config.orderflow.max_spoof_score, Decimal)
        self.assertIsInstance(config.orderflow.take_profit_r_multiple, Decimal)
        self.assertIsInstance(config.orderflow.stop_loss_size, Decimal)
        self.assertIsInstance(config.orderflow.symbols, tuple)

    def test_app_config_preserves_existing_defaults(self) -> None:
        self.assertEqual(APP_CONFIG.orderflow.analysis_reference_exchange, "bybit")
        self.assertEqual(APP_CONFIG.orderflow.symbols[:2], ("BTCUSDT", "ETHUSDT"))
        self.assertEqual(APP_CONFIG.telegram.app_timezone, "Europe/Kyiv")
        self.assertEqual(APP_CONFIG.orderflow.stop_loss_size, Decimal("0.5"))

    def test_build_strategy_config_uses_app_config_values(self) -> None:
        strategy_config = build_strategy_config()

        self.assertEqual(
            strategy_config.signal_generation.analysis_reference_exchange,
            APP_CONFIG.orderflow.analysis_reference_exchange,
        )
        self.assertEqual(
            strategy_config.signal_generation.cross_confirmation_bps,
            APP_CONFIG.orderflow.cross_confirmation_bps,
        )
        self.assertEqual(
            strategy_config.signal_generation.stop_loss_size,
            APP_CONFIG.orderflow.stop_loss_size,
        )
        self.assertEqual(
            strategy_config.liquidity_detection.min_wall_notional_usdt,
            APP_CONFIG.orderflow.min_wall_notional_usdt,
        )
