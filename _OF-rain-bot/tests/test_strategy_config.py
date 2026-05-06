from __future__ import annotations

from decimal import Decimal
import unittest

from trading.domain.strategy_config import DEFAULT_STRATEGY_CONFIG


class StrategyConfigTests(unittest.TestCase):
    def test_default_strategy_config_uses_decimal_thresholds(self) -> None:
        self.assertIsInstance(DEFAULT_STRATEGY_CONFIG.liquidity_detection.max_wall_distance_bps, Decimal)
        self.assertIsInstance(DEFAULT_STRATEGY_CONFIG.liquidity_detection.max_wall_size_drop_pct, Decimal)
        self.assertIsInstance(DEFAULT_STRATEGY_CONFIG.spoof_filter.max_spoof_score, Decimal)
        self.assertIsInstance(DEFAULT_STRATEGY_CONFIG.signal_generation.cross_confirmation_bps, Decimal)
        self.assertIsInstance(DEFAULT_STRATEGY_CONFIG.signal_generation.take_profit_r_multiple, Decimal)

    def test_default_strategy_config_contains_expected_defaults(self) -> None:
        self.assertEqual(DEFAULT_STRATEGY_CONFIG.signal_generation.analysis_reference_exchange, "bybit")
        self.assertEqual(DEFAULT_STRATEGY_CONFIG.signal_generation.max_spread_ticks, 3)
        self.assertEqual(DEFAULT_STRATEGY_CONFIG.signal_generation.min_test_count, 2)
        self.assertEqual(DEFAULT_STRATEGY_CONFIG.liquidity_detection.max_chase_ticks, 2)
