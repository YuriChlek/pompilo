import unittest
from dataclasses import replace
from unittest.mock import patch

from application.bootstrap import build_runtime_strategy_config, validate_strategy_config
from domain.strategy_config import DEFAULT_STRATEGY_CONFIG


class RuntimeConfigOverrideTests(unittest.TestCase):
    def test_runtime_strategy_config_uses_env_backed_max_new_orders_per_cycle(self):
        with patch("application.bootstrap.MAX_NEW_ORDERS_PER_CYCLE", 7):
            runtime_config = build_runtime_strategy_config()

        self.assertEqual(runtime_config.execution.max_new_orders_per_cycle, 7)

    def test_validate_strategy_config_warns_when_max_symbol_cap_exceeds_inventory_cap(self):
        config = replace(
            DEFAULT_STRATEGY_CONFIG,
            risk=replace(
                DEFAULT_STRATEGY_CONFIG.risk,
                max_inventory_notional=300.0,
                max_symbol_notional_cap=400.0,
            ),
        )

        with patch("application.bootstrap.logger.warning") as warning_log:
            validate_strategy_config(config)

        warning_log.assert_called_once()

    def test_validate_strategy_config_raises_when_min_entry_meets_or_exceeds_symbol_cap(self):
        config = replace(
            DEFAULT_STRATEGY_CONFIG,
            risk=replace(
                DEFAULT_STRATEGY_CONFIG.risk,
                min_symbol_entry_notional=500.0,
                max_symbol_notional_cap=400.0,
            ),
        )

        with self.assertRaises(ValueError):
            validate_strategy_config(config)
