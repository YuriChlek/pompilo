import unittest
from unittest.mock import patch

from application.bootstrap import build_runtime_strategy_config


class RuntimeConfigOverrideTests(unittest.TestCase):
    def test_runtime_strategy_config_uses_env_backed_max_new_orders_per_cycle(self):
        with patch("application.bootstrap.MAX_NEW_ORDERS_PER_CYCLE", 7):
            runtime_config = build_runtime_strategy_config()

        self.assertEqual(runtime_config.execution.max_new_orders_per_cycle, 7)
