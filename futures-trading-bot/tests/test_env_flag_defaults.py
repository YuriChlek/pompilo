import unittest
from decimal import Decimal
from unittest.mock import patch

from utils.config import _env_decimal, _env_flag, _env_int
from trading.application.bootstrap import build_runtime_strategy_config


class EnvFlagDefaultsTests(unittest.TestCase):
    def test_env_flag_uses_default_when_variable_is_missing(self):
        with patch("utils.config.os.getenv", return_value=None):
            self.assertTrue(_env_flag("ENABLE_BREAKEVEN_STOP_MANAGEMENT", True))
            self.assertFalse(_env_flag("ENABLE_BREAKEVEN_PARTIAL_CLOSE", False))

    def test_env_flag_parses_true_values(self):
        for raw_value in ("1", "true", "TRUE", "yes", "on"):
            with self.subTest(raw_value=raw_value):
                with patch("utils.config.os.getenv", return_value=raw_value):
                    self.assertTrue(_env_flag("FLAG", False))

    def test_env_flag_parses_false_values(self):
        for raw_value in ("0", "false", "FALSE", "no", "off"):
            with self.subTest(raw_value=raw_value):
                with patch("utils.config.os.getenv", return_value=raw_value):
                    self.assertFalse(_env_flag("FLAG", True))

    def test_env_flag_falls_back_to_default_for_invalid_value(self):
        with patch("utils.config.os.getenv", return_value="maybe"):
            self.assertTrue(_env_flag("FLAG", True))
            self.assertFalse(_env_flag("FLAG", False))

    def test_env_decimal_uses_default_when_variable_is_missing(self):
        with patch("utils.config.os.getenv", return_value=None):
            self.assertEqual(_env_decimal("BREAKEVEN_TRIGGER_R", "1.5"), Decimal("1.5"))

    def test_env_decimal_parses_decimal_value(self):
        with patch("utils.config.os.getenv", return_value="2.25"):
            self.assertEqual(_env_decimal("BREAKEVEN_TRIGGER_R", "1.5"), Decimal("2.25"))

    def test_env_decimal_falls_back_to_default_for_invalid_value(self):
        with patch("utils.config.os.getenv", return_value="abc"):
            self.assertEqual(_env_decimal("BREAKEVEN_TRIGGER_R", "1.5"), Decimal("1.5"))

    def test_env_int_uses_default_when_variable_is_missing(self):
        with patch("utils.config.os.getenv", return_value=None):
            self.assertEqual(_env_int("ANALYSIS_CANDLE_LIMIT", 1500), 1500)

    def test_env_int_parses_integer_value(self):
        with patch("utils.config.os.getenv", return_value="1800"):
            self.assertEqual(_env_int("ANALYSIS_CANDLE_LIMIT", 1500), 1800)

    def test_env_int_falls_back_to_default_for_invalid_value(self):
        with patch("utils.config.os.getenv", return_value="abc"):
            self.assertEqual(_env_int("ANALYSIS_CANDLE_LIMIT", 1500), 1500)

    def test_env_int_falls_back_to_default_when_below_minimum(self):
        with patch("utils.config.os.getenv", return_value="99"):
            self.assertEqual(_env_int("ANALYSIS_CANDLE_LIMIT", 1500, minimum=100), 1500)

    def test_runtime_strategy_config_overrides_domain_default_from_composition_root(self):
        with patch("utils.config.BREAKEVEN_TRIGGER_R", Decimal("2.25")):
            strategy_config = build_runtime_strategy_config()

        self.assertEqual(strategy_config.exit.breakeven_trigger_r, Decimal("2.25"))
