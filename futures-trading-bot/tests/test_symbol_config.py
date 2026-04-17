import unittest

from tests.support import install_common_test_stubs


install_common_test_stubs()

from utils.config import POSITION_ROUNDING_RULES, SYMBOLS_ROUNDING, TRADING_SYMBOLS


class SymbolConfigTests(unittest.TestCase):
    def test_all_trading_symbols_have_price_rounding(self):
        missing = sorted(symbol for symbol in TRADING_SYMBOLS if symbol not in SYMBOLS_ROUNDING)
        self.assertEqual(missing, [])

    def test_price_rounding_does_not_contain_unused_symbols(self):
        unused_symbols = sorted(symbol for symbol in SYMBOLS_ROUNDING if symbol not in TRADING_SYMBOLS)
        self.assertEqual(unused_symbols, [])

    def test_zecusdt_has_consistent_rounding_rules(self):
        self.assertIn("ZECUSDT", TRADING_SYMBOLS)
        self.assertIn("ZECUSDT", POSITION_ROUNDING_RULES)
        self.assertEqual(SYMBOLS_ROUNDING["ZECUSDT"], 2)
