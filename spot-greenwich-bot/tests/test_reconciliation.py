from __future__ import annotations

import unittest
from decimal import Decimal

from infrastructure.bybit_spot import derive_avg_entry_price_from_trades, split_symbol


class ReconciliationTests(unittest.TestCase):
    def test_symbol_is_split_into_base_and_quote_assets(self) -> None:
        self.assertEqual(split_symbol("ETHUSDT"), ("ETH", "USDT"))

    def test_avg_entry_price_is_derived_from_remaining_inventory(self) -> None:
        trades = [
            {"execId": "1", "execTime": 1, "side": "Buy", "execQty": "1", "execPrice": "100"},
            {"execId": "2", "execTime": 2, "side": "Buy", "execQty": "1", "execPrice": "120"},
            {"execId": "3", "execTime": 3, "side": "Sell", "execQty": "1", "execPrice": "140"},
        ]
        result = derive_avg_entry_price_from_trades("ETHUSDT", Decimal("1"), trades)
        self.assertEqual(result, Decimal("120"))

    def test_avg_entry_returns_zero_when_position_is_flat(self) -> None:
        trades = [
            {"execId": "1", "execTime": 1, "side": "Buy", "execQty": "1", "execPrice": "100"},
            {"execId": "2", "execTime": 2, "side": "Sell", "execQty": "1", "execPrice": "110"},
        ]
        result = derive_avg_entry_price_from_trades("ETHUSDT", Decimal("0"), trades)
        self.assertEqual(result, Decimal("0"))
