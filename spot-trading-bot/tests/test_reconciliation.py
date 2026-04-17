from __future__ import annotations

import unittest
from decimal import Decimal

from trading.infrastructure.binance_spot import base_asset_from_symbol, derive_avg_entry_price_from_trades


class ReconciliationTests(unittest.TestCase):
    def test_base_asset_is_derived_from_usdt_symbol(self) -> None:
        self.assertEqual(base_asset_from_symbol("ETHUSDT"), "ETH")

    def test_avg_entry_price_is_derived_from_remaining_inventory(self) -> None:
        trades = [
            {"id": 1, "time": 1, "isBuyer": True, "qty": "1", "price": "100"},
            {"id": 2, "time": 2, "isBuyer": True, "qty": "1", "price": "120"},
            {"id": 3, "time": 3, "isBuyer": False, "qty": "1", "price": "140"},
        ]
        result = derive_avg_entry_price_from_trades("ETHUSDT", Decimal("1"), trades)
        self.assertEqual(result, Decimal("120"))

    def test_avg_entry_returns_zero_when_position_is_flat(self) -> None:
        trades = [
            {"id": 1, "time": 1, "isBuyer": True, "qty": "1", "price": "100"},
            {"id": 2, "time": 2, "isBuyer": False, "qty": "1", "price": "110"},
        ]
        result = derive_avg_entry_price_from_trades("ETHUSDT", Decimal("0"), trades)
        self.assertEqual(result, Decimal("0"))
