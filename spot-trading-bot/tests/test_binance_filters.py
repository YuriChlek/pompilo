from __future__ import annotations

import unittest
from decimal import Decimal

from infrastructure.bybit_spot import BybitSpotFilters, normalize_order_quantity, satisfies_min_notional


class BybitFilterTests(unittest.TestCase):
    def test_quantity_is_rounded_down_to_step_size(self) -> None:
        filters = BybitSpotFilters(
            symbol="ETHUSDT",
            min_qty=Decimal("0.001"),
            max_qty=Decimal("1000"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("5"),
            tick_size=Decimal("0.01"),
        )
        result = normalize_order_quantity(Decimal("0.123456"), filters)
        self.assertEqual(result, Decimal("0.123"))

    def test_quantity_below_min_qty_returns_zero(self) -> None:
        filters = BybitSpotFilters(
            symbol="ETHUSDT",
            min_qty=Decimal("0.01"),
            max_qty=Decimal("1000"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("5"),
            tick_size=Decimal("0.01"),
        )
        result = normalize_order_quantity(Decimal("0.0099"), filters)
        self.assertEqual(result, Decimal("0"))

    def test_min_notional_validation_uses_reference_price(self) -> None:
        filters = BybitSpotFilters(
            symbol="ETHUSDT",
            min_qty=Decimal("0.001"),
            max_qty=Decimal("1000"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("10"),
            tick_size=Decimal("0.01"),
        )
        self.assertFalse(satisfies_min_notional(Decimal("0.05"), Decimal("100"), filters))
        self.assertTrue(satisfies_min_notional(Decimal("0.2"), Decimal("100"), filters))
