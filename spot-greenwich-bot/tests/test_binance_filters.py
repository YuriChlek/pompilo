from __future__ import annotations

import unittest
from decimal import Decimal
from unittest.mock import patch

from infrastructure.bybit_spot import BybitSpotClient, BybitSpotFilters, normalize_order_quantity, satisfies_min_notional


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

    def test_symbol_filters_cache_expires_after_ttl(self) -> None:
        calls: list[str] = []

        class _RawClient:
            def get_instruments_info(self, *, category: str, symbol: str):
                calls.append(symbol)
                return {
                    "result": {
                        "list": [
                            {
                                "priceFilter": {"tickSize": "0.01"},
                                "lotSizeFilter": {
                                    "basePrecision": "0.001",
                                    "minOrderQty": "0.001",
                                    "maxLimitOrderQty": "1000",
                                    "minOrderAmt": "5",
                                },
                            }
                        ]
                    }
                }

        client = BybitSpotClient.__new__(BybitSpotClient)
        client.client = _RawClient()
        client._symbol_filters_cache = {}

        with patch("infrastructure.bybit_spot.time.monotonic", side_effect=[0, 10, 3601]):
            first = client.get_symbol_filters("ethusdt")
            second = client.get_symbol_filters("ETHUSDT")
            third = client.get_symbol_filters("ETHUSDT")

        self.assertEqual(first, second)
        self.assertEqual(third.symbol, "ETHUSDT")
        self.assertEqual(calls, ["ETHUSDT", "ETHUSDT"])
