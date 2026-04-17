import unittest
from decimal import Decimal
from unittest.mock import patch

from domain.models import InventorySnapshot, LiveOrder, OrderSide, TargetOrder
from domain.strategy_config import DEFAULT_STRATEGY_CONFIG
from infrastructure.execution_gateway import (
    BybitSpotExecutionService,
    BybitSpotExchange,
    _build_exchange_order_link_id,
    _is_bot_managed_order_link_id,
)


class _FakeExchange:
    def __init__(self, final_orders):
        self.final_orders = final_orders

    def get_balances(self, symbol: str):
        return InventorySnapshot(base_balance=0.0, quote_balance=1000.0, reserved_quote=0.0, mark_price=100.0)

    def get_open_orders(self, symbol: str):
        return []

    def normalize_target_order(self, order: TargetOrder):
        return order

    def sync_orders(self, symbol: str, target_orders: list[TargetOrder]):
        return list(self.final_orders)


class _NormalizeToNoneExchange(_FakeExchange):
    def normalize_target_order(self, order: TargetOrder):
        return None


class _DuplicateRecoveryExchange(BybitSpotExchange):
    def __init__(self):
        self._lookup_count = 0

    def _find_existing_order_id_by_link_id(self, symbol: str, order_link_id: str):
        self._lookup_count += 1
        return "existing-order-id" if self._lookup_count >= 2 else None


class _ValidateOrderExchange(BybitSpotExchange):
    def __init__(self):
        self.strategy_config = DEFAULT_STRATEGY_CONFIG

    def get_instrument_filters(self, symbol: str):
        from infrastructure.bybit_spot_types import SpotInstrumentFilters

        return SpotInstrumentFilters(
            tick_size=Decimal("0.01"),
            qty_step=Decimal("0.0001"),
            min_order_qty=Decimal("0.0001"),
            min_order_amt=Decimal("5"),
            max_limit_order_qty=Decimal("0"),
            max_market_order_qty=Decimal("0"),
        )


class ExecutionSyncTests(unittest.IsolatedAsyncioTestCase):
    def test_exchange_order_link_id_is_unique_per_create_request(self):
        order = TargetOrder(
            client_order_id="range-range_buy-0",
            symbol="SOLUSDT",
            side=OrderSide.BUY,
            price=90.0,
            size=0.1,
        )

        first = _build_exchange_order_link_id(order)
        second = _build_exchange_order_link_id(order)

        self.assertNotEqual(first, second)
        self.assertTrue(_is_bot_managed_order_link_id(first))
        self.assertTrue(_is_bot_managed_order_link_id(second))

    async def test_sync_returns_false_when_exchange_did_not_reach_expected_live_set(self):
        service = BybitSpotExecutionService(DEFAULT_STRATEGY_CONFIG)
        service.exchange = _FakeExchange(final_orders=[])

        result = await service.sync_orders(
            "SOLUSDT",
            [TargetOrder(client_order_id="range-range_buy-0", symbol="SOLUSDT", side=OrderSide.BUY, price=90.0, size=0.1)],
        )

        self.assertFalse(result)

    async def test_sync_returns_true_when_exchange_live_set_matches_expected_orders(self):
        order = LiveOrder(
            order_id="1",
            symbol="SOLUSDT",
            side=OrderSide.BUY,
            price=90.0,
            size=0.1,
            filled_size=0.0,
            status="New",
            client_order_id="range-range-buy--0123456789abcdef",
        )
        service = BybitSpotExecutionService(DEFAULT_STRATEGY_CONFIG)
        service.exchange = _FakeExchange(final_orders=[order])

        result = await service.sync_orders(
            "SOLUSDT",
            [TargetOrder(client_order_id="range-range_buy-0", symbol="SOLUSDT", side=OrderSide.BUY, price=90.0, size=0.1)],
        )

        self.assertTrue(result)

    async def test_sync_returns_false_when_venue_normalization_filters_every_guarded_order(self):
        service = BybitSpotExecutionService(DEFAULT_STRATEGY_CONFIG)
        service.exchange = _NormalizeToNoneExchange(final_orders=[])

        result = await service.sync_orders(
            "SOLUSDT",
            [TargetOrder(client_order_id="range-range_buy-0", symbol="SOLUSDT", side=OrderSide.BUY, price=90.0, size=0.1)],
        )

        self.assertFalse(result)

    def test_duplicate_recovery_polling_returns_existing_order_after_short_delay(self):
        exchange = _DuplicateRecoveryExchange()
        with patch("infrastructure.execution_gateway.time.sleep", return_value=None):
            order_id = exchange._wait_for_existing_order_id_by_link_id("SOLUSDT", "test-link-id")

        self.assertEqual(order_id, "existing-order-id")

    def test_validate_order_uses_two_usdt_notional_floor_for_bybit_spot(self):
        exchange = _ValidateOrderExchange()

        self.assertTrue(exchange._validate_order("ETHUSDT", Decimal("0.0030"), Decimal("2050")))
        self.assertFalse(exchange._validate_order("ETHUSDT", Decimal("0.0005"), Decimal("2050")))

    def test_normalize_target_order_drops_orders_below_two_usdt_notional(self):
        exchange = _ValidateOrderExchange()
        order = TargetOrder(
            client_order_id="tiny-buy",
            symbol="ETHUSDT",
            side=OrderSide.BUY,
            price=2050.0,
            size=0.0005,
        )

        self.assertIsNone(exchange.normalize_target_order(order))
