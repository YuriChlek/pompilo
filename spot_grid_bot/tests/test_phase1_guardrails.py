import unittest

from domain.models import InventorySnapshot, LiveOrder, OrderSide, TargetOrder
from domain.strategy_config import DEFAULT_STRATEGY_CONFIG
from infrastructure.execution_gateway import _apply_execution_guardrails


class Phase1GuardrailsTests(unittest.TestCase):
    def test_guardrails_drop_sell_orders_below_cost_basis(self):
        inventory = InventorySnapshot(
            base_balance=1.0,
            quote_balance=1_000.0,
            reserved_quote=0.0,
            mark_price=100.0,
            cost_basis_price=120.0,
        )
        current_orders = [
            LiveOrder(
                order_id="1",
                symbol="SOLUSDT",
                side=OrderSide.BUY,
                price=90.0,
                size=0.1,
                filled_size=0.0,
                status="New",
                client_order_id="existing-buy",
            )
        ]
        target_orders = [
            TargetOrder(client_order_id="existing-buy", symbol="SOLUSDT", side=OrderSide.BUY, price=90.0, size=0.1),
            TargetOrder(client_order_id="loss-sell", symbol="SOLUSDT", side=OrderSide.SELL, price=110.0, size=0.1),
            TargetOrder(client_order_id="valid-sell", symbol="SOLUSDT", side=OrderSide.SELL, price=121.0, size=0.1),
        ]

        guarded_orders = _apply_execution_guardrails("SOLUSDT", current_orders, target_orders, inventory, DEFAULT_STRATEGY_CONFIG)

        self.assertEqual(
            [(order.client_order_id, order.side.value, order.price) for order in guarded_orders],
            [("existing-buy", "BUY", 90.0)],
        )

    def test_guardrails_limit_new_orders_per_cycle(self):
        inventory = InventorySnapshot(
            base_balance=0.0,
            quote_balance=10_000.0,
            reserved_quote=0.0,
            mark_price=100.0,
        )
        current_orders: list[LiveOrder] = []
        target_orders = [
            TargetOrder(client_order_id=f"buy-{index}", symbol="SOLUSDT", side=OrderSide.BUY, price=100.0 - index, size=0.1)
            for index in range(5)
        ]

        guarded_orders = _apply_execution_guardrails("SOLUSDT", current_orders, target_orders, inventory, DEFAULT_TRADING_CONFIG_WITH_LIMITS())

        self.assertEqual(len(guarded_orders), 2)

    def test_guardrails_shift_whole_buy_ladder_when_top_level_is_too_close_to_live_price(self):
        inventory = InventorySnapshot(
            base_balance=1.0,
            quote_balance=10_000.0,
            reserved_quote=0.0,
            mark_price=100.0,
            cost_basis_price=95.0,
        )
        current_orders: list[LiveOrder] = []
        target_orders = [
            TargetOrder(client_order_id="too-close-buy", symbol="SOLUSDT", side=OrderSide.BUY, price=99.8, size=0.1),
            TargetOrder(client_order_id="valid-buy", symbol="SOLUSDT", side=OrderSide.BUY, price=98.0, size=0.1),
            TargetOrder(client_order_id="too-low-sell", symbol="SOLUSDT", side=OrderSide.SELL, price=100.05, size=0.1),
            TargetOrder(client_order_id="valid-sell", symbol="SOLUSDT", side=OrderSide.SELL, price=102.0, size=0.1),
        ]

        guarded_orders = _apply_execution_guardrails("SOLUSDT", current_orders, target_orders, inventory, DEFAULT_STRATEGY_CONFIG)

        self.assertEqual(
            [(order.client_order_id, order.side.value, order.price) for order in guarded_orders],
            [("valid-buy", "BUY", 97.9), ("too-close-buy", "BUY", 99.7), ("valid-sell", "SELL", 102.0)],
        )


def DEFAULT_TRADING_CONFIG_WITH_LIMITS():
    from dataclasses import replace

    return replace(
        DEFAULT_STRATEGY_CONFIG,
        execution=replace(
            DEFAULT_STRATEGY_CONFIG.execution,
            max_new_orders_per_cycle=2,
            max_total_open_orders=3,
        ),
    )
