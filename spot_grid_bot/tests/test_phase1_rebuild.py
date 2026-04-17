import unittest
from unittest.mock import Mock

from domain.models import Candle, InventorySnapshot, LiveOrder, MarketContext, OrderSide, RegimeSnapshot, RegimeType
from domain.spot_grid_planner import SpotGridPlanner
from domain.strategy_config import DEFAULT_STRATEGY_CONFIG


def _build_candles() -> list[Candle]:
    return [
        Candle(
            timestamp=index,
            open=100.0 + ((index % 6) - 3) * 0.05,
            high=100.8 + ((index % 4) * 0.03),
            low=99.2 - ((index % 4) * 0.03),
            close=100.0 + ((index % 5) - 2) * 0.06,
            volume=10.0,
        )
        for index in range(260)
    ]


class Phase1RebuildTests(unittest.TestCase):
    def test_second_pass_skips_rebuild_when_live_orders_match_target(self):
        candles = _build_candles()
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        planner.detector.detect = Mock(return_value=RegimeSnapshot(RegimeType.RANGE, 1.0, ["test_fixed_range"]))
        inventory = InventorySnapshot(
            base_balance=0.0,
            quote_balance=1_000.0,
            reserved_quote=0.0,
            mark_price=candles[-1].close,
        )

        first_decision = planner.plan(MarketContext(symbol="SOLUSDT", candles=candles, inventory=inventory, live_orders=[]))
        live_orders = [
            LiveOrder(
                order_id=str(index),
                symbol=order.symbol,
                side=order.side,
                price=order.price,
                size=order.size,
                filled_size=0.0,
                status="New",
                client_order_id=order.client_order_id,
            )
            for index, order in enumerate(first_decision.target_orders, start=1)
        ]

        second_decision = planner.plan(MarketContext(symbol="SOLUSDT", candles=candles, inventory=inventory, live_orders=live_orders))

        self.assertFalse(second_decision.rebuild_required)
        self.assertEqual(second_decision.target_order_diff_count, 0)

    def test_rebuild_required_when_live_order_diff_exists(self):
        candles = _build_candles()
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        planner.detector.detect = Mock(return_value=RegimeSnapshot(RegimeType.RANGE, 1.0, ["test_fixed_range"]))
        inventory = InventorySnapshot(
            base_balance=0.0,
            quote_balance=1_000.0,
            reserved_quote=0.0,
            mark_price=candles[-1].close,
        )
        live_orders = [
            LiveOrder(
                order_id="1",
                symbol="SOLUSDT",
                side=OrderSide.BUY,
                price=1.0,
                size=0.1,
                filled_size=0.0,
                status="New",
                client_order_id="wrong-order",
            )
        ]

        decision = planner.plan(MarketContext(symbol="SOLUSDT", candles=candles, inventory=inventory, live_orders=live_orders))

        self.assertTrue(decision.rebuild_required)
        self.assertGreater(decision.target_order_diff_count, 0)
