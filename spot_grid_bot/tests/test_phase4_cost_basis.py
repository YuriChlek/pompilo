import unittest
from unittest.mock import Mock, patch

from domain.cost_basis import minimum_exit_price, minimum_take_profit_price
from domain.models import Candle, InventorySnapshot, MarketContext, OrderSide, RegimeSnapshot, RegimeType
from domain.spot_grid_planner import SpotGridPlanner
from domain.strategy_config import DEFAULT_STRATEGY_CONFIG


def _candles(price: float = 100.0) -> list[Candle]:
    candles: list[Candle] = []
    current = price
    for index in range(260):
        candles.append(
            Candle(
                timestamp=index,
                open=current,
                high=current + 1.0,
                low=current - 1.0,
                close=current,
                volume=10.0,
            )
        )
    return candles


class Phase4CostBasisTests(unittest.TestCase):
    def test_minimum_exit_price_requires_one_percent_markup_over_cost_basis(self):
        inventory = InventorySnapshot(
            base_balance=1.0,
            quote_balance=1000.0,
            reserved_quote=0.0,
            mark_price=100.0,
            cost_basis_price=100.0,
        )

        floor = minimum_exit_price(inventory, DEFAULT_STRATEGY_CONFIG)

        self.assertEqual(floor, 101.0)

    def test_range_sell_targets_are_rebased_above_cost_basis_floor(self):
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        planner.detector.detect = Mock(return_value=RegimeSnapshot(RegimeType.RANGE, 1.0, ["fixed_range"]))
        candles = _candles(100.0)
        inventory = InventorySnapshot(
            base_balance=3.0,
            quote_balance=1000.0,
            reserved_quote=0.0,
            mark_price=100.0,
            cost_basis_price=120.0,
        )

        decision = planner.plan(MarketContext(symbol="SOLUSDT", candles=candles, inventory=inventory, live_orders=[]))
        sell_orders = [order for order in decision.target_orders if order.side == OrderSide.SELL]

        self.assertTrue(sell_orders)
        floor = minimum_take_profit_price(inventory, decision.indicators, DEFAULT_STRATEGY_CONFIG)
        self.assertIsNotNone(floor)
        self.assertTrue(all(order.price >= floor for order in sell_orders))

    def test_uptrend_sell_targets_are_rebased_from_cost_basis_not_buy_ladder_only(self):
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        planner.detector.detect = Mock(return_value=RegimeSnapshot(RegimeType.UPTREND, 1.0, ["fixed_uptrend"]))
        candles = _candles(130.0)
        inventory = InventorySnapshot(
            base_balance=2.0,
            quote_balance=1000.0,
            reserved_quote=0.0,
            mark_price=130.0,
            cost_basis_price=125.0,
        )

        decision = planner.plan(MarketContext(symbol="SOLUSDT", candles=candles, inventory=inventory, live_orders=[]))
        sell_orders = [order for order in decision.target_orders if order.side == OrderSide.SELL]

        self.assertTrue(sell_orders)
        floor = minimum_take_profit_price(inventory, decision.indicators, DEFAULT_STRATEGY_CONFIG)
        self.assertIsNotNone(floor)
        self.assertGreaterEqual(min(order.price for order in sell_orders), floor)

    def test_planner_does_not_build_sell_targets_when_cost_basis_is_unknown(self):
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        planner.detector.detect = Mock(return_value=RegimeSnapshot(RegimeType.RANGE, 1.0, ["fixed_range"]))
        candles = _candles(100.0)
        inventory = InventorySnapshot(
            base_balance=3.0,
            quote_balance=1000.0,
            reserved_quote=0.0,
            mark_price=100.0,
            cost_basis_price=None,
        )

        decision = planner.plan(MarketContext(symbol="SOLUSDT", candles=candles, inventory=inventory, live_orders=[]))
        sell_orders = [order for order in decision.target_orders if order.side == OrderSide.SELL]

        self.assertEqual(sell_orders, [])

    def test_planner_logs_when_no_loss_floor_blocks_sell_levels(self):
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        planner.detector.detect = Mock(return_value=RegimeSnapshot(RegimeType.RANGE, 1.0, ["fixed_range"]))
        candles = _candles(100.0)
        inventory = InventorySnapshot(
            base_balance=3.0,
            quote_balance=1000.0,
            reserved_quote=0.0,
            mark_price=100.0,
            cost_basis_price=None,
        )

        with patch("domain.target_order_builder.logger.warning") as warning_mock:
            planner.plan(MarketContext(symbol="SOLUSDT", candles=candles, inventory=inventory, live_orders=[]))

        warning_mock.assert_called_once()
        args = warning_mock.call_args[0]
        self.assertEqual(args[0], "planner_no_loss_sell_blocked symbol=%s blocked_levels=%s min_exit_price=%s mark_price=%s cost_basis=%s")
        self.assertEqual(args[1], "SOLUSDT")
        self.assertGreater(args[2], 0)
        self.assertIsNone(args[3])
        self.assertEqual(args[4], 100.0)
        self.assertIsNone(args[5])
