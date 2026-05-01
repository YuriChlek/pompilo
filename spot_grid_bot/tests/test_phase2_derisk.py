import unittest
from unittest.mock import Mock

from domain.models import Candle, InventorySnapshot, LiveOrder, MarketContext, OrderSide, RegimeSnapshot, RegimeType, StrategyState, SymbolRuntimeState
from domain.spot_grid_planner import SpotGridPlanner
from domain.strategy_config import DEFAULT_STRATEGY_CONFIG
from infrastructure.cost_basis_resolver import calculate_cost_basis_from_executions


def _build_candles() -> list[Candle]:
    candles = []
    price = 100.0
    for index in range(260):
        high = price + 1.0
        low = price - 1.0
        close = price - 0.2
        candles.append(Candle(timestamp=index, open=price, high=high, low=low, close=close, volume=10.0))
        price = close
    return candles


class Phase2DeRiskTests(unittest.TestCase):
    def test_downtrend_does_not_sell_inventory_at_loss(self):
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        planner.restore_symbol_runtime(SymbolRuntimeState(symbol="SOLUSDT", strategy_state=StrategyState(regime=RegimeType.DOWNTREND)))
        planner.detector.detect = Mock(return_value=RegimeSnapshot(RegimeType.DOWNTREND, 1.0, ["forced_downtrend"]))
        candles = _build_candles()
        inventory = InventorySnapshot(
            base_balance=2.0,
            quote_balance=500.0,
            reserved_quote=0.0,
            mark_price=candles[-1].close,
            cost_basis_price=150.0,
        )

        decision = planner.plan(MarketContext(symbol="SOLUSDT", candles=candles, inventory=inventory, live_orders=[]))

        self.assertEqual(decision.regime, RegimeType.DOWNTREND)
        self.assertEqual(decision.target_orders, [])

    def test_high_volatility_uses_explicit_derisk_policy_when_profitable(self):
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        planner.restore_symbol_runtime(SymbolRuntimeState(symbol="SOLUSDT", strategy_state=StrategyState(regime=RegimeType.HIGH_VOLATILITY)))
        planner.detector.detect = Mock(return_value=RegimeSnapshot(RegimeType.HIGH_VOLATILITY, 1.0, ["forced_volatility"]))
        candles = _build_candles()
        inventory = InventorySnapshot(
            base_balance=2.0,
            quote_balance=500.0,
            reserved_quote=0.0,
            mark_price=160.0,
            cost_basis_price=120.0,
        )

        decision = planner.plan(MarketContext(symbol="SOLUSDT", candles=candles, inventory=inventory, live_orders=[]))

        self.assertEqual(decision.regime, RegimeType.HIGH_VOLATILITY)
        self.assertTrue(all(order.side == OrderSide.SELL for order in decision.target_orders))
        self.assertTrue(all(order.price >= inventory.cost_basis_price for order in decision.target_orders))

    def test_downtrend_forces_rebuild_to_cancel_remaining_entry_buys(self):
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        planner.restore_symbol_runtime(SymbolRuntimeState(symbol="SOLUSDT", strategy_state=StrategyState(regime=RegimeType.DOWNTREND)))
        planner.detector.detect = Mock(return_value=RegimeSnapshot(RegimeType.DOWNTREND, 1.0, ["forced_downtrend"]))
        candles = _build_candles()
        inventory = InventorySnapshot(
            base_balance=0.0,
            quote_balance=500.0,
            reserved_quote=0.0,
            mark_price=candles[-1].close,
            cost_basis_price=None,
        )
        live_orders = [
            LiveOrder(
                order_id="1",
                symbol="SOLUSDT",
                side=OrderSide.BUY,
                price=95.0,
                size=1.0,
                filled_size=0.0,
                status="New",
                client_order_id="range-range-buy--0123456789abcdef",
            )
        ]

        decision = planner.plan(MarketContext(symbol="SOLUSDT", candles=candles, inventory=inventory, live_orders=live_orders))

        self.assertTrue(decision.rebuild_required)
        self.assertIn("protective_regime_cancel_entries", decision.reasons)
        self.assertTrue(all(order.side != OrderSide.BUY for order in decision.target_orders))

    def test_downtrend_does_not_treat_manual_buy_order_as_bot_managed_entry(self):
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        planner.restore_symbol_runtime(SymbolRuntimeState(symbol="SOLUSDT", strategy_state=StrategyState(regime=RegimeType.DOWNTREND)))
        planner.detector.detect = Mock(return_value=RegimeSnapshot(RegimeType.DOWNTREND, 1.0, ["forced_downtrend"]))
        candles = _build_candles()
        inventory = InventorySnapshot(
            base_balance=0.0,
            quote_balance=500.0,
            reserved_quote=0.0,
            mark_price=candles[-1].close,
            cost_basis_price=None,
        )
        live_orders = [
            LiveOrder(
                order_id="1",
                symbol="SOLUSDT",
                side=OrderSide.BUY,
                price=95.0,
                size=1.0,
                filled_size=0.0,
                status="New",
                client_order_id="manual-user-order",
            )
        ]

        decision = planner.plan(MarketContext(symbol="SOLUSDT", candles=candles, inventory=inventory, live_orders=live_orders))

        self.assertTrue(decision.rebuild_required)
        self.assertNotIn("protective_regime_cancel_entries", decision.reasons)

    def test_cost_basis_is_reduced_by_sells_and_keeps_remaining_inventory_average(self):
        executions = [
            {"side": "Buy", "execQty": "1", "execPrice": "100", "execFee": "0.1", "feeCurrency": "USDT"},
            {"side": "Buy", "execQty": "1", "execPrice": "120", "execFee": "0.1", "feeCurrency": "USDT"},
            {"side": "Sell", "execQty": "1", "execPrice": "150", "execFee": "0.0", "feeCurrency": "USDT"},
        ]

        cost_basis = calculate_cost_basis_from_executions(executions, base_balance=1.0)

        self.assertIsNotNone(cost_basis)
        self.assertAlmostEqual(cost_basis, 110.1, places=6)
