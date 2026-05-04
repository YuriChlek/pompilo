import unittest
from unittest.mock import Mock

from domain.order_diff import target_orders_diff_count
from domain.models import Candle, DeRiskMode, InventorySnapshot, LiveOrder, MarketContext, OrderSide, RegimeSnapshot, RegimeType, RiskDecision, StrategyState, TargetOrder
from domain.rebuild_policy import should_rebuild
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
    def test_should_rebuild_ignores_single_diff_when_below_threshold(self):
        risk = RiskDecision(
            can_trade=True,
            pause_entries=False,
            force_risk_off=False,
            cancel_entries=False,
            allow_exit_only=False,
            de_risk_mode=DeRiskMode.NONE,
        )
        rebuild_required, reasons = should_rebuild(
            StrategyState(regime=RegimeType.RANGE, bars_in_state=3, last_rebuild_price=100.0),
            price=100.1,
            atr14=0.5,
            live_orders=[LiveOrder("1", "SOLUSDT", OrderSide.BUY, 99.0, 1.0, 0.0, "New", "live-1")],
            target_orders=[TargetOrder("target-1", "SOLUSDT", OrderSide.BUY, 99.0, 1.0)],
            risk=risk,
            diff_count=1,
            rebuild_price_deviation_pct=DEFAULT_STRATEGY_CONFIG.execution.rebuild_price_deviation_pct,
            diff_count_threshold=DEFAULT_STRATEGY_CONFIG.execution.rebuild_diff_threshold,
        )

        self.assertFalse(rebuild_required)
        self.assertEqual(reasons, [])

    def test_should_rebuild_uses_atr_adaptive_threshold_before_price_deviation(self):
        risk = RiskDecision(
            can_trade=True,
            pause_entries=False,
            force_risk_off=False,
            cancel_entries=False,
            allow_exit_only=False,
            de_risk_mode=DeRiskMode.NONE,
        )
        rebuild_required, reasons = should_rebuild(
            StrategyState(regime=RegimeType.RANGE, bars_in_state=3, last_rebuild_price=100.0),
            price=100.5,
            atr14=10.0,
            live_orders=[LiveOrder("1", "SOLUSDT", OrderSide.BUY, 99.0, 1.0, 0.0, "New", "live-1")],
            target_orders=[TargetOrder("target-1", "SOLUSDT", OrderSide.BUY, 99.0, 1.0)],
            risk=risk,
            diff_count=0,
            rebuild_price_deviation_pct=DEFAULT_STRATEGY_CONFIG.execution.rebuild_price_deviation_pct,
            diff_count_threshold=DEFAULT_STRATEGY_CONFIG.execution.rebuild_diff_threshold,
        )

        self.assertFalse(rebuild_required)
        self.assertEqual(reasons, [])

    def test_should_rebuild_when_diff_exceeds_threshold(self):
        risk = RiskDecision(
            can_trade=True,
            pause_entries=False,
            force_risk_off=False,
            cancel_entries=False,
            allow_exit_only=False,
            de_risk_mode=DeRiskMode.NONE,
        )
        rebuild_required, reasons = should_rebuild(
            StrategyState(regime=RegimeType.RANGE, bars_in_state=3, last_rebuild_price=100.0),
            price=100.1,
            atr14=0.5,
            live_orders=[LiveOrder("1", "SOLUSDT", OrderSide.BUY, 99.0, 1.0, 0.0, "New", "live-1")],
            target_orders=[TargetOrder("target-1", "SOLUSDT", OrderSide.BUY, 99.0, 1.0)],
            risk=risk,
            diff_count=3,
            rebuild_price_deviation_pct=DEFAULT_STRATEGY_CONFIG.execution.rebuild_price_deviation_pct,
            diff_count_threshold=DEFAULT_STRATEGY_CONFIG.execution.rebuild_diff_threshold,
        )

        self.assertTrue(rebuild_required)
        self.assertIn("target_diff=3", reasons)

    def test_order_diff_treats_one_tick_price_change_as_match_for_cheap_symbol(self):
        live_orders = [
            LiveOrder(
                order_id="1",
                symbol="DOGEUSDT",
                side=OrderSide.BUY,
                price=0.1800,
                size=100.0,
                filled_size=0.0,
                status="New",
                client_order_id="doge-live-buy",
            )
        ]
        target_orders = [
            TargetOrder(
                client_order_id="doge-target-buy",
                symbol="DOGEUSDT",
                side=OrderSide.BUY,
                price=0.1801,
                size=100.0,
            )
        ]

        diff_count = target_orders_diff_count(
            live_orders,
            target_orders,
            price_diff_bps=DEFAULT_STRATEGY_CONFIG.execution.target_price_diff_bps,
            size_diff_ratio=DEFAULT_STRATEGY_CONFIG.execution.target_size_diff_ratio,
            tick_size=0.0001,
        )

        self.assertEqual(diff_count, 0)

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
