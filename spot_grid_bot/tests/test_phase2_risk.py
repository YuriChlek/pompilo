import unittest

from domain.models import Candle, InventorySnapshot, LiveOrder, OrderSide, RegimeType, RiskRuntimeState
from domain.risk_manager import RiskManager
from domain.strategy_config import DEFAULT_STRATEGY_CONFIG


def _candles(spike: bool = False) -> list[Candle]:
    candles: list[Candle] = []
    price = 100.0
    for index in range(260):
        high = price + 1.0
        low = price - 1.0
        close = price + (0.1 if index % 2 == 0 else -0.1)
        if spike and index == 259:
            high = price + 8.0
            low = price - 8.0
            close = price + 3.0
        candles.append(Candle(timestamp=index, open=price, high=high, low=low, close=close, volume=10.0))
        price = close
    return candles


class _Indicators:
    ema20 = 100.0
    ema50 = 100.0
    ema200 = 100.0
    atr14 = 1.0
    realized_volatility = 0.01
    ema50_slope = 0.0
    range_width = 4.0
    price_vs_ema50 = 0.0
    directional_move = 0.3
    abnormal_candle = False
    atr_spike = False


class Phase2RiskTests(unittest.TestCase):
    def test_outstanding_buy_orders_are_accounted_in_projected_exposure(self):
        manager = RiskManager(DEFAULT_STRATEGY_CONFIG)
        inventory = InventorySnapshot(base_balance=0.05, quote_balance=1_000.0, reserved_quote=0.0, mark_price=100.0)
        live_orders = [
            LiveOrder(
                order_id="1",
                symbol="SOLUSDT",
                side=OrderSide.BUY,
                price=100.0,
                size=60.0,
                filled_size=0.0,
                status="New",
                client_order_id="big-buy",
            )
        ]

        decision = manager.evaluate(_candles(), _Indicators(), inventory, live_orders, RiskRuntimeState(), RegimeType.RANGE, 0)

        self.assertGreater(decision.outstanding_buy_notional, 0.0)
        self.assertIn("projected_quote_usage_limit", decision.reasons)
        self.assertTrue(decision.pause_entries)

    def test_volatility_cooldown_keeps_entries_paused(self):
        manager = RiskManager(DEFAULT_STRATEGY_CONFIG)
        inventory = InventorySnapshot(base_balance=0.0, quote_balance=1_000.0, reserved_quote=0.0, mark_price=100.0)

        decision = manager.evaluate(_candles(), _Indicators(), inventory, [], RiskRuntimeState(), RegimeType.RANGE, 2)

        self.assertTrue(decision.pause_entries)
        self.assertIn("volatility_cooldown_active", decision.reasons)
