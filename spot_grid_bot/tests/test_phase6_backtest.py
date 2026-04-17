import unittest

from backtesting.engine import BacktestEngine, FillEvent
from backtesting.reporting import build_backtest_summary
from domain.models import Candle, InventorySnapshot, OrderSide
from domain.strategy_config import DEFAULT_STRATEGY_CONFIG


def _candles() -> list[Candle]:
    candles: list[Candle] = []
    price = 100.0
    for index in range(320):
        drift = -0.6 if 120 <= index < 180 else 0.45 if 180 <= index < 260 else 0.05
        open_price = price
        close_price = max(60.0, price + drift)
        high = max(open_price, close_price) + 0.8
        low = min(open_price, close_price) - 0.8
        candles.append(Candle(timestamp=index, open=open_price, high=high, low=low, close=close_price, volume=10.0))
        price = close_price
    return candles


class Phase6BacktestTests(unittest.TestCase):
    def test_fill_accounting_updates_cost_basis_and_realized_pnl(self):
        engine = BacktestEngine(DEFAULT_STRATEGY_CONFIG)
        inventory = InventorySnapshot(base_balance=0.0, quote_balance=1000.0, reserved_quote=0.0, mark_price=100.0)

        realized_buy = engine._apply_fills(
            inventory,
            [FillEvent(OrderSide.BUY, 100.0, 1.0), FillEvent(OrderSide.BUY, 120.0, 1.0)],
        )
        self.assertEqual(realized_buy, 0.0)
        self.assertIsNotNone(inventory.cost_basis_price)
        self.assertGreater(inventory.cost_basis_price, 110.0)

        realized_sell = engine._apply_fills(
            inventory,
            [FillEvent(OrderSide.SELL, 130.0, 1.0)],
        )
        self.assertGreater(realized_sell, 0.0)
        self.assertEqual(inventory.base_balance, 1.0)
        self.assertIsNotNone(inventory.cost_basis_price)

    def test_backtest_result_contains_diagnostics_and_summary(self):
        engine = BacktestEngine(DEFAULT_STRATEGY_CONFIG)
        result = engine.run("SOLUSDT", _candles())
        summary = build_backtest_summary(result)

        self.assertIn("realized_pnl", summary)
        self.assertIn("unrealized_pnl", summary)
        self.assertIn("rebuild_count", summary)
        self.assertIn("blocked_no_loss_sell_count", summary)
        self.assertIn("risk_reason_counts", summary)
        self.assertGreaterEqual(result.rebuild_count, 0)
        self.assertGreaterEqual(result.average_inventory_utilization, 0.0)
