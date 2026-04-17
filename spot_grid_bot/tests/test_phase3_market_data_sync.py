import unittest

from application.bootstrap import build_live_trading_scheduler, build_market_data_synchronizer
from infrastructure.binance_market_data_synchronizer import BinanceMarketDataSynchronizer


class Phase3MarketDataSyncTests(unittest.TestCase):
    def test_build_market_data_synchronizer_returns_binance_synchronizer(self):
        synchronizer = build_market_data_synchronizer()

        self.assertIsInstance(synchronizer, BinanceMarketDataSynchronizer)

    def test_live_scheduler_uses_real_market_data_synchronizer(self):
        scheduler = build_live_trading_scheduler()

        self.assertIsInstance(scheduler.market_data_synchronizer, BinanceMarketDataSynchronizer)
