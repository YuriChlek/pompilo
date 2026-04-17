from __future__ import annotations

from trading.application.scheduler import TradingScheduler
from trading.application.services import TradingCycleService
from trading.infrastructure.data_collection import BinanceMarketDataSynchronizer
from trading.infrastructure.execution_service import BinanceSpotExecutor
from trading.infrastructure.market_data import DatabaseMarketDataProvider
from trading.infrastructure.notifications import LoggingSignalNotifier


def build_live_trading_cycle() -> TradingCycleService:
    return TradingCycleService(
        market_data_provider=DatabaseMarketDataProvider(),
        executor=BinanceSpotExecutor(),
        notifier=LoggingSignalNotifier(),
    )


def build_live_trading_scheduler() -> TradingScheduler:
    return TradingScheduler(
        trading_cycle=build_live_trading_cycle(),
        market_data_synchronizer=BinanceMarketDataSynchronizer(),
    )
