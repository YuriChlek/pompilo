from __future__ import annotations

from application.initialization_service import TradingInitializationService
from application.scheduler import TradingScheduler
from application.trading_cycle_service import TradingCycleService


def build_live_trading_cycle() -> TradingCycleService:
    """Compose the current live trading cycle from concrete adapters."""

    from domain.planner import MultiTimeframeSpotPlanner
    from infrastructure import BybitSpotExecutor, CompositeSignalNotifier, LoggingSignalNotifier, MultiTimeframeMarketDataProvider, TelegramSignalNotifier
    from utils.config import D1_REGIME_FILTER_ENABLED

    return TradingCycleService(
        market_data_provider=MultiTimeframeMarketDataProvider(),
        executor=BybitSpotExecutor(),
        notifier=CompositeSignalNotifier(LoggingSignalNotifier(), TelegramSignalNotifier()),
        planner=MultiTimeframeSpotPlanner(d1_regime_filter_enabled=D1_REGIME_FILTER_ENABLED),
    )


def build_initialization_service() -> TradingInitializationService:
    """Compose startup services for tables, migrations, and reconciliation."""

    from infrastructure import BybitSpotExecutor
    from utils.create_tables import main as create_tables_main
    from utils.run_migrations import main as run_migrations_main

    executor = BybitSpotExecutor()
    return TradingInitializationService(
        table_initializer=create_tables_main,
        migration_runner=run_migrations_main,
        executor=executor,
    )


def build_live_trading_scheduler() -> TradingScheduler:
    """Compose the live scheduler with market-data refresh and initialization."""

    from infrastructure import BinanceMarketDataSynchronizer

    return TradingScheduler(
        trading_cycle=build_live_trading_cycle(),
        market_data_synchronizer=BinanceMarketDataSynchronizer(),
        initialization_service=build_initialization_service(),
    )


__all__ = [
    "build_initialization_service",
    "build_live_trading_cycle",
    "build_live_trading_scheduler",
]
