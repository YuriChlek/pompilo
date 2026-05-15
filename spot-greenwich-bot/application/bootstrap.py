from __future__ import annotations

from application.initialization_service import TradingInitializationService
from application.scheduler import TradingScheduler
from application.trading_cycle_service import TradingCycleService


def _resolve_notification_only_mode(notification_only_mode: bool | None) -> bool:
    from utils.config import NOTIFICATION_ONLY_MODE

    if notification_only_mode is None:
        return NOTIFICATION_ONLY_MODE
    return notification_only_mode


def build_live_trading_cycle(*, notification_only_mode: bool | None = None) -> TradingCycleService:
    """Compose the current live trading cycle from concrete adapters."""

    from domain.planner import MultiTimeframeSpotPlanner
    from infrastructure import BybitSpotExecutor, CompositeSignalNotifier, LoggingSignalNotifier, MultiTimeframeMarketDataProvider, TelegramSignalNotifier
    from utils.config import D1_REGIME_FILTER_ENABLED

    return TradingCycleService(
        market_data_provider=MultiTimeframeMarketDataProvider(),
        executor=BybitSpotExecutor(notification_only_mode=_resolve_notification_only_mode(notification_only_mode)),
        notifier=CompositeSignalNotifier(LoggingSignalNotifier(), TelegramSignalNotifier()),
        planner=MultiTimeframeSpotPlanner(d1_regime_filter_enabled=D1_REGIME_FILTER_ENABLED),
    )


def build_initialization_service(*, notification_only_mode: bool | None = None) -> TradingInitializationService:
    """Compose startup services for tables, migrations, and reconciliation."""

    from infrastructure import BybitSpotExecutor
    from utils.create_tables import main as create_tables_main
    from utils.run_migrations import main as run_migrations_main

    executor = BybitSpotExecutor(notification_only_mode=_resolve_notification_only_mode(notification_only_mode))
    return TradingInitializationService(
        table_initializer=create_tables_main,
        migration_runner=run_migrations_main,
        executor=executor,
    )


def build_live_trading_scheduler(*, notification_only_mode: bool | None = None) -> TradingScheduler:
    """Compose the live scheduler with market-data refresh and initialization."""

    return TradingScheduler(
        trading_cycle=build_live_trading_cycle(notification_only_mode=notification_only_mode),
        market_data_synchronizer=build_market_data_synchronizer(),
        initialization_service=build_initialization_service(notification_only_mode=notification_only_mode),
    )


def build_market_data_synchronizer():
    """Compose the market-data synchronization adapter used by live-like flows."""

    from infrastructure import BinanceMarketDataSynchronizer

    return BinanceMarketDataSynchronizer()


__all__ = [
    "build_initialization_service",
    "build_live_trading_cycle",
    "build_market_data_synchronizer",
    "build_live_trading_scheduler",
]
