from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trading.application.ports import MarketDataSynchronizer
    from trading.application.scheduler import TradingScheduler
    from trading.application.services import TradingCycleService


def build_runtime_strategy_config():
    """Build the runtime strategy config by overlaying environment-driven settings onto domain defaults."""
    from trading.domain.strategy_config import DEFAULT_STRATEGY_CONFIG
    from utils.config import BREAKEVEN_TRIGGER_R

    return replace(
        DEFAULT_STRATEGY_CONFIG,
        exit=replace(
            DEFAULT_STRATEGY_CONFIG.exit,
            breakeven_trigger_r=BREAKEVEN_TRIGGER_R,
        ),
    )


def build_position_management_settings():
    """Build runtime position-management switches from environment configuration."""
    from trading.infrastructure.execution_service import PositionManagementSettings
    from utils.config import (
        ENABLE_BREAKEVEN_PARTIAL_CLOSE,
        ENABLE_BREAKEVEN_STOP_MANAGEMENT,
    )

    return PositionManagementSettings(
        enable_breakeven_stop_management=ENABLE_BREAKEVEN_STOP_MANAGEMENT,
        enable_breakeven_partial_close=ENABLE_BREAKEVEN_PARTIAL_CLOSE,
    )


def build_live_trading_cycle() -> "TradingCycleService":
    """Compose the default production trading cycle from infrastructure adapters."""
    from trading.application.services import TradingCycleService
    from trading.infrastructure.execution_service import BybitPositionExecutor
    from trading.infrastructure.market_data import IndicatorMarketDataProvider
    from trading.infrastructure.notifications import TelegramSignalNotifier

    runtime_strategy_config = build_runtime_strategy_config()
    position_management_settings = build_position_management_settings()

    return TradingCycleService(
        market_data_provider=IndicatorMarketDataProvider(strategy_config=runtime_strategy_config),
        notifier=TelegramSignalNotifier(),
        executor=BybitPositionExecutor(
            strategy_config=runtime_strategy_config,
            position_management_settings=position_management_settings,
        ),
    )


def build_market_data_synchronizer() -> "MarketDataSynchronizer":
    """Compose the default market-data synchronization adapter."""
    from trading.infrastructure.data_collection import BinanceMarketDataSynchronizer

    return BinanceMarketDataSynchronizer()


def build_live_trading_scheduler() -> "TradingScheduler":
    """Compose the default scheduler for recurring live trading execution."""
    from trading.application.scheduler import TradingScheduler

    return TradingScheduler(
        trading_cycle=build_live_trading_cycle(),
        market_data_synchronizer=build_market_data_synchronizer(),
    )
