from __future__ import annotations

from dataclasses import replace

from application.scheduler import TradingScheduler
from application.trading_cycle_service import SpotTradingCycleService
from domain.portfolio_allocator import PortfolioAllocator
from domain.spot_grid_planner import SpotGridPlanner
from domain.strategy_config import DEFAULT_STRATEGY_CONFIG
from infrastructure.binance_market_data_synchronizer import BinanceMarketDataSynchronizer
from infrastructure.execution_gateway import BybitSpotExecutionService, PaperExecutionService
from infrastructure.market_data_provider import DatabaseMarketDataProvider
from infrastructure.notifications import LoggingSignalNotifier
from infrastructure.state_store import PostgresStateStore
from utils.config import (
    EXECUTION_MODE,
    CANDLE_LOOKBACK,
    DEFAULT_BASE_ASSET,
    DEFAULT_QUOTE_ASSET,
    MAX_NEW_ORDERS_PER_CYCLE,
    MAX_SYMBOL_INVENTORY_PCT_OF_EQUITY,
    MAX_SYMBOL_NEW_ENTRY_PCT_OF_FREE_QUOTE,
    MIN_SYMBOL_ENTRY_NOTIONAL,
    STARTING_BASE_BALANCE,
    STARTING_QUOTE_BALANCE,
    UNDERWATER_AVERAGING_ENABLED,
    UNDERWATER_AVERAGING_TRIGGER_PCT,
    UNDERWATER_RECOVERY_BUDGET_PCT,
    UNDERWATER_RANGE_BUDGET_MULTIPLIER,
    UNDERWATER_UPTREND_BUDGET_MULTIPLIER,
    UNDERWATER_MAX_RECOVERY_LEVELS,
    UNDERWATER_DEEP_STOP_PCT,
)


def build_runtime_strategy_config():
    """Build runtime strategy settings by overlaying environment values onto defaults."""
    return replace(
        DEFAULT_STRATEGY_CONFIG,
        market_data=replace(DEFAULT_STRATEGY_CONFIG.market_data, candles_lookback=CANDLE_LOOKBACK),
        execution=replace(
            DEFAULT_STRATEGY_CONFIG.execution,
            max_new_orders_per_cycle=MAX_NEW_ORDERS_PER_CYCLE,
        ),
        grid=replace(
            DEFAULT_STRATEGY_CONFIG.grid,
            underwater_averaging_enabled=UNDERWATER_AVERAGING_ENABLED,
            underwater_averaging_trigger_pct=UNDERWATER_AVERAGING_TRIGGER_PCT,
            underwater_recovery_budget_pct=UNDERWATER_RECOVERY_BUDGET_PCT,
            underwater_range_budget_multiplier=UNDERWATER_RANGE_BUDGET_MULTIPLIER,
            underwater_uptrend_budget_multiplier=UNDERWATER_UPTREND_BUDGET_MULTIPLIER,
            underwater_max_recovery_levels=UNDERWATER_MAX_RECOVERY_LEVELS,
            underwater_deep_stop_pct=UNDERWATER_DEEP_STOP_PCT,
        ),
        risk=replace(
            DEFAULT_STRATEGY_CONFIG.risk,
            min_symbol_entry_notional=MIN_SYMBOL_ENTRY_NOTIONAL,
            max_symbol_inventory_pct_of_equity=MAX_SYMBOL_INVENTORY_PCT_OF_EQUITY,
            max_symbol_new_entry_pct_of_free_quote=MAX_SYMBOL_NEW_ENTRY_PCT_OF_FREE_QUOTE,
        ),
        portfolio=replace(
            DEFAULT_STRATEGY_CONFIG.portfolio,
            base_asset=DEFAULT_BASE_ASSET,
            quote_asset=DEFAULT_QUOTE_ASSET,
            starting_quote_balance=STARTING_QUOTE_BALANCE,
            starting_base_balance=STARTING_BASE_BALANCE,
        ),
    )


def build_live_trading_cycle() -> SpotTradingCycleService:
    """Compose the default live trading cycle from infrastructure adapters."""
    runtime_strategy_config = build_runtime_strategy_config()
    executor = build_order_executor(runtime_strategy_config)
    return SpotTradingCycleService(
        market_data_provider=DatabaseMarketDataProvider(runtime_strategy_config, executor.exchange),
        executor=executor,
        notifier=LoggingSignalNotifier(),
        planner=SpotGridPlanner(runtime_strategy_config),
        state_store=PostgresStateStore(),
        portfolio_allocator=PortfolioAllocator(runtime_strategy_config),
    )


def build_order_executor(runtime_strategy_config):
    """Build the configured execution adapter for live or paper trading."""
    if EXECUTION_MODE == "paper":
        return PaperExecutionService(runtime_strategy_config)
    return BybitSpotExecutionService(runtime_strategy_config)


def build_market_data_synchronizer() -> BinanceMarketDataSynchronizer:
    """Build the default market-data synchronization adapter."""
    return BinanceMarketDataSynchronizer()


def build_live_trading_scheduler() -> TradingScheduler:
    """Compose the default scheduler for recurring spot trading execution."""
    return TradingScheduler(
        trading_cycle=build_live_trading_cycle(),
        market_data_synchronizer=build_market_data_synchronizer(),
    )
