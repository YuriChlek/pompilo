from __future__ import annotations

import logging
from dataclasses import replace

from application.health import HealthCheckServer, RuntimeHealthTracker
from application.scheduler import TradingScheduler
from application.trading_cycle_service import SpotTradingCycleService
from domain.portfolio_allocator import PortfolioAllocator
from domain.spot_grid_planner import SpotGridPlanner
from domain.strategy_config import DEFAULT_STRATEGY_CONFIG
from infrastructure.binance_market_data_synchronizer import BinanceMarketDataSynchronizer
from infrastructure.execution_gateway import BybitSpotExecutionService, PaperExecutionService
from infrastructure.live_price_monitor import BybitLivePriceMonitor
from infrastructure.market_data_provider import DatabaseMarketDataProvider
from infrastructure.notifications import LoggingSignalNotifier, TelegramNotifierConfig, TelegramSignalNotifier
from infrastructure.state_store import PostgresStateStore
from utils.config import (
    EXECUTION_MODE,
    BYBIT_PUBLIC_WS_URL,
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
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    HEALTHCHECK_HOST,
    HEALTHCHECK_PORT,
    ENABLE_LIVE_PRICE_MONITOR,
    LIVE_PRICE_DEVIATION_ATR_MULTIPLIER,
    LIVE_PRICE_MONITOR_COOLDOWN_SECONDS,
    LIVE_PRICE_MONITOR_OPEN_TIMEOUT_SECONDS,
    LIVE_PRICE_MONITOR_RECONNECT_DELAY_SECONDS,
)

logger = logging.getLogger(__name__)


def validate_strategy_config(config) -> None:
    """Validate runtime config coherence and warn about ineffective settings."""
    risk = config.risk
    if risk.max_inventory_notional < risk.max_symbol_notional_cap:
        logger.warning(
            "config_warn max_symbol_notional_cap=%.2f exceeds max_inventory_notional=%.2f; max_symbol_notional_cap may never be active",
            risk.max_symbol_notional_cap,
            risk.max_inventory_notional,
        )
    if risk.min_symbol_entry_notional >= risk.max_symbol_notional_cap:
        raise ValueError(
            "min_symbol_entry_notional >= max_symbol_notional_cap -- new symbol entry orders are impossible"
        )


def build_runtime_strategy_config():
    """Build runtime strategy settings by overlaying environment values onto defaults."""
    runtime_config = replace(
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
    validate_strategy_config(runtime_config)
    return runtime_config


def build_live_trading_cycle(*, health_tracker: RuntimeHealthTracker | None = None) -> SpotTradingCycleService:
    """Compose the default live trading cycle from infrastructure adapters."""
    runtime_strategy_config = build_runtime_strategy_config()
    executor = build_order_executor(runtime_strategy_config)
    return SpotTradingCycleService(
        market_data_provider=DatabaseMarketDataProvider(runtime_strategy_config, executor.exchange),
        executor=executor,
        notifier=build_signal_notifier(),
        planner=SpotGridPlanner(runtime_strategy_config),
        state_store=PostgresStateStore(),
        portfolio_allocator=PortfolioAllocator(runtime_strategy_config),
        health_tracker=health_tracker,
    )


def build_order_executor(runtime_strategy_config):
    """Build the configured execution adapter for live or paper trading."""
    if EXECUTION_MODE == "paper":
        return PaperExecutionService(runtime_strategy_config)
    return BybitSpotExecutionService(runtime_strategy_config)


def build_market_data_synchronizer() -> BinanceMarketDataSynchronizer:
    """Build the default market-data synchronization adapter."""
    runtime_strategy_config = build_runtime_strategy_config()
    return BinanceMarketDataSynchronizer(
        timeframe=None,
        higher_timeframe=runtime_strategy_config.market_data.higher_timeframe_interval,
    )


def build_signal_notifier():
    """Build the configured signal notifier with safe fallback to logging."""
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        return TelegramSignalNotifier(
            TelegramNotifierConfig(
                bot_token=TELEGRAM_BOT_TOKEN,
                chat_id=TELEGRAM_CHAT_ID,
            )
        )
    return LoggingSignalNotifier()


def build_live_trading_scheduler() -> TradingScheduler:
    """Compose the default scheduler for recurring spot trading execution."""
    health_tracker = RuntimeHealthTracker()
    trading_cycle = build_live_trading_cycle(health_tracker=health_tracker)
    live_price_monitor = None
    if ENABLE_LIVE_PRICE_MONITOR and EXECUTION_MODE != "paper":
        live_price_monitor = BybitLivePriceMonitor(
            reference_provider=trading_cycle.planner.get_live_price_reference,
            on_deviation=None,
            atr_multiplier=LIVE_PRICE_DEVIATION_ATR_MULTIPLIER,
            cooldown_seconds=LIVE_PRICE_MONITOR_COOLDOWN_SECONDS,
            open_timeout_seconds=LIVE_PRICE_MONITOR_OPEN_TIMEOUT_SECONDS,
            reconnect_delay_seconds=LIVE_PRICE_MONITOR_RECONNECT_DELAY_SECONDS,
            websocket_url=BYBIT_PUBLIC_WS_URL,
        )
    return TradingScheduler(
        trading_cycle=trading_cycle,
        market_data_synchronizer=build_market_data_synchronizer(),
        live_price_monitor=live_price_monitor,
        health_tracker=health_tracker,
    )


def build_health_check_server(tracker: RuntimeHealthTracker) -> HealthCheckServer:
    """Build the runtime health-check server for the current process."""
    return HealthCheckServer(tracker, host=HEALTHCHECK_HOST, port=HEALTHCHECK_PORT)
