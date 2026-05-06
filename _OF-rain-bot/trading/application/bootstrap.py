"""Composition root for the canonical trading package."""

from __future__ import annotations

from dataclasses import replace

from trading.application.ports import ExecutionPort
from trading.application.signal_engine import ScalpSignalEngine
from trading.application.state_machine import SymbolRuntimeState
from trading.application.services import TradingService
from trading.domain.risk_manager import RiskManager
from trading.domain.strategy_config import DEFAULT_STRATEGY_CONFIG, StrategyConfig
from trading.infrastructure.execution_service import BybitExecutionService
from trading.infrastructure.market_data import CanonicalMarketDataProvider
from trading.infrastructure.storage.migrations import DatabaseMigrationRunner
from trading.infrastructure.storage.queued_repository import QueuedRuntimeEventRepository
from trading.infrastructure.storage.repository import RuntimeEventRepository
from utils.config import APP_CONFIG


def build_strategy_config() -> StrategyConfig:
    """Build the typed strategy config from domain defaults and env overrides."""
    orderflow = APP_CONFIG.orderflow
    return StrategyConfig(
        liquidity_detection=replace(
            DEFAULT_STRATEGY_CONFIG.liquidity_detection,
            max_chase_ticks=orderflow.max_chase_ticks,
            max_wall_distance_bps=orderflow.max_wall_distance_bps,
            max_wall_distance_ticks=orderflow.max_wall_distance_ticks,
            max_wall_size_drop_pct=orderflow.max_wall_size_drop_pct,
            min_defended_ratio=orderflow.min_defended_ratio,
            min_wall_notional_usdt=orderflow.min_wall_notional_usdt,
            min_wall_persist_ms=orderflow.min_wall_persist_ms,
            min_wall_relative_size=orderflow.min_wall_relative_size,
            test_debounce_ms=orderflow.test_debounce_ms,
            test_touch_ticks=orderflow.test_touch_ticks,
            min_wall_score=orderflow.min_wall_score,
        ),
        spoof_filter=replace(
            DEFAULT_STRATEGY_CONFIG.spoof_filter,
            max_chase_ticks=orderflow.max_chase_ticks,
            max_pull_events=orderflow.max_pull_events,
            max_spoof_score=orderflow.max_spoof_score,
            min_defended_ratio=orderflow.min_defended_ratio,
        ),
        signal_generation=replace(
            DEFAULT_STRATEGY_CONFIG.signal_generation,
            analysis_reference_exchange=orderflow.analysis_reference_exchange,
            book_stale_ms=orderflow.book_stale_ms,
            cross_confirmation_bps=orderflow.cross_confirmation_bps,
            entry_offset_ticks=orderflow.entry_offset_ticks,
            stop_loss_size=orderflow.stop_loss_size,
            invalidation_offset_ticks=orderflow.invalidation_offset_ticks,
            max_spread_ticks=orderflow.max_spread_ticks,
            min_cross_exchange_confirmations=orderflow.min_cross_exchange_confirmations,
            min_defended_ratio=orderflow.min_defended_ratio,
            min_rejection_ticks=orderflow.min_rejection_ticks,
            min_tape_pressure_ratio=orderflow.min_tape_pressure_ratio,
            min_test_count=orderflow.min_test_count,
            min_wall_score=orderflow.min_wall_score,
            stop_offset_ticks=orderflow.stop_offset_ticks,
            take_profit_r_multiple=orderflow.take_profit_r_multiple,
            tape_window_ms=orderflow.tape_window_ms,
            test_touch_ticks=orderflow.test_touch_ticks,
        ),
    )


def build_execution_port(repository: RuntimeEventRepository) -> ExecutionPort:
    """Build the canonical execution adapter behind the application port."""
    return BybitExecutionService(repository)


def build_market_data_provider() -> CanonicalMarketDataProvider:
    """Build the canonical market-data adapter behind the application layer."""
    return CanonicalMarketDataProvider()


def build_trading_runtime(*, dry_run: bool = False):
    """Build the fully wired canonical runtime object."""
    from trading.application.runner import CanonicalTradingRuntime
    from trading.application.runtime import OrderFlowScalpBot

    strategy_config = build_strategy_config()
    market_data = build_market_data_provider()
    repository = QueuedRuntimeEventRepository()
    execution = build_execution_port(repository)
    risk_manager = RiskManager()
    migrations = DatabaseMigrationRunner()
    signal_engine = ScalpSignalEngine(
        market_data.orderbooks,
        market_data.tape_store,
        config=strategy_config,
    )
    symbol_states = {symbol: SymbolRuntimeState(symbol=symbol) for symbol in APP_CONFIG.orderflow.symbols}

    bot = OrderFlowScalpBot(
        dry_run=dry_run,
        orderbooks=market_data.orderbooks,
        tape_store=market_data.tape_store,
        feed_manager=market_data.feed_manager,
        strategy_config=strategy_config,
        signal_engine=signal_engine,
        risk_manager=risk_manager,
        migrations=migrations,
        repository=repository,
        executor=execution,
        symbol_states=symbol_states,
    )
    bot.trading_service = TradingService(runtime=bot, dry_run=dry_run)
    return CanonicalTradingRuntime(bot=bot, dry_run=dry_run)
