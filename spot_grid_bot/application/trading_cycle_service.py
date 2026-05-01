from __future__ import annotations

import logging
from typing import Iterable

from application.analysis_batch_service import TradingCycleAnalysisBatchService
from application.execution_service import TradingCycleExecutionService
from application.health import RuntimeHealthTracker
from application.initialization_service import TradingCycleInitializationService
from application.ports import MarketDataProvider, OrderExecutor, SignalNotifier, StateStore
from domain.portfolio_allocator import PortfolioAllocator
from domain.spot_grid_planner import SpotGridPlanner
from infrastructure.db import ensure_candle_tables

logger = logging.getLogger(__name__)


def _is_infrastructure_error(exc: Exception) -> bool:
    """Return whether the failure looks like an infrastructure/runtime error."""
    if isinstance(exc, (OSError, MemoryError)):
        return True
    return exc.__class__.__name__ == "PostgresError"


class SpotTradingCycleService:
    """Application service that orchestrates one spot trading cycle for a single symbol."""

    def __init__(
        self,
        market_data_provider: MarketDataProvider,
        executor: OrderExecutor,
        notifier: SignalNotifier,
        planner: SpotGridPlanner,
        state_store: StateStore | None = None,
        portfolio_allocator: PortfolioAllocator | None = None,
        health_tracker: RuntimeHealthTracker | None = None,
    ) -> None:
        """Store application-level collaborators for spot trading cycles."""
        self.market_data_provider = market_data_provider
        self.executor = executor
        self.notifier = notifier
        self.planner = planner
        self.state_store = state_store
        self.portfolio_allocator = portfolio_allocator or PortfolioAllocator()
        self.health_tracker = health_tracker
        self._disabled_symbols: set[str] = set()
        self._initialization_service = TradingCycleInitializationService(executor, planner, state_store)
        self._analysis_service = TradingCycleAnalysisBatchService(market_data_provider, planner, self.portfolio_allocator)
        self._execution_service = TradingCycleExecutionService(executor, notifier, planner, state_store)

    async def initialize(self, symbols: Iterable[str]) -> None:
        """Reconcile exchange state and restore persisted runtime state for symbols."""
        self._disabled_symbols = await self._initialization_service.initialize(symbols, ensure_tables=ensure_candle_tables)

    async def run(self, symbol: str) -> object:
        """Plan one trading cycle, persist runtime state, and sync orders when needed."""
        results = await self.run_many([symbol])
        return results.get(symbol.upper())

    async def preview_many(self, symbols: Iterable[str]) -> dict[str, object | None]:
        """Build final strategy decisions without executing or persisting runtime state."""
        return await self._plan_many(symbols, execute=False)

    async def run_many(self, symbols: Iterable[str]) -> dict[str, object | None]:
        """Run one two-pass trading cycle for multiple symbols and return per-symbol results."""
        return await self._plan_many(symbols, execute=True)

    async def _plan_many(self, symbols: Iterable[str], *, execute: bool) -> dict[str, object | None]:
        """Run the analysis/planning pass and optionally execute resulting decisions."""
        results, context_by_symbol, analysis_by_symbol, allocation_plan = await self._analysis_service.analyze(
            symbols,
            self._disabled_symbols,
        )
        if allocation_plan is None:
            return results
        for symbol in context_by_symbol:
            try:
                decision = self.planner.plan_from_analysis(
                    context_by_symbol[symbol],
                    analysis_by_symbol[symbol],
                    portfolio_budget=allocation_plan.budget_for(symbol),
                )
                if self.health_tracker is not None:
                    self.health_tracker.record_symbol_state(
                        symbol,
                        {
                            "regime": decision.regime.value,
                            "kill_switch_count": decision.kill_switch_count,
                            "target_order_diff_count": decision.target_order_diff_count,
                            "rebuild_required": decision.rebuild_required,
                            "mark_price": context_by_symbol[symbol].inventory.mark_price,
                            "reasons": list(decision.reasons),
                        },
                    )
                if execute:
                    await self._execution_service.save_runtime_state(symbol)
            except Exception as exc:
                log_fn = logger.critical if _is_infrastructure_error(exc) else logger.exception
                log_fn(
                    "trading_cycle_failed symbol=%s phase=planning error_type=%s",
                    symbol,
                    type(exc).__name__,
                )
                results[symbol] = None
                continue

            results[symbol] = decision
            if execute:
                await self._execution_service.execute(symbol, decision)
        return results
