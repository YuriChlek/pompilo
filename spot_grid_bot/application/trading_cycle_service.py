from __future__ import annotations

from typing import Iterable

from application.analysis_batch_service import TradingCycleAnalysisBatchService
from application.execution_service import TradingCycleExecutionService
from application.initialization_service import TradingCycleInitializationService
from application.ports import MarketDataProvider, OrderExecutor, SignalNotifier, StateStore
from domain.portfolio_allocator import PortfolioAllocator
from domain.spot_grid_planner import SpotGridPlanner
from infrastructure.db import ensure_candle_tables


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
    ) -> None:
        """Store application-level collaborators for spot trading cycles."""
        self.market_data_provider = market_data_provider
        self.executor = executor
        self.notifier = notifier
        self.planner = planner
        self.state_store = state_store
        self.portfolio_allocator = portfolio_allocator or PortfolioAllocator()
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

    async def run_many(self, symbols: Iterable[str]) -> dict[str, object | None]:
        """Run one two-pass trading cycle for multiple symbols and return per-symbol results."""
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
                await self._execution_service.save_runtime_state(symbol)
            except Exception:
                import logging
                logger = logging.getLogger(__name__)
                logger.exception("trading_cycle_failed symbol=%s phase=planning", symbol)
                results[symbol] = None
                continue

            results[symbol] = decision
            await self._execution_service.execute(symbol, decision)
        return results
