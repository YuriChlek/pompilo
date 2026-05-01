from __future__ import annotations

import logging
from typing import Iterable

logger = logging.getLogger(__name__)


def _is_infrastructure_error(exc: Exception) -> bool:
    """Return whether the failure looks like an infrastructure/runtime error."""
    if isinstance(exc, (OSError, MemoryError)):
        return True
    return exc.__class__.__name__ == "PostgresError"


class TradingCycleAnalysisBatchService:
    """Collect market contexts, run preliminary analysis, and build allocation plan."""

    def __init__(self, market_data_provider, planner, portfolio_allocator) -> None:
        self.market_data_provider = market_data_provider
        self.planner = planner
        self.portfolio_allocator = portfolio_allocator

    async def analyze(self, symbols: Iterable[str], disabled_symbols: set[str]) -> tuple[dict[str, object | None], dict[str, object], dict[str, object], object | None]:
        """Return partial results, context map, analysis map, and allocation plan."""
        contexts = []
        analyses = []
        results: dict[str, object | None] = {}

        for symbol in [str(item).upper() for item in symbols]:
            if symbol in disabled_symbols:
                logger.warning("trading_cycle_skipped symbol=%s reason=unsupported_symbol", symbol)
                results[symbol] = None
                continue
            logger.info("trading_cycle_started symbol=%s", symbol)
            try:
                persisted_cost_basis = self.planner.get_persisted_cost_basis(symbol)
                context = await self.market_data_provider.get_market_context(
                    symbol,
                    persisted_cost_basis=persisted_cost_basis,
                )
                self.planner.update_cost_basis(symbol, context.inventory.cost_basis_price)
                analysis = self.planner.analyze(context)
            except Exception as exc:
                log_fn = logger.critical if _is_infrastructure_error(exc) else logger.exception
                log_fn(
                    "trading_cycle_failed symbol=%s phase=analysis error_type=%s",
                    symbol,
                    type(exc).__name__,
                )
                results[symbol] = None
                continue
            contexts.append(context)
            analyses.append(analysis)

        if not contexts:
            return results, {}, {}, None

        snapshot = self.portfolio_allocator.build_snapshot(contexts, analyses)
        allocation_plan = self.portfolio_allocator.allocate(snapshot)
        logger.info(
            "portfolio_allocation_snapshot symbols=%s total_equity=%.2f total_available_quote=%.2f total_inventory_notional=%.2f outstanding_buy_notional=%.2f",
            len(snapshot.symbols),
            snapshot.total_equity,
            snapshot.total_available_quote,
            snapshot.total_inventory_notional,
            snapshot.total_outstanding_buy_notional,
        )
        analysis_by_symbol = {analysis.symbol.upper(): analysis for analysis in analyses}
        context_by_symbol = {context.symbol.upper(): context for context in contexts}
        return results, context_by_symbol, analysis_by_symbol, allocation_plan
