from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class TradingCycleExecutionService:
    """Apply finalized planner decisions to persistence, execution, and notifications."""

    def __init__(self, executor, notifier, planner, state_store=None) -> None:
        self.executor = executor
        self.notifier = notifier
        self.planner = planner
        self.state_store = state_store

    async def execute(self, symbol: str, decision) -> None:
        """Execute one finalized decision for one symbol."""
        if not decision.rebuild_required:
            logger.info(
                "grid_rebuild_skipped symbol=%s reasons=%s target_diff=%s",
                symbol,
                ",".join(decision.reasons),
                decision.target_order_diff_count,
            )
            logger.info("trading_cycle_finished symbol=%s rebuild_required=false", symbol)
            return

        logger.info(
            "grid_rebuild_required symbol=%s regime=%s target_orders=%s target_diff=%s reasons=%s",
            symbol,
            decision.regime.value,
            len(decision.target_orders),
            decision.target_order_diff_count,
            ",".join(decision.reasons),
        )
        executed = await self.executor.sync_orders(symbol, decision.target_orders)
        if executed:
            await self.notifier.notify_rebuild(decision)
        logger.info(
            "trading_cycle_finished symbol=%s rebuild_required=true executed=%s target_orders=%s",
            symbol,
            executed,
            len(decision.target_orders),
        )

    async def save_runtime_state(self, symbol: str) -> None:
        """Persist current planner runtime snapshot when state store is enabled."""
        if self.state_store is None:
            return
        await self.state_store.save_symbol_state(self.planner.export_symbol_runtime(symbol))
