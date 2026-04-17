from __future__ import annotations

import logging
from typing import Iterable

from infrastructure.db import ensure_candle_tables

logger = logging.getLogger(__name__)


class TradingCycleInitializationService:
    """Handle startup initialization for trading cycles."""

    def __init__(self, executor, planner, state_store=None) -> None:
        self.executor = executor
        self.planner = planner
        self.state_store = state_store

    async def initialize(self, symbols: Iterable[str], *, ensure_tables) -> set[str]:
        """Ensure candle tables, reconcile venue state, and restore persisted runtime."""
        symbol_list = [str(symbol).upper() for symbol in symbols]
        logger.info("trading_cycle_initialize_started symbols=%s", ",".join(symbol_list))
        await ensure_tables(symbol_list)
        await self.executor.reconcile_state(symbol_list)
        disabled_symbols = {
            symbol
            for symbol in symbol_list
            if hasattr(self.executor, "is_symbol_supported") and not self.executor.is_symbol_supported(symbol)
        }
        if disabled_symbols:
            logger.warning("trading_cycle_symbols_disabled symbols=%s", ",".join(sorted(disabled_symbols)))
        if self.state_store is None:
            logger.info("trading_cycle_initialize_finished symbols=%s state_store=disabled", ",".join(symbol_list))
            return disabled_symbols
        await self.state_store.initialize()
        for symbol in symbol_list:
            if symbol in disabled_symbols:
                continue
            runtime_state = await self.state_store.load_symbol_state(symbol)
            if runtime_state is not None:
                self.planner.restore_symbol_runtime(runtime_state)
        logger.info("trading_cycle_initialize_finished symbols=%s state_store=enabled", ",".join(symbol_list))
        return disabled_symbols
