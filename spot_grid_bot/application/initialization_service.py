from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import logging
from typing import Iterable

from domain.models import SymbolRuntimeState
from infrastructure.db import ensure_candle_tables

logger = logging.getLogger(__name__)
STATE_STALE_AFTER_HOURS = 48


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
            runtime_state = self._reconcile_runtime_restore(symbol, runtime_state)
            if runtime_state is not None:
                self.planner.restore_symbol_runtime(runtime_state)
        logger.info("trading_cycle_initialize_finished symbols=%s state_store=enabled", ",".join(symbol_list))
        return disabled_symbols

    def _reconcile_runtime_restore(self, symbol: str, runtime_state: SymbolRuntimeState | None) -> SymbolRuntimeState | None:
        """Align persisted runtime state with live exchange inventory before restore."""
        exchange = getattr(self.executor, "exchange", self.executor)
        get_balances = getattr(exchange, "get_balances", None)
        get_open_orders = getattr(exchange, "get_open_orders", None)
        if get_balances is None:
            if runtime_state is None:
                logger.info("state_missing_using_defaults symbol=%s", symbol.upper())
            else:
                logger.info("state_restored symbol=%s source=state_store", symbol.upper())
            return runtime_state

        persisted_cost_basis = runtime_state.cost_basis_price if runtime_state is not None else None
        inventory = get_balances(symbol, persisted_cost_basis=persisted_cost_basis)
        live_orders = get_open_orders(symbol) if get_open_orders is not None else []
        live_order_count = len(live_orders)

        if runtime_state is None:
            if inventory.base_balance > 0 or live_order_count > 0:
                logger.warning(
                    "state_missing_live_state_detected symbol=%s base_balance=%.6f live_orders=%s -- using_defaults",
                    symbol.upper(),
                    inventory.base_balance,
                    live_order_count,
                )
            else:
                logger.info("state_missing_using_defaults symbol=%s", symbol.upper())
            return None

        restored_state = replace(
            runtime_state,
            strategy_state=replace(runtime_state.strategy_state),
            risk_state=replace(runtime_state.risk_state),
        )

        if inventory.base_balance <= 0:
            if restored_state.cost_basis_price is not None:
                logger.info(
                    "state_cost_basis_cleared_no_inventory symbol=%s persisted=%.4f",
                    symbol.upper(),
                    restored_state.cost_basis_price,
                )
            restored_state.cost_basis_price = None
        elif inventory.cost_basis_price is not None and inventory.cost_basis_price > 0:
            if restored_state.cost_basis_price != inventory.cost_basis_price:
                logger.info(
                    "state_cost_basis_refreshed_from_live symbol=%s persisted=%s live=%.4f",
                    symbol.upper(),
                    restored_state.cost_basis_price,
                    inventory.cost_basis_price,
                )
            restored_state.cost_basis_price = inventory.cost_basis_price
        restored_state.last_known_base_balance = inventory.base_balance
        restored_state.last_known_quote_balance = inventory.quote_balance
        restored_state.last_known_reserved_quote = inventory.reserved_quote
        restored_state.last_known_mark_price = inventory.mark_price
        if restored_state.last_cycle_completed_at is not None:
            stale_before = datetime.now(timezone.utc) - timedelta(hours=STATE_STALE_AFTER_HOURS)
            if restored_state.last_cycle_completed_at < stale_before:
                logger.warning(
                    "state_stale symbol=%s last_cycle_completed_at=%s threshold_hours=%s",
                    symbol.upper(),
                    restored_state.last_cycle_completed_at.isoformat(),
                    STATE_STALE_AFTER_HOURS,
                )

        logger.info(
            "state_restored symbol=%s base_balance=%.6f live_orders=%s cost_basis=%s",
            symbol.upper(),
            inventory.base_balance,
            live_order_count,
            restored_state.cost_basis_price,
        )
        return restored_state
