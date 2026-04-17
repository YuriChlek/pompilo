from __future__ import annotations

from math import inf

from domain.models import InventorySnapshot, RiskDecision
from domain.strategy_config import StrategyConfig


def reserve_quote_amount(inventory: InventorySnapshot, config: StrategyConfig) -> float:
    """Return the quote amount that should remain globally reserved."""
    return max(
        inventory.total_equity * config.risk.global_quote_reserve_pct,
        config.risk.min_quote_balance,
    )


def calculate_symbol_entry_budget(
    inventory: InventorySnapshot,
    risk: RiskDecision,
    config: StrategyConfig,
    *,
    portfolio_budget: float | None = None,
) -> float:
    """Return the maximum new-entry quote budget currently allowed for one symbol."""
    if risk.pause_entries or risk.force_risk_off or not risk.can_trade:
        return 0.0

    reserved_quote = reserve_quote_amount(inventory, config)
    available_quote_after_reserve = max(inventory.quote_balance - reserved_quote, 0.0)
    symbol_inventory_cap = inventory.total_equity * config.risk.max_symbol_inventory_pct_of_equity
    inventory_room = max(symbol_inventory_cap - inventory.inventory_notional, 0.0)
    absolute_room = max(config.risk.max_symbol_notional_cap - inventory.inventory_notional, 0.0)
    symbol_free_quote_cap = available_quote_after_reserve * config.risk.max_symbol_new_entry_pct_of_free_quote
    global_free_quote_cap = available_quote_after_reserve * config.risk.global_max_new_entry_pct_of_free_quote
    portfolio_room = inf if portfolio_budget is None else max(portfolio_budget, 0.0)

    budget = min(
        inventory_room,
        absolute_room,
        symbol_free_quote_cap,
        global_free_quote_cap,
        portfolio_room,
        inventory.available_quote,
    )
    if budget < config.risk.min_symbol_entry_notional:
        return 0.0
    return max(budget, 0.0)
