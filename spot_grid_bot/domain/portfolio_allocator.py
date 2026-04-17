from __future__ import annotations

from math import isfinite

from domain.strategy_config import StrategyConfig, DEFAULT_STRATEGY_CONFIG
from domain.models import (
    MarketContext,
    PortfolioAllocationPlan,
    PortfolioSnapshot,
    PreliminarySymbolAnalysis,
    RegimeType,
    SymbolAllocationBudget,
    SymbolAllocationInput,
)


class PortfolioAllocator:
    """Build portfolio-wide snapshots and placeholder allocation plans for multi-symbol cycles."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        """Store runtime configuration for portfolio-level budget distribution."""
        self.config = config or DEFAULT_STRATEGY_CONFIG

    def build_snapshot(
        self,
        contexts: list[MarketContext],
        analyses: list[PreliminarySymbolAnalysis],
    ) -> PortfolioSnapshot:
        """Aggregate portfolio totals and per-symbol inputs from one analysis pass."""
        analysis_by_symbol = {analysis.symbol.upper(): analysis for analysis in analyses}
        symbols: list[SymbolAllocationInput] = []
        total_inventory_notional = 0.0
        total_outstanding_buy_notional = 0.0
        shared_quote_balance = 0.0
        shared_reserved_quote = 0.0
        shared_available_quote = 0.0

        for context in contexts:
            symbol = context.symbol.upper()
            analysis = analysis_by_symbol[symbol]
            inventory = context.inventory
            shared_quote_balance = max(shared_quote_balance, inventory.quote_balance)
            shared_reserved_quote = max(shared_reserved_quote, inventory.reserved_quote)
            shared_available_quote = max(shared_available_quote, inventory.available_quote)
            total_inventory_notional += inventory.inventory_notional
            total_outstanding_buy_notional += analysis.risk.outstanding_buy_notional
            symbols.append(
                SymbolAllocationInput(
                    symbol=symbol,
                    regime=analysis.regime_snapshot.regime,
                    confidence=analysis.regime_snapshot.confidence,
                    inventory_notional=inventory.inventory_notional,
                    mark_price=inventory.mark_price,
                    cost_basis_price=inventory.cost_basis_price,
                    underwater_ratio=_underwater_ratio(inventory),
                    available_quote=inventory.available_quote,
                    outstanding_buy_notional=analysis.risk.outstanding_buy_notional,
                    projected_inventory_notional=analysis.risk.projected_inventory_notional,
                    projected_quote_usage=analysis.risk.projected_quote_usage,
                    pause_entries=analysis.risk.pause_entries,
                    force_risk_off=analysis.risk.force_risk_off,
                    reasons=list(analysis.risk.reasons),
                )
            )

        total_quote_balance = shared_quote_balance
        total_reserved_quote = shared_reserved_quote
        total_available_quote = shared_available_quote
        total_equity = total_quote_balance + total_inventory_notional
        return PortfolioSnapshot(
            total_equity=total_equity,
            total_quote_balance=total_quote_balance,
            total_reserved_quote=total_reserved_quote,
            total_available_quote=total_available_quote,
            total_inventory_notional=total_inventory_notional,
            total_outstanding_buy_notional=total_outstanding_buy_notional,
            symbols=symbols,
        )

    def allocate(self, snapshot: PortfolioSnapshot) -> PortfolioAllocationPlan:
        """Distribute one portfolio-level new-entry budget across eligible symbols."""
        if not snapshot.symbols:
            return PortfolioAllocationPlan(budgets=[], total_allocatable_quote=0.0, total_allocated_quote=0.0)

        config = self.config
        total_allocatable_quote = max(
            snapshot.total_available_quote * config.risk.global_max_new_entry_pct_of_free_quote
            - snapshot.total_outstanding_buy_notional,
            0.0,
        )
        portfolio_inventory_cap = snapshot.total_equity * config.risk.global_max_portfolio_inventory_pct_of_equity
        remaining_portfolio_inventory_room = max(portfolio_inventory_cap - snapshot.total_inventory_notional, 0.0)
        total_allocatable_quote = min(total_allocatable_quote, remaining_portfolio_inventory_room)

        weighted_inputs = []
        budgets: list[SymbolAllocationBudget] = []
        for symbol_input in snapshot.symbols:
            eligible = self._is_entry_eligible(symbol_input)
            if not eligible:
                budgets.append(
                    SymbolAllocationBudget(
                        symbol=symbol_input.symbol,
                        portfolio_budget=0.0,
                        eligible=False,
                        weight=0.0,
                        reasons=["entries_not_eligible"],
                    )
                )
                continue
            weight = self._score_symbol(symbol_input, snapshot, config)
            if weight <= 0 or not isfinite(weight):
                budgets.append(
                    SymbolAllocationBudget(
                        symbol=symbol_input.symbol,
                        portfolio_budget=0.0,
                        eligible=False,
                        weight=0.0,
                        reasons=["non_positive_weight"],
                    )
                )
                continue
            weighted_inputs.append((symbol_input, weight))

        weighted_inputs.sort(key=lambda item: item[1], reverse=True)
        selected_inputs = weighted_inputs[: config.risk.max_concurrent_entry_symbols]
        selected_symbols = {symbol_input.symbol for symbol_input, _ in selected_inputs}
        total_weight = sum(weight for _, weight in selected_inputs)

        for symbol_input, weight in selected_inputs:
            share = total_allocatable_quote * (weight / total_weight) if total_weight > 0 else 0.0
            budgets.append(
                SymbolAllocationBudget(
                    symbol=symbol_input.symbol,
                    portfolio_budget=max(share, 0.0),
                    eligible=True,
                    weight=weight,
                    reasons=["weighted_allocation"],
                )
            )

        for symbol_input, weight in weighted_inputs[config.risk.max_concurrent_entry_symbols :]:
            budgets.append(
                SymbolAllocationBudget(
                    symbol=symbol_input.symbol,
                    portfolio_budget=0.0,
                    eligible=False,
                    weight=weight,
                    reasons=["concurrency_limit"],
                )
            )

        for symbol_input in snapshot.symbols:
            if symbol_input.symbol not in {budget.symbol for budget in budgets}:
                budgets.append(
                    SymbolAllocationBudget(
                        symbol=symbol_input.symbol,
                        portfolio_budget=0.0,
                        eligible=False,
                        weight=0.0,
                        reasons=["entries_not_eligible"],
                    )
                )

        total_allocated_quote = sum((budget.portfolio_budget or 0.0) for budget in budgets)
        budgets.sort(key=lambda budget: budget.symbol)
        return PortfolioAllocationPlan(
            budgets=budgets,
            total_allocatable_quote=total_allocatable_quote,
            total_allocated_quote=total_allocated_quote,
        )

    def _is_entry_eligible(self, symbol_input: SymbolAllocationInput) -> bool:
        """Return whether a symbol is eligible for fresh entry budget allocation."""
        return (
            symbol_input.regime in {RegimeType.RANGE, RegimeType.UPTREND}
            and not symbol_input.pause_entries
            and not symbol_input.force_risk_off
        )

    def _score_symbol(self, symbol_input: SymbolAllocationInput, snapshot: PortfolioSnapshot, config) -> float:
        """Return a simple rules-based portfolio allocation weight for one symbol."""
        base_weight = (
            config.risk.allocation_weight_uptrend
            if symbol_input.regime == RegimeType.UPTREND
            else config.risk.allocation_weight_range
        )
        confidence_multiplier = max(symbol_input.confidence, 0.25)
        inventory_pressure = 1.0
        if snapshot.total_equity > 0:
            symbol_inventory_ratio = symbol_input.inventory_notional / max(snapshot.total_equity, 1e-9)
            inventory_pressure = max(0.25, 1.0 - symbol_inventory_ratio / max(config.risk.max_symbol_inventory_pct_of_equity, 1e-9))
        outstanding_pressure = 0.5 if symbol_input.outstanding_buy_notional > 0 else 1.0
        projected_quote_pressure = 1.0
        if snapshot.total_available_quote > 0 and symbol_input.projected_quote_usage > 0:
            usage_ratio = symbol_input.projected_quote_usage / max(snapshot.total_available_quote, 1e-9)
            projected_quote_pressure = max(0.35, 1.0 - usage_ratio)
        underwater_penalty = 1.0
        if symbol_input.underwater_ratio > 0:
            if symbol_input.regime == RegimeType.RANGE:
                underwater_penalty = max(0.35, 1.0 - symbol_input.underwater_ratio * 4.0)
            elif symbol_input.regime == RegimeType.UPTREND:
                underwater_penalty = max(0.55, 1.0 - symbol_input.underwater_ratio * 2.0)
        return base_weight * confidence_multiplier * inventory_pressure * outstanding_pressure * projected_quote_pressure * underwater_penalty


def _underwater_ratio(inventory) -> float:
    """Return the current drawdown ratio of inventory versus its cost basis."""
    if inventory.base_balance <= 0 or not inventory.cost_basis_price or inventory.cost_basis_price <= 0:
        return 0.0
    if inventory.mark_price >= inventory.cost_basis_price:
        return 0.0
    return (inventory.cost_basis_price - inventory.mark_price) / inventory.cost_basis_price
