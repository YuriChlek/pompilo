from __future__ import annotations

from math import isfinite

from domain.strategy_config import StrategyConfig, DEFAULT_STRATEGY_CONFIG
from domain.uptrend_policy import build_underwater_recovery_profile
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
            recovery_profile = build_underwater_recovery_profile(
                inventory,
                analysis.strategy_state.regime,
                self.config,
                bars_in_state=analysis.strategy_state.bars_in_state,
            )
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
                    atr_pct=_atr_pct(analysis.indicators, inventory),
                    reasons=list(analysis.risk.reasons),
                    bars_in_state=analysis.strategy_state.bars_in_state,
                    recovery_active=recovery_profile.active,
                    recovery_budget_fraction=recovery_profile.recovery_budget if recovery_profile.active else 0.0,
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
        """Distribute portfolio-level new-entry and recovery budgets across symbols."""
        if not snapshot.symbols:
            return PortfolioAllocationPlan(
                budgets=[],
                total_allocatable_quote=0.0,
                total_allocated_quote=0.0,
                total_recovery_allocated_quote=0.0,
                total_entry_allocated_quote=0.0,
            )

        config = self.config
        total_allocatable_quote = max(
            snapshot.total_available_quote * config.risk.global_max_new_entry_pct_of_free_quote
            - snapshot.total_outstanding_buy_notional,
            0.0,
        )
        portfolio_inventory_cap = snapshot.total_equity * config.risk.global_max_portfolio_inventory_pct_of_equity
        remaining_portfolio_inventory_room = max(portfolio_inventory_cap - snapshot.total_inventory_notional, 0.0)
        total_allocatable_quote = min(total_allocatable_quote, remaining_portfolio_inventory_room)

        recovery_quota_fraction = min(max(config.risk.global_recovery_quota_fraction, 0.0), 1.0)
        initial_recovery_quota = total_allocatable_quote * recovery_quota_fraction
        recovery_allocations, unused_recovery_quota = self._allocate_recovery_budgets(snapshot, initial_recovery_quota)
        total_recovery_allocated_quote = sum(recovery_allocations.values())
        entry_quota = total_allocatable_quote - initial_recovery_quota + unused_recovery_quota

        weighted_inputs = []
        budgets: list[SymbolAllocationBudget] = []
        for symbol_input in snapshot.symbols:
            recovery_budget = recovery_allocations.get(symbol_input.symbol.upper(), 0.0)
            recovery_eligible = self._is_recovery_eligible(symbol_input)
            eligible = self._is_new_entry_eligible(symbol_input)
            if not eligible:
                budgets.append(
                    SymbolAllocationBudget(
                        symbol=symbol_input.symbol,
                        portfolio_budget=0.0,
                        eligible=False,
                        recovery_budget=recovery_budget,
                        recovery_eligible=recovery_eligible,
                        weight=0.0,
                        reasons=["entries_not_eligible"] if not recovery_eligible else ["recovery_only_symbol"],
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
                        recovery_budget=recovery_budget,
                        recovery_eligible=recovery_eligible,
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
            share = entry_quota * (weight / total_weight) if total_weight > 0 else 0.0
            budgets.append(
                SymbolAllocationBudget(
                    symbol=symbol_input.symbol,
                    portfolio_budget=max(share, 0.0),
                    eligible=True,
                    recovery_budget=recovery_allocations.get(symbol_input.symbol.upper(), 0.0),
                    recovery_eligible=self._is_recovery_eligible(symbol_input),
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
                    recovery_budget=recovery_allocations.get(symbol_input.symbol.upper(), 0.0),
                    recovery_eligible=self._is_recovery_eligible(symbol_input),
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
                        recovery_budget=recovery_allocations.get(symbol_input.symbol.upper(), 0.0),
                        recovery_eligible=self._is_recovery_eligible(symbol_input),
                        weight=0.0,
                        reasons=["entries_not_eligible"],
                    )
                )

        total_entry_allocated_quote = sum((budget.portfolio_budget or 0.0) for budget in budgets)
        total_allocated_quote = total_entry_allocated_quote + total_recovery_allocated_quote
        budgets.sort(key=lambda budget: budget.symbol)
        return PortfolioAllocationPlan(
            budgets=budgets,
            total_allocatable_quote=total_allocatable_quote,
            total_allocated_quote=total_allocated_quote,
            total_recovery_allocated_quote=total_recovery_allocated_quote,
            total_entry_allocated_quote=total_entry_allocated_quote,
        )

    def _is_entry_eligible(self, symbol_input: SymbolAllocationInput) -> bool:
        """Return whether a symbol is eligible for fresh entry budget allocation."""
        return (
            symbol_input.regime in {RegimeType.RANGE, RegimeType.UPTREND}
            and not symbol_input.pause_entries
            and not symbol_input.force_risk_off
        )

    def _is_new_entry_eligible(self, symbol_input: SymbolAllocationInput) -> bool:
        """Return whether a symbol should draw from the fresh-entry pool."""
        return self._is_entry_eligible(symbol_input) and not symbol_input.recovery_active

    def _is_recovery_eligible(self, symbol_input: SymbolAllocationInput) -> bool:
        """Return whether a symbol should draw from the portfolio recovery pool."""
        return self._is_entry_eligible(symbol_input) and symbol_input.recovery_active

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
        volatility_multiplier = self._atr_normalized_multiplier(symbol_input, config)
        return (
            base_weight
            * confidence_multiplier
            * inventory_pressure
            * outstanding_pressure
            * projected_quote_pressure
            * underwater_penalty
            * volatility_multiplier
        )

    def _allocate_recovery_budgets(
        self,
        snapshot: PortfolioSnapshot,
        total_recovery_quota: float,
    ) -> tuple[dict[str, float], float]:
        """Allocate the portfolio recovery pool and return any unused quota."""
        allocations = {symbol_input.symbol.upper(): 0.0 for symbol_input in snapshot.symbols}
        if total_recovery_quota <= 0:
            return allocations, 0.0

        eligible_inputs = [symbol_input for symbol_input in snapshot.symbols if self._is_recovery_eligible(symbol_input)]
        if not eligible_inputs:
            return allocations, total_recovery_quota

        demand_by_symbol = {
            symbol_input.symbol.upper(): max(symbol_input.available_quote * symbol_input.recovery_budget_fraction, 0.0)
            for symbol_input in eligible_inputs
        }
        active_inputs = [
            symbol_input
            for symbol_input in eligible_inputs
            if demand_by_symbol[symbol_input.symbol.upper()] >= self.config.risk.min_symbol_entry_notional
        ]
        if not active_inputs:
            return allocations, total_recovery_quota

        remaining_quota = total_recovery_quota
        remaining_demand = dict(demand_by_symbol)
        while remaining_quota > 0 and active_inputs:
            weighted_inputs = [
                (symbol_input, self._score_recovery_symbol(symbol_input))
                for symbol_input in active_inputs
            ]
            weighted_inputs = [(symbol_input, weight) for symbol_input, weight in weighted_inputs if weight > 0 and isfinite(weight)]
            if not weighted_inputs:
                break
            total_weight = sum(weight for _, weight in weighted_inputs)
            if total_weight <= 0:
                break

            consumed = 0.0
            next_active_inputs: list[SymbolAllocationInput] = []
            for symbol_input, weight in weighted_inputs:
                symbol = symbol_input.symbol.upper()
                share = remaining_quota * (weight / total_weight)
                grant = min(share, remaining_demand[symbol])
                allocations[symbol] += grant
                remaining_demand[symbol] -= grant
                consumed += grant
                if remaining_demand[symbol] >= self.config.risk.min_symbol_entry_notional:
                    next_active_inputs.append(symbol_input)

            if consumed <= 0:
                break
            remaining_quota = max(remaining_quota - consumed, 0.0)
            active_inputs = next_active_inputs

        for symbol, amount in list(allocations.items()):
            if 0 < amount < self.config.risk.min_symbol_entry_notional:
                remaining_quota += amount
                allocations[symbol] = 0.0

        return allocations, remaining_quota

    def _score_recovery_symbol(self, symbol_input: SymbolAllocationInput) -> float:
        """Return a recovery-pool allocation weight for one underwater symbol."""
        regime_multiplier = 1.0 if symbol_input.regime == RegimeType.UPTREND else 0.85
        confidence_multiplier = max(symbol_input.confidence, 0.25)
        underwater_pressure = max(
            symbol_input.underwater_ratio,
            self.config.grid.underwater_averaging_trigger_pct,
        )
        volatility_multiplier = self._atr_normalized_multiplier(symbol_input, self.config)
        return regime_multiplier * confidence_multiplier * underwater_pressure * volatility_multiplier

    def _atr_normalized_multiplier(self, symbol_input: SymbolAllocationInput, config) -> float:
        """Return a clamped ATR-normalized allocation multiplier for one symbol."""
        if symbol_input.atr_pct <= 0:
            return 1.0
        target_atr_pct = max(config.risk.allocation_target_atr_pct, 1e-9)
        raw_multiplier = target_atr_pct / symbol_input.atr_pct
        return min(
            max(raw_multiplier, config.risk.allocation_atr_multiplier_min),
            config.risk.allocation_atr_multiplier_max,
        )


def _underwater_ratio(inventory) -> float:
    """Return the current drawdown ratio of inventory versus its cost basis."""
    if inventory.base_balance <= 0 or not inventory.cost_basis_price or inventory.cost_basis_price <= 0:
        return 0.0
    if inventory.mark_price >= inventory.cost_basis_price:
        return 0.0
    return (inventory.cost_basis_price - inventory.mark_price) / inventory.cost_basis_price


def _atr_pct(indicators, inventory) -> float:
    """Return ATR as a fraction of mark price for allocator-side volatility normalization."""
    if inventory.mark_price <= 0:
        return 0.0
    return max(indicators.atr14, 0.0) / inventory.mark_price
