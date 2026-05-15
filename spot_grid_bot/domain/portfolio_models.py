from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from domain.market_models import IndicatorSnapshot, RegimeSnapshot, RegimeType
from domain.risk_models import RiskDecision, RiskRuntimeState
from domain.runtime_models import StrategyState

if TYPE_CHECKING:
    from domain.market_structure import StructureSnapshot


@dataclass(slots=True, frozen=True)
class SymbolAllocationInput:
    """Portfolio-level allocation input derived from one symbol preliminary analysis."""

    symbol: str
    regime: RegimeType
    confidence: float
    inventory_notional: float
    mark_price: float
    cost_basis_price: Optional[float]
    underwater_ratio: float
    available_quote: float
    outstanding_buy_notional: float
    projected_inventory_notional: float
    projected_quote_usage: float
    pause_entries: bool
    force_risk_off: bool
    atr_pct: float = 0.0
    reasons: list[str] = field(default_factory=list)
    bars_in_state: int = 0
    recovery_active: bool = False
    recovery_budget_fraction: float = 0.0


@dataclass(slots=True, frozen=True)
class PortfolioSnapshot:
    """Portfolio-wide state used as allocator input for one multi-symbol cycle."""

    total_equity: float
    total_quote_balance: float
    total_reserved_quote: float
    total_available_quote: float
    total_inventory_notional: float
    total_outstanding_buy_notional: float
    symbols: list[SymbolAllocationInput]


@dataclass(slots=True, frozen=True)
class SymbolAllocationBudget:
    """Allocator output budget for one symbol before local symbol caps are applied."""

    symbol: str
    portfolio_budget: Optional[float]
    eligible: bool
    recovery_budget: Optional[float] = None
    recovery_eligible: bool = False
    weight: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class PortfolioAllocationPlan:
    """Portfolio-level budget plan keyed by symbol for one execution cycle."""

    budgets: list[SymbolAllocationBudget]
    total_allocatable_quote: float
    total_allocated_quote: float
    total_recovery_allocated_quote: float = 0.0
    total_entry_allocated_quote: float = 0.0

    def budget_for(self, symbol: str) -> Optional[float]:
        """Return the portfolio budget cap for one symbol if a cap is defined."""
        normalized_symbol = symbol.upper()
        for budget in self.budgets:
            if budget.symbol.upper() == normalized_symbol:
                return budget.portfolio_budget
        return None

    def recovery_budget_for(self, symbol: str) -> Optional[float]:
        """Return the portfolio recovery budget cap for one symbol if defined."""
        normalized_symbol = symbol.upper()
        for budget in self.budgets:
            if budget.symbol.upper() == normalized_symbol:
                return budget.recovery_budget
        return None


@dataclass(slots=True, frozen=True)
class PreliminarySymbolAnalysis:
    """Pre-planning analysis bundle used for portfolio-level allocation passes."""

    symbol: str
    indicators: IndicatorSnapshot
    regime_snapshot: RegimeSnapshot
    risk: RiskDecision
    strategy_state: StrategyState
    risk_state: RiskRuntimeState
    structure_snapshot: Optional[StructureSnapshot] = None
