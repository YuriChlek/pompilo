from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import logging

from domain.grid_builder import GridBuilder
from domain.inventory_manager import InventoryManager
from domain.live_order_policy import is_bot_managed_order_link_id, live_to_target
from domain.order_diff import target_orders_diff_count
from domain.models import (
    IndicatorSnapshot,
    MarketContext,
    OrderSide,
    PreliminarySymbolAnalysis,
    RegimeType,
    RiskDecision,
    RiskRuntimeState,
    SymbolRuntimeState,
    StrategyDecision,
    StrategyState,
    TargetOrder,
)
from domain.rebuild_policy import is_protective_regime, should_rebuild
from domain.regime_detector import MarketRegimeDetector
from domain.risk_manager import RiskManager
from domain.symbol_analyzer import analyze_symbol
from domain.strategy_config import StrategyConfig
from domain.target_order_builder import build_target_orders
logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class LivePriceReference:
    """Ephemeral per-symbol price reference used for off-cycle price deviation checks."""

    symbol: str
    cached_price: float
    atr14: float
    regime: RegimeType
    kill_switch_count: int


class SpotGridPlanner:
    """Build one-symbol trading decisions from market context and runtime state."""

    def __init__(self, config: StrategyConfig) -> None:
        """Store strategy collaborators and initialize per-symbol runtime storage."""
        self.config = config
        self.detector = MarketRegimeDetector(config)
        self.grid_builder = GridBuilder(config)
        self.inventory_manager = InventoryManager(config)
        self.risk_manager = RiskManager(config)
        self.regime_counter: Counter[RegimeType] = Counter()
        self._runtime_by_symbol: dict[str, SymbolRuntimeState] = {}
        self._live_price_references: dict[str, LivePriceReference] = {}

    def plan(self, context: MarketContext) -> StrategyDecision:
        """Generate the next strategy decision for one symbol market context."""
        analysis = self.analyze(context)
        return self.plan_from_analysis(context, analysis)

    def analyze(self, context: MarketContext) -> PreliminarySymbolAnalysis:
        """Build a non-committing preliminary analysis for one symbol market context."""
        runtime = self._ensure_runtime(context.symbol)
        analysis = analyze_symbol(
            context,
            runtime,
            config=self.config,
            detector=self.detector,
            risk_manager=self.risk_manager,
        )
        self._live_price_references[context.symbol.upper()] = LivePriceReference(
            symbol=context.symbol.upper(),
            cached_price=context.inventory.mark_price if context.inventory.mark_price > 0 else context.candles[-1].close,
            atr14=analysis.indicators.atr14,
            regime=analysis.strategy_state.regime,
            kill_switch_count=analysis.risk_state.kill_switch_count,
        )
        return analysis

    def plan_from_analysis(
        self,
        context: MarketContext,
        analysis: PreliminarySymbolAnalysis,
        *,
        portfolio_budget: float | None = None,
    ) -> StrategyDecision:
        """Commit a preliminary analysis and build a final strategy decision for one symbol."""
        runtime = self._ensure_runtime(context.symbol)
        runtime.strategy_state = replace(analysis.strategy_state)
        runtime.risk_state = RiskRuntimeState(
            kill_switch_count=analysis.risk_state.kill_switch_count,
            recent_equity=list(analysis.risk_state.recent_equity),
        )

        state = runtime.strategy_state
        indicators = analysis.indicators
        risk = analysis.risk
        self.regime_counter[state.regime] += 1
        target_orders = build_target_orders(
            price=context.candles[-1].close,
            indicators=indicators,
            inventory=context.inventory,
            regime=state.regime,
            risk=risk,
            symbol=context.symbol,
            config=self.config,
            grid_builder=self.grid_builder,
            inventory_manager=self.inventory_manager,
            tick_size=context.venue_constraints.tick_size if context.venue_constraints is not None else None,
            venue_constraints=context.venue_constraints,
            portfolio_budget=portfolio_budget,
        )
        diff_count = target_orders_diff_count(
            context.live_orders,
            target_orders,
            price_diff_bps=self.config.execution.target_price_diff_bps,
            size_diff_ratio=self.config.execution.target_size_diff_ratio,
            tick_size=context.venue_constraints.tick_size if context.venue_constraints is not None else 0.0,
        )

        rebuild_required, rebuild_reasons = should_rebuild(
            state,
            context.candles[-1].close,
            indicators.atr14,
            context.live_orders,
            target_orders,
            risk,
            diff_count,
            self.config.execution.rebuild_price_deviation_pct,
            self.config.execution.rebuild_diff_threshold,
        )
        if not rebuild_required:
            return StrategyDecision(
                symbol=context.symbol,
                regime=state.regime,
                target_orders=live_to_target(context.live_orders),
                live_orders=context.live_orders,
                indicators=indicators,
                risk=risk,
                rebuild_required=False,
                target_order_diff_count=diff_count,
                kill_switch_count=analysis.risk_state.kill_switch_count,
                reasons=["rebuild_not_required"],
            )

        state.last_rebuild_price = context.candles[-1].close
        reasons = [f"regime={state.regime.value.lower()}"] + rebuild_reasons + list(risk.reasons)
        return StrategyDecision(
            symbol=context.symbol,
            regime=state.regime,
            target_orders=target_orders,
            live_orders=context.live_orders,
            indicators=indicators,
            risk=risk,
            rebuild_required=True,
            target_order_diff_count=diff_count,
            kill_switch_count=analysis.risk_state.kill_switch_count,
            reasons=reasons,
        )

    def restore_symbol_runtime(self, runtime_state: SymbolRuntimeState) -> None:
        """Restore a persisted runtime snapshot for one symbol into planner memory."""
        self._runtime_by_symbol[runtime_state.symbol.upper()] = runtime_state

    def get_persisted_cost_basis(self, symbol: str) -> float | None:
        """Return the last known persisted cost basis from in-memory runtime state."""
        runtime = self._runtime_by_symbol.get(symbol.upper())
        return runtime.cost_basis_price if runtime is not None else None

    def update_cost_basis(self, symbol: str, cost_basis_price: float | None) -> None:
        """Update runtime cost basis from the latest exchange snapshot or clear it when absent."""
        runtime = self._ensure_runtime(symbol)
        runtime.cost_basis_price = cost_basis_price if cost_basis_price and cost_basis_price > 0 else None

    def update_inventory_snapshot(self, symbol: str, inventory) -> None:
        """Store the latest live inventory snapshot for restart diagnostics and recovery."""
        runtime = self._ensure_runtime(symbol)
        runtime.last_known_base_balance = inventory.base_balance
        runtime.last_known_quote_balance = inventory.quote_balance
        runtime.last_known_reserved_quote = inventory.reserved_quote
        runtime.last_known_mark_price = inventory.mark_price

    def mark_cycle_started(self, symbol: str) -> None:
        """Record the start time of the latest per-symbol trading cycle."""
        runtime = self._ensure_runtime(symbol)
        runtime.last_cycle_started_at = datetime.now(timezone.utc)

    def mark_cycle_completed(self, symbol: str, *, status: str, successful_execution: bool = False) -> None:
        """Record the latest per-symbol cycle completion status."""
        runtime = self._ensure_runtime(symbol)
        now = datetime.now(timezone.utc)
        runtime.last_cycle_completed_at = now
        runtime.last_execution_status = status
        if successful_execution:
            runtime.last_successful_execution_at = now

    def export_symbol_runtime(self, symbol: str) -> SymbolRuntimeState:
        """Export a detached runtime snapshot suitable for persistence."""
        runtime = self._ensure_runtime(symbol)
        return SymbolRuntimeState(
            symbol=symbol.upper(),
            strategy_state=replace(runtime.strategy_state),
            risk_state=RiskRuntimeState(
                kill_switch_count=runtime.risk_state.kill_switch_count,
                recent_equity=list(runtime.risk_state.recent_equity),
            ),
            cost_basis_price=runtime.cost_basis_price,
            state_version=runtime.state_version,
            last_cycle_started_at=runtime.last_cycle_started_at,
            last_cycle_completed_at=runtime.last_cycle_completed_at,
            last_successful_execution_at=runtime.last_successful_execution_at,
            last_execution_status=runtime.last_execution_status,
            last_known_base_balance=runtime.last_known_base_balance,
            last_known_quote_balance=runtime.last_known_quote_balance,
            last_known_reserved_quote=runtime.last_known_reserved_quote,
            last_known_mark_price=runtime.last_known_mark_price,
        )

    def get_total_kill_switch_count(self) -> int:
        """Return the accumulated kill-switch count across all planner symbols."""
        return sum(runtime.risk_state.kill_switch_count for runtime in self._runtime_by_symbol.values())

    def get_live_price_reference(self, symbol: str) -> LivePriceReference | None:
        """Return the latest cached mark-price and ATR reference for one symbol."""
        return self._live_price_references.get(symbol.upper())

    def _ensure_runtime(self, symbol: str) -> SymbolRuntimeState:
        """Return the mutable in-memory runtime state for one symbol."""
        key = symbol.upper()
        runtime = self._runtime_by_symbol.get(key)
        if runtime is None:
            runtime = SymbolRuntimeState(
                symbol=key,
                strategy_state=StrategyState(regime=RegimeType.RANGE),
                risk_state=RiskRuntimeState(),
            )
            self._runtime_by_symbol[key] = runtime
        return runtime
