from __future__ import annotations

from dataclasses import replace

from domain.indicators import compute_snapshot
from domain.models import MarketContext, PreliminarySymbolAnalysis, RiskRuntimeState, SymbolRuntimeState
from domain.state_machine import StrategyStateMachine


def analyze_symbol(
    context: MarketContext,
    runtime: SymbolRuntimeState,
    *,
    config,
    detector,
    risk_manager,
) -> PreliminarySymbolAnalysis:
    """Build a non-committing preliminary analysis for one symbol market context."""
    candles = context.candles
    inventory = context.inventory
    live_orders = context.live_orders
    indicators = compute_snapshot(candles, config)
    regime_snapshot = detector.detect(candles, indicators, risk_off=False)
    risk_runtime = RiskRuntimeState(
        kill_switch_count=runtime.risk_state.kill_switch_count,
        recent_equity=list(runtime.risk_state.recent_equity),
    )
    risk = risk_manager.evaluate(
        candles,
        indicators,
        inventory,
        live_orders,
        risk_runtime,
        runtime.strategy_state.regime,
        runtime.strategy_state.volatility_cooldown_remaining,
    )
    if risk.force_risk_off:
        regime_snapshot = detector.detect(candles, indicators, risk_off=True)
    preview_state = StrategyStateMachine(config, replace(runtime.strategy_state)).on_bar(regime_snapshot)
    return PreliminarySymbolAnalysis(
        symbol=context.symbol.upper(),
        indicators=indicators,
        regime_snapshot=regime_snapshot,
        risk=risk,
        strategy_state=replace(preview_state),
        risk_state=RiskRuntimeState(
            kill_switch_count=risk_runtime.kill_switch_count,
            recent_equity=list(risk_runtime.recent_equity),
        ),
    )
