from __future__ import annotations

from domain.live_order_policy import has_live_entry_orders
from domain.models import LiveOrder, RegimeType, RiskDecision, StrategyState, TargetOrder


def is_protective_regime(regime: RegimeType) -> bool:
    """Return whether the confirmed regime should cancel outstanding entry buys."""
    return regime in {RegimeType.DOWNTREND, RegimeType.HIGH_VOLATILITY, RegimeType.RISK_OFF}


def should_rebuild(
    state: StrategyState,
    price: float,
    atr14: float,
    live_orders: list[LiveOrder],
    target_orders: list[TargetOrder],
    risk: RiskDecision,
    diff_count: int,
    rebuild_price_deviation_pct: float,
    diff_count_threshold: int,
) -> tuple[bool, list[str]]:
    """Decide whether the current target set requires a live rebuild."""
    reasons: list[str] = []
    if risk.force_risk_off:
        reasons.append("risk_off")
        return True, reasons
    if is_protective_regime(state.regime) and has_live_entry_orders(live_orders):
        reasons.append("protective_regime_cancel_entries")
        return True, reasons
    if not live_orders:
        reasons.append("no_live_orders")
        return True, reasons
    if not target_orders and live_orders:
        reasons.append("target_orders_empty")
        return True, reasons
    if state.bars_in_state == 0:
        reasons.append("state_transition")
        return True, reasons
    if state.last_rebuild_price is None or state.last_rebuild_price <= 0:
        reasons.append("missing_last_rebuild_price")
        return True, reasons

    effective_diff_threshold = max(diff_count_threshold, 1)
    if diff_count > effective_diff_threshold:
        reasons.append(f"target_diff={diff_count}")
        return True, reasons

    deviation = abs(price - state.last_rebuild_price) / state.last_rebuild_price
    atr_threshold = (atr14 / price) * 0.20 if price > 0 and atr14 > 0 else 0.0
    effective_threshold = max(rebuild_price_deviation_pct, atr_threshold)
    if deviation >= effective_threshold:
        reasons.append("price_deviation")
        return True, reasons
    return False, reasons
