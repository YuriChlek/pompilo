from __future__ import annotations

from dataclasses import dataclass, field

from domain.market_models import IndicatorSnapshot, RegimeType
from domain.order_models import LiveOrder, TargetOrder
from domain.risk_models import RiskDecision


@dataclass(slots=True, frozen=True)
class StrategyDecision:
    """Planner output describing the next desired order state for one symbol."""

    symbol: str
    regime: RegimeType
    target_orders: list[TargetOrder]
    live_orders: list[LiveOrder]
    indicators: IndicatorSnapshot
    risk: RiskDecision
    rebuild_required: bool
    target_order_diff_count: int = 0
    reasons: list[str] = field(default_factory=list)
