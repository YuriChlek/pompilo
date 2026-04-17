from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DeRiskMode(str, Enum):
    """Severity level for protective inventory reduction logic."""

    NONE = "NONE"
    SOFT = "SOFT"
    HARD = "HARD"
    PANIC = "PANIC"


@dataclass(slots=True, frozen=True)
class RiskDecision:
    """Risk outcome for one cycle including protection flags and staged de-risk mode."""

    can_trade: bool
    pause_entries: bool
    force_risk_off: bool
    cancel_entries: bool
    allow_exit_only: bool
    de_risk_mode: DeRiskMode = DeRiskMode.NONE
    outstanding_buy_notional: float = 0.0
    projected_inventory_notional: float = 0.0
    projected_quote_usage: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RiskRuntimeState:
    """Mutable per-symbol risk memory persisted between trading cycles."""

    kill_switch_count: int = 0
    recent_equity: list[float] = field(default_factory=list)

