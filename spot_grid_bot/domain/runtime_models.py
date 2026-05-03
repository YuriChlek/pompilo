from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from domain.market_models import RegimeType
from domain.risk_models import RiskRuntimeState

PERSISTED_RUNTIME_STATE_FIELDS = (
    "symbol",
    "regime",
    "bars_in_state",
    "cooldown_remaining",
    "volatility_cooldown_remaining",
    "pending_regime",
    "pending_count",
    "last_rebuild_price",
    "kill_switch_count",
    "cost_basis_price",
    "recent_equity",
    "state_version",
    "last_cycle_started_at",
    "last_cycle_completed_at",
    "last_successful_execution_at",
    "last_execution_status",
    "last_known_base_balance",
    "last_known_quote_balance",
    "last_known_reserved_quote",
    "last_known_mark_price",
)


@dataclass(slots=True)
class StrategyState:
    """Mutable per-symbol strategy state persisted across trading cycles."""

    regime: RegimeType
    bars_in_state: int = 0
    cooldown_remaining: int = 0
    volatility_cooldown_remaining: int = 0
    pending_regime: Optional[RegimeType] = None
    pending_count: int = 0
    last_rebuild_price: Optional[float] = None


@dataclass(slots=True)
class SymbolRuntimeState:
    """Persisted runtime snapshot combining all restart-critical state for one symbol."""

    symbol: str
    strategy_state: StrategyState
    risk_state: RiskRuntimeState = field(default_factory=RiskRuntimeState)
    cost_basis_price: Optional[float] = None
    state_version: int = 1
    last_cycle_started_at: Optional[datetime] = None
    last_cycle_completed_at: Optional[datetime] = None
    last_successful_execution_at: Optional[datetime] = None
    last_execution_status: Optional[str] = None
    last_known_base_balance: Optional[float] = None
    last_known_quote_balance: Optional[float] = None
    last_known_reserved_quote: Optional[float] = None
    last_known_mark_price: Optional[float] = None
