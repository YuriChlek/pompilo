from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from domain.market_models import RegimeType
from domain.risk_models import RiskRuntimeState


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
    """Persisted runtime snapshot combining strategy and risk state for one symbol."""

    symbol: str
    strategy_state: StrategyState
    risk_state: RiskRuntimeState = field(default_factory=RiskRuntimeState)
    cost_basis_price: Optional[float] = None
