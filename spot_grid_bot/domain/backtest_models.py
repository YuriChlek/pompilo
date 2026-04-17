from __future__ import annotations

from dataclasses import dataclass

from domain.inventory_models import InventorySnapshot
from domain.market_models import RegimeType


@dataclass(slots=True, frozen=True)
class BacktestResult:
    """Historical simulation result with PnL, diagnostics, and final runtime state."""

    equity_curve: list[float]
    pnl: float
    max_drawdown: float
    trade_count: int
    realized_pnl: float
    unrealized_pnl: float
    rebuild_count: int
    average_inventory_utilization: float
    de_risk_event_count: int
    blocked_no_loss_sell_count: int
    risk_reason_counts: dict[str, int]
    final_inventory: InventorySnapshot
    regime_statistics: dict[RegimeType, int]
    kill_switch_count: int
