from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class InventorySnapshot:
    """Current base and quote balances with derived inventory metrics."""

    base_balance: float
    quote_balance: float
    reserved_quote: float
    mark_price: float
    cost_basis_price: Optional[float] = None

    @property
    def inventory_notional(self) -> float:
        """Return current inventory market value using the latest mark price."""
        return self.base_balance * self.mark_price

    @property
    def total_equity(self) -> float:
        """Return total equity as quote balance plus marked-to-market inventory value."""
        return self.quote_balance + self.inventory_notional

    @property
    def available_quote(self) -> float:
        """Return quote balance still available after reserved funds are excluded."""
        return max(self.quote_balance - self.reserved_quote, 0.0)

