from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from domain.market_models import RegimeType


class OrderSide(str, Enum):
    """Normalized order side for target and live order models."""

    BUY = "BUY"
    SELL = "SELL"


@dataclass(slots=True)
class GridLevel:
    """Single planned price level inside a grid specification."""

    price: float
    size: float
    side: OrderSide
    level_index: int
    notional: float
    tag: str


@dataclass(slots=True)
class GridSpec:
    """Regime-specific grid geometry before execution sizing and filtering."""

    regime: RegimeType
    reference_price: float
    lower_bound: float
    upper_bound: float
    step: float
    levels: list[GridLevel]


@dataclass(slots=True, frozen=True)
class TargetOrder:
    """Planned order payload to be synchronized to the execution venue."""

    client_order_id: str
    symbol: str
    side: OrderSide
    price: float
    size: float
    reduce_only: bool = False
    tag: str = ""


@dataclass(slots=True, frozen=True)
class LiveOrder:
    """Normalized live exchange order snapshot used for diffing and reconciliation."""

    order_id: str
    symbol: str
    side: OrderSide
    price: float
    size: float
    filled_size: float
    status: str
    client_order_id: str = ""

