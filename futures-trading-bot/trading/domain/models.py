from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional


@dataclass(frozen=True)
class TradeSignal(Mapping[str, Any]):
    """Typed domain model for a generated trade signal with dict-like compatibility."""

    time: Any
    symbol: str
    strategy_mode: str
    order_type: str
    direction: str
    price: Decimal
    size: Any
    take_profit: Decimal
    stop_loss: Decimal
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "time": self.time,
            "symbol": self.symbol,
            "strategy_mode": self.strategy_mode,
            "order_type": self.order_type,
            "direction": self.direction,
            "price": self.price,
            "size": self.size,
            "take_profit": self.take_profit,
            "stop_loss": self.stop_loss,
        }
        payload.update(self.metadata)
        return payload

    def __getitem__(self, key: str) -> Any:
        return self.to_payload()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_payload())

    def __len__(self) -> int:
        return len(self.to_payload())

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_payload().get(key, default)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return self.to_payload() == dict(other)
        return False


@dataclass(frozen=True)
class StopLossUpdate(Mapping[str, Any]):
    """Typed domain model for a live stop-management update."""

    symbol: str
    direction: str
    entry_price: Decimal
    current_price: Decimal
    stop_loss: Decimal
    take_profit: Optional[Decimal]
    position_idx: int
    update_type: str
    partial_close_qty: Optional[Decimal] = None
    partial_close_side: Optional[str] = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_idx": self.position_idx,
            "update_type": self.update_type,
            "partial_close_qty": self.partial_close_qty,
            "partial_close_side": self.partial_close_side,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_payload()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_payload())

    def __len__(self) -> int:
        return len(self.to_payload())

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_payload().get(key, default)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return self.to_payload() == dict(other)
        return False


@dataclass(frozen=True)
class MarketRegime:
    """Describe the current directional regime used by breakout filtering."""

    name: str
    direction: str
    is_breakout_enabled: bool
    is_high_vol: bool = False
    reason: str = ""


@dataclass(frozen=True)
class SignalContext:
    """Provide additional context for a generated signal or a skipped breakout."""

    regime: MarketRegime
    breakout_high: Optional[Decimal] = None
    breakout_low: Optional[Decimal] = None
    volume_spike_ratio: Optional[Decimal] = None
    funding_rate: Optional[Decimal] = None


@dataclass(frozen=True)
class ClusterExposure:
    """Describe the active exposure summary for one correlation cluster."""

    cluster: str
    active_positions: int
    heat_pct: Decimal


@dataclass(frozen=True)
class PortfolioRiskState:
    """Summarize the live portfolio state used for entry admission checks."""

    active_positions: int
    portfolio_heat_pct: Decimal
    daily_realized_loss_r: Decimal
    cluster_exposures: dict[str, ClusterExposure] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionAdmissionDecision:
    """Return whether a new entry is allowed under portfolio-wide risk controls."""

    allowed: bool
    reason: str
    detail: str = ""
