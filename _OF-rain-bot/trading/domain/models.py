from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Mapping


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    NONE = "none"


@dataclass(frozen=True)
class LiquidityWall:
    exchange: str
    symbol: str
    side: str
    price: Decimal
    size: Decimal
    notional: Decimal
    distance_ticks: int
    distance_bps: Decimal
    first_seen_ms: int
    last_seen_ms: int
    persistence_ms: int
    relative_size_ratio: Decimal
    size_stability_score: Decimal
    pull_count: int
    test_count: int
    reload_count: int
    defended_count: int
    chase_count: int
    score: Decimal
    spoof_risk_score: Decimal
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScalpSignal:
    symbol: str
    direction: SignalDirection
    confidence: Decimal
    reason: str
    wall: LiquidityWall | None = None
    analysis_entry_price: Decimal | None = None
    analysis_stop_price: Decimal | None = None
    analysis_take_profit_price: Decimal | None = None
    analysis_invalidation_price: Decimal | None = None
    execution_entry_price: Decimal | None = None
    execution_stop_price: Decimal | None = None
    execution_take_profit_price: Decimal | None = None
    execution_invalidation_price: Decimal | None = None
    basis_bps: Decimal | None = None
    tape_bias: str = "neutral"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Any = None


@dataclass(frozen=True)
class PositionSnapshot:
    symbol: str
    side: str
    size: Decimal
    entry_price: Decimal
    tick_size: Decimal
    opened_at_ms: int
    stop_price: Decimal
    take_profit_price: Decimal
    best_price_seen: Decimal
    status: str = "open"


def to_domain_signal_direction(value: Any) -> SignalDirection:
    normalized = str(getattr(value, "value", value) or "").strip().lower()
    if normalized == SignalDirection.LONG.value:
        return SignalDirection.LONG
    if normalized == SignalDirection.SHORT.value:
        return SignalDirection.SHORT
    return SignalDirection.NONE


def to_domain_liquidity_wall(wall: Any) -> LiquidityWall | None:
    if wall is None:
        return None
    return LiquidityWall(
        exchange=str(wall.exchange),
        symbol=str(wall.symbol),
        side=str(wall.side),
        price=_to_decimal(wall.price),
        size=_to_decimal(wall.size),
        notional=_to_decimal(wall.notional),
        distance_ticks=int(wall.distance_ticks),
        distance_bps=_to_decimal(wall.distance_bps),
        first_seen_ms=int(wall.first_seen_ms),
        last_seen_ms=int(wall.last_seen_ms),
        persistence_ms=int(wall.persistence_ms),
        relative_size_ratio=_to_decimal(wall.relative_size_ratio),
        size_stability_score=_to_decimal(wall.size_stability_score),
        pull_count=int(wall.pull_count),
        test_count=int(wall.test_count),
        reload_count=int(wall.reload_count),
        defended_count=int(wall.defended_count),
        chase_count=int(wall.chase_count),
        score=_to_decimal(wall.score),
        spoof_risk_score=_to_decimal(wall.spoof_risk_score),
        metadata=dict(getattr(wall, "metadata", {}) or {}),
    )


def to_domain_scalp_signal(signal: Any) -> ScalpSignal:
    return ScalpSignal(
        symbol=str(signal.symbol),
        direction=to_domain_signal_direction(signal.direction),
        confidence=_to_decimal(signal.confidence),
        reason=str(signal.reason),
        wall=to_domain_liquidity_wall(getattr(signal, "wall", None)),
        analysis_entry_price=_to_optional_decimal(getattr(signal, "analysis_entry_price", None)),
        analysis_stop_price=_to_optional_decimal(getattr(signal, "analysis_stop_price", None)),
        analysis_take_profit_price=_to_optional_decimal(getattr(signal, "analysis_take_profit_price", None)),
        analysis_invalidation_price=_to_optional_decimal(getattr(signal, "analysis_invalidation_price", None)),
        execution_entry_price=_to_optional_decimal(getattr(signal, "execution_entry_price", None)),
        execution_stop_price=_to_optional_decimal(getattr(signal, "execution_stop_price", None)),
        execution_take_profit_price=_to_optional_decimal(getattr(signal, "execution_take_profit_price", None)),
        execution_invalidation_price=_to_optional_decimal(getattr(signal, "execution_invalidation_price", None)),
        basis_bps=_to_optional_decimal(getattr(signal, "basis_bps", None)),
        tape_bias=str(getattr(signal, "tape_bias", "neutral")),
        metadata=dict(getattr(signal, "metadata", {}) or {}),
        diagnostics=getattr(signal, "diagnostics", None),
    )


def to_domain_position_snapshot(position: Any) -> PositionSnapshot:
    return PositionSnapshot(
        symbol=str(position.symbol),
        side=str(position.side),
        size=_to_decimal(position.size),
        entry_price=_to_decimal(position.entry_price),
        tick_size=_to_decimal(position.tick_size),
        opened_at_ms=int(position.opened_at_ms),
        stop_price=_to_decimal(position.stop_price),
        take_profit_price=_to_decimal(position.take_profit_price),
        best_price_seen=_to_decimal(position.best_price_seen),
        status=str(getattr(position, "status", "open")),
    )


def _to_decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot convert {value!r} to Decimal") from exc


def _to_optional_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    return _to_decimal(value)
