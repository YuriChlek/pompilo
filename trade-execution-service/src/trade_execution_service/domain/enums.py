from __future__ import annotations

from enum import StrEnum


class ExchangeId(StrEnum):
    """Supported exchange identifiers."""

    BYBIT = "bybit"
    BINANCE = "binance"
    OKX = "okx"


class ExecutionIntentType(StrEnum):
    """Execution-neutral intent categories consumed from bot signals."""

    OPEN_POSITION = "open_position"
    CLOSE_POSITION = "close_position"
    REDUCE_RISK = "reduce_risk"
    HOLD = "hold"
    ALERT = "alert"


class OrderSide(StrEnum):
    """Normalized order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    """Normalized order type."""

    MARKET = "market"
    LIMIT = "limit"


class ExecutionDecisionStatus(StrEnum):
    """Decision status before or after venue execution."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PLACED = "placed"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"
    SKIPPED = "skipped"
