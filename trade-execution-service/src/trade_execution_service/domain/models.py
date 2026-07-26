from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Mapping

from trade_execution_service.domain.enums import (
    ExchangeId,
    ExecutionDecisionStatus,
    ExecutionIntentType,
    OrderSide,
    OrderType,
)

JsonMapping = Mapping[str, object]


@dataclass(frozen=True, slots=True)
class BotSignalPersistedEvent:
    """Metadata-only event emitted after Bot Platform persists a bot signal."""

    event_type: str
    signal_id: str
    signal_key: str | None = None
    correlation_id: str | None = None


@dataclass(frozen=True, slots=True)
class PersistedBotSignalRef:
    """Reference to a signal persisted by Bot Platform."""

    signal_id: str
    signal_key: str
    instance_id: str
    module_id: str
    symbol: str
    timeframe: str
    snapshot_id: str
    payload_schema: str
    payload_schema_version: int
    payload_hash: str


@dataclass(frozen=True, slots=True)
class ExecutionAccountRef:
    """User/tenant exchange account selected for execution."""

    tenant_id: str
    user_id: str
    account_id: str
    exchange_id: ExchangeId
    secret_ref: str


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    """Normalized live or persisted position view."""

    account_id: str
    exchange_id: ExchangeId
    symbol: str
    base_quantity: Decimal
    average_entry_price: Decimal | None
    quote_notional: Decimal | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class BalanceSnapshot:
    """Normalized balance view for one asset."""

    account_id: str
    exchange_id: ExchangeId
    asset: str
    available: Decimal
    total: Decimal


@dataclass(frozen=True, slots=True)
class VenueConstraints:
    """Venue-specific normalized trading constraints."""

    exchange_id: ExchangeId
    symbol: str
    min_quantity: Decimal
    min_notional: Decimal
    quantity_step: Decimal
    price_tick: Decimal


@dataclass(frozen=True, slots=True)
class ExecutionIntent:
    """Execution-neutral intent after validating a bot signal payload."""

    signal: PersistedBotSignalRef
    intent_type: ExecutionIntentType
    symbol: str
    side: OrderSide | None
    order_type: OrderType | None
    target_price: Decimal | None
    suggested_quote_notional: Decimal | None
    max_quote_notional: Decimal | None
    reason_codes: tuple[str, ...] = ()
    payload: JsonMapping = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExecutionDecision:
    """Result of execution validation and optional venue action."""

    signal_id: str
    status: ExecutionDecisionStatus
    reason: str
    exchange_id: ExchangeId | None = None
    venue_order_id: str | None = None
    filled_base_quantity: Decimal | None = None
    average_fill_price: Decimal | None = None
