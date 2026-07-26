from __future__ import annotations

from sqlalchemy import Column, DateTime, Integer, Table, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB

from market_data_service.persistence.tables.metadata import metadata

outbox_events = Table(
    "outbox_events",
    metadata,
    Column("id", Text, primary_key=True),
    Column("event_type", Text, nullable=False),
    Column("aggregate_type", Text, nullable=False),
    Column("aggregate_id", Text, nullable=False),
    Column("payload_json", JSONB, nullable=False),
    Column("idempotency_key", Text, nullable=False),
    Column("status", Text, nullable=False),
    Column("attempts", Integer, nullable=False, server_default="0"),
    Column("next_attempt_at", DateTime(timezone=True), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("published_at", DateTime(timezone=True), nullable=True),
    UniqueConstraint("event_type", "idempotency_key", name="outbox_events_event_type_idempotency_key_uq"),
)
