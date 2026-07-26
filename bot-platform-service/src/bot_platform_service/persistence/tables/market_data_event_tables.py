from __future__ import annotations

from sqlalchemy import CheckConstraint, Column, DateTime, Table, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB

from bot_platform_service.persistence.tables.metadata import metadata


market_data_processed_events = Table(
    "market_data_processed_events",
    metadata,
    Column("idempotency_key", Text, primary_key=True),
    Column("event_type", Text, nullable=False),
    Column("contract_version", Text, nullable=False),
    Column("source", Text, nullable=False),
    Column("symbol", Text, nullable=False),
    Column("timeframe", Text, nullable=False),
    Column("snapshot_id", Text, nullable=False),
    Column("redis_message_id", Text, nullable=False),
    Column("processing_status", Text, nullable=False),
    Column("payload_json", JSONB, nullable=False),
    Column("processed_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint("redis_message_id", name="market_data_processed_events_redis_message_id_uq"),
    CheckConstraint(
        "processing_status in ('PROCESSED', 'DUPLICATE', 'INVALID', 'PROCESSING', 'FAILED_RETRYABLE')",
        name="processing_status_values",
    ),
)
