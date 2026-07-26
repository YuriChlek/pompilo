from __future__ import annotations

from sqlalchemy import Column, DateTime, Integer, Table, Text, func

from market_data_service.persistence.tables.metadata import metadata

market_data_batches = Table(
    "market_data_batches",
    metadata,
    Column("batch_id", Text, primary_key=True),
    Column("source", Text, nullable=False),
    Column("canonical_symbol", Text, nullable=False),
    Column("timeframe", Text, nullable=False),
    Column("requested_from", DateTime(timezone=True), nullable=False),
    Column("requested_to", DateTime(timezone=True), nullable=False),
    Column("expected_close_time", DateTime(timezone=True), nullable=False),
    Column("status", Text, nullable=False),
    Column("outbox_status", Text, nullable=False),
    Column("rows_fetched", Integer, nullable=False),
    Column("rows_inserted", Integer, nullable=False),
    Column("rows_skipped_duplicate", Integer, nullable=False),
    Column("rows_hash_mismatch", Integer, nullable=False),
    Column("gap_count", Integer, nullable=False),
    Column("first_open_time", DateTime(timezone=True), nullable=True),
    Column("last_close_time", DateTime(timezone=True), nullable=True),
    Column("error_code", Text, nullable=True),
    Column("error_message_redacted", Text, nullable=True),
    Column("started_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("completed_at", DateTime(timezone=True), nullable=True),
)
