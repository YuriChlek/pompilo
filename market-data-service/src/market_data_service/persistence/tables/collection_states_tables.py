from __future__ import annotations

from sqlalchemy import Column, DateTime, PrimaryKeyConstraint, Table, Text, func

from market_data_service.persistence.tables.metadata import metadata

collection_states = Table(
    "collection_states",
    metadata,
    Column("source", Text, nullable=False),
    Column("canonical_symbol", Text, nullable=False),
    Column("provider_symbol", Text, nullable=False),
    Column("timeframe", Text, nullable=False),
    Column("bootstrap_from", DateTime(timezone=True), nullable=False),
    Column("bootstrap_to", DateTime(timezone=True), nullable=False),
    Column("bootstrap_next_from", DateTime(timezone=True), nullable=False),
    Column("bootstrap_completed_at", DateTime(timezone=True), nullable=True),
    Column("last_successful_close_time", DateTime(timezone=True), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    PrimaryKeyConstraint("source", "canonical_symbol", "timeframe", name="collection_states_pk"),
)
