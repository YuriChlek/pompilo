from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    ForeignKeyConstraint,
    Identity,
    Integer,
    Table,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

from market_data_service.persistence.tables.metadata import MARKET_DATA_SCHEMA, metadata

provider_symbols = Table(
    "provider_symbols",
    metadata,
    Column("id", BigInteger, Identity(always=False), primary_key=True),
    Column("source", Text, nullable=False),
    Column("canonical_symbol", Text, nullable=False),
    Column("provider_symbol", Text, nullable=False),
    Column("status", Text, nullable=False),
    Column("supported_timeframes", ARRAY(Text), nullable=False),
    Column("min_available_time", DateTime(timezone=True), nullable=True),
    Column("max_backfill_days", Integer, nullable=True),
    Column("metadata_json", JSONB, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    ForeignKeyConstraint(
        ["canonical_symbol"],
        [f"{MARKET_DATA_SCHEMA}.market_symbols.canonical_symbol"],
        name="provider_symbols_canonical_symbol_fk",
        ondelete="RESTRICT",
    ),
    UniqueConstraint("source", "canonical_symbol", name="provider_symbols_source_canonical_symbol_uq"),
    UniqueConstraint("source", "provider_symbol", name="provider_symbols_source_provider_symbol_uq"),
)
