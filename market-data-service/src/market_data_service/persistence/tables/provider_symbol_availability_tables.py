from __future__ import annotations

from sqlalchemy import (
    Column,
    DateTime,
    Table,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB

from market_data_service.persistence.tables.metadata import metadata

provider_symbol_availability = Table(
    "provider_symbol_availability",
    metadata,
    Column("source", Text, primary_key=True),
    Column("requested_symbol", Text, primary_key=True),
    Column("provider_symbol", Text, nullable=True),
    Column("status", Text, nullable=False),
    Column("first_seen_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("last_checked_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("next_check_at", DateTime(timezone=True), nullable=False),
    Column("failure_reason", Text, nullable=True),
    Column("metadata_json", JSONB, nullable=False, server_default="{}"),
)
