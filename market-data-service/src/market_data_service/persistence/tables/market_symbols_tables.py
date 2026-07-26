from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Identity,
    Table,
    Text,
    UniqueConstraint,
    func,
)

from market_data_service.persistence.tables.metadata import metadata

market_symbols = Table(
    "market_symbols",
    metadata,
    Column("id", BigInteger, Identity(always=False), primary_key=True),
    Column("canonical_symbol", Text, nullable=False),
    Column("base_asset", Text, nullable=True),
    Column("quote_asset", Text, nullable=True),
    Column("status", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint("canonical_symbol", name="market_symbols_canonical_symbol_uq"),
)
