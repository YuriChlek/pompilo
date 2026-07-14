from __future__ import annotations

from sqlalchemy import Boolean, Column, DateTime, Integer, Numeric, PrimaryKeyConstraint, Table, Text, UniqueConstraint, func

from market_data_service.persistence.tables.metadata import metadata

market_candles = Table(
    "market_candles",
    metadata,
    Column("candle_id", Text, nullable=False),
    Column("source", Text, nullable=False),
    Column("canonical_symbol", Text, nullable=False),
    Column("provider_symbol", Text, nullable=False),
    Column("timeframe", Text, nullable=False),
    Column("open_time", DateTime(timezone=True), nullable=False),
    Column("close_time", DateTime(timezone=True), nullable=False),
    Column("open", Numeric, nullable=False),
    Column("high", Numeric, nullable=False),
    Column("low", Numeric, nullable=False),
    Column("close", Numeric, nullable=False),
    Column("volume", Numeric, nullable=False),
    Column("quote_volume", Numeric, nullable=True),
    Column("taker_buy_base_volume", Numeric, nullable=True),
    Column("taker_buy_quote_volume", Numeric, nullable=True),
    Column("taker_sell_base_volume", Numeric, nullable=True),
    Column("taker_sell_quote_volume", Numeric, nullable=True),
    Column("trades_count", Integer, nullable=True),
    Column("is_closed", Boolean, nullable=False),
    Column("provider_payload_hash", Text, nullable=False),
    Column("inserted_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    PrimaryKeyConstraint("timeframe", "candle_id", name="market_candles_pk"),
    UniqueConstraint("source", "canonical_symbol", "timeframe", "open_time", name="market_candles_natural_key_uq"),
    postgresql_partition_by="LIST (timeframe)",
)
