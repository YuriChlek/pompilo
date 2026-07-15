from __future__ import annotations

from sqlalchemy import Boolean, Column, DateTime, Integer, MetaData, Numeric, Table, Text

MARKET_DATA_SCHEMA = "_market_data"

metadata = MetaData(schema=MARKET_DATA_SCHEMA)

market_snapshots = Table(
    "market_snapshots",
    metadata,
    Column("id", Text, primary_key=True),
    Column("source", Text, nullable=False),
    Column("canonical_symbol", Text, nullable=False),
    Column("timeframe", Text, nullable=False),
    Column("last_closed_candle_time", DateTime(timezone=True), nullable=False),
    Column("lookback_start_time", DateTime(timezone=True), nullable=False),
    Column("lookback_end_time", DateTime(timezone=True), nullable=False),
    Column("candle_count", Integer, nullable=False),
    Column("data_hash", Text, nullable=False),
    Column("batch_id", Text, nullable=False),
    Column("completeness_status", Text, nullable=False),
    Column("snapshot_version", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

market_snapshot_candles = Table(
    "market_snapshot_candles",
    metadata,
    Column("snapshot_id", Text, nullable=False),
    Column("candle_id", Text, nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("candle_hash_at_snapshot", Text, nullable=False),
)

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
    Column("is_closed", Boolean, nullable=False),
    Column("provider_payload_hash", Text, nullable=False),
)
