from __future__ import annotations

from sqlalchemy import Column, DateTime, ForeignKeyConstraint, Integer, PrimaryKeyConstraint, Table, Text, UniqueConstraint, func

from market_data_service.persistence.tables.metadata import MARKET_DATA_SCHEMA, metadata

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
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint(
        "source",
        "canonical_symbol",
        "timeframe",
        "lookback_start_time",
        "lookback_end_time",
        "data_hash",
        "completeness_status",
        name="market_snapshots_logical_data_hash_uq",
    ),
    UniqueConstraint(
        "source",
        "canonical_symbol",
        "timeframe",
        "lookback_start_time",
        "lookback_end_time",
        "snapshot_version",
        name="market_snapshots_logical_version_uq",
    ),
)

market_snapshot_candles = Table(
    "market_snapshot_candles",
    metadata,
    Column("snapshot_id", Text, nullable=False),
    Column("candle_id", Text, nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("candle_hash_at_snapshot", Text, nullable=False),
    ForeignKeyConstraint(
        ["snapshot_id"],
        [f"{MARKET_DATA_SCHEMA}.market_snapshots.id"],
        name="market_snapshot_candles_snapshot_id_fk",
        ondelete="CASCADE",
    ),
    PrimaryKeyConstraint("snapshot_id", "ordinal", name="market_snapshot_candles_pk"),
    UniqueConstraint("snapshot_id", "candle_id", name="market_snapshot_candles_snapshot_id_candle_id_uq"),
)
