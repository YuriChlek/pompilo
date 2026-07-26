from __future__ import annotations

from sqlalchemy import CheckConstraint, Column, DateTime, ForeignKeyConstraint, Integer, Numeric, Table, Text, func
from sqlalchemy.dialects.postgresql import JSONB

from bot_platform_service.persistence.tables.metadata import BOT_PLATFORM_SCHEMA, metadata

bot_signals = Table(
    "bot_signals",
    metadata,
    Column("signal_id", Text, primary_key=True),
    Column("signal_key", Text, nullable=False, unique=True),
    Column("run_id", Text, nullable=False),
    Column("instance_id", Text, nullable=False),
    Column("module_id", Text, nullable=False),
    Column("symbol", Text, nullable=False),
    Column("timeframe", Text, nullable=False),
    Column("snapshot_id", Text, nullable=False),
    Column("signal_type", Text, nullable=False),
    Column("side", Text, nullable=True),
    Column("confidence", Numeric(), nullable=True),
    Column("reason", Text, nullable=False),
    Column("payload_schema", Text, nullable=False),
    Column("payload_schema_version", Integer, nullable=False),
    Column("payload_hash", Text, nullable=False),
    Column("payload_json", JSONB, nullable=False),
    Column("status", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("correlation_id", Text, nullable=True),
    ForeignKeyConstraint(
        ["run_id"],
        [f"{BOT_PLATFORM_SCHEMA}.bot_runs.run_id"],
        name="bot_signals_run_id_fk",
        ondelete="CASCADE",
    ),
    ForeignKeyConstraint(
        ["instance_id"],
        [f"{BOT_PLATFORM_SCHEMA}.bot_instances.instance_id"],
        name="bot_signals_instance_id_fk",
        ondelete="CASCADE",
    ),
    ForeignKeyConstraint(
        ["module_id"],
        [f"{BOT_PLATFORM_SCHEMA}.bot_modules.module_id"],
        name="bot_signals_module_id_fk",
        ondelete="RESTRICT",
    ),
    CheckConstraint("signal_type in ('entry', 'exit', 'rebalance', 'hold', 'alert')", name="signal_type_values"),
    CheckConstraint("side is null or side in ('buy', 'sell')", name="side_values"),
    CheckConstraint("status in ('CREATED', 'PUBLISHED', 'IGNORED')", name="status_values"),
)
