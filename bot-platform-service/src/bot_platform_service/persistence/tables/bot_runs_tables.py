from __future__ import annotations

from sqlalchemy import CheckConstraint, Column, DateTime, ForeignKeyConstraint, Table, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB

from bot_platform_service.persistence.tables.metadata import BOT_PLATFORM_SCHEMA, metadata

bot_runs = Table(
    "bot_runs",
    metadata,
    Column("run_id", Text, primary_key=True),
    Column("instance_id", Text, nullable=False),
    Column("module_id", Text, nullable=False),
    Column("trigger_type", Text, nullable=False),
    Column("trigger_event_id", Text, nullable=True),
    Column("snapshot_id", Text, nullable=True),
    Column("idempotency_key", Text, nullable=True),
    Column("status", Text, nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("completed_at", DateTime(timezone=True), nullable=True),
    Column("error_code", Text, nullable=True),
    Column("error_message_redacted", Text, nullable=True),
    Column("correlation_id", Text, nullable=True),
    ForeignKeyConstraint(
        ["instance_id"],
        [f"{BOT_PLATFORM_SCHEMA}.bot_instances.instance_id"],
        name="bot_runs_instance_id_fk",
        ondelete="CASCADE",
    ),
    ForeignKeyConstraint(
        ["module_id"],
        [f"{BOT_PLATFORM_SCHEMA}.bot_modules.module_id"],
        name="bot_runs_module_id_fk",
        ondelete="RESTRICT",
    ),
    UniqueConstraint("idempotency_key", name="bot_runs_idempotency_key_uq"),
    CheckConstraint("trigger_type in ('manual', 'scheduler', 'event')", name="trigger_type_values"),
    CheckConstraint("status in ('RUNNING', 'COMPLETE', 'FAILED', 'CANCELLED')", name="status_values"),
)

bot_run_events = Table(
    "bot_run_events",
    metadata,
    Column("event_id", Text, primary_key=True),
    Column("run_id", Text, nullable=False),
    Column("instance_id", Text, nullable=False),
    Column("module_id", Text, nullable=False),
    Column("event_type", Text, nullable=False),
    Column("payload_json", JSONB, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("correlation_id", Text, nullable=True),
    ForeignKeyConstraint(
        ["run_id"],
        [f"{BOT_PLATFORM_SCHEMA}.bot_runs.run_id"],
        name="bot_run_events_run_id_fk",
        ondelete="CASCADE",
    ),
    ForeignKeyConstraint(
        ["instance_id"],
        [f"{BOT_PLATFORM_SCHEMA}.bot_instances.instance_id"],
        name="bot_run_events_instance_id_fk",
        ondelete="CASCADE",
    ),
    ForeignKeyConstraint(
        ["module_id"],
        [f"{BOT_PLATFORM_SCHEMA}.bot_modules.module_id"],
        name="bot_run_events_module_id_fk",
        ondelete="RESTRICT",
    ),
)

bot_health_checks = Table(
    "bot_health_checks",
    metadata,
    Column("health_check_id", Text, primary_key=True),
    Column("instance_id", Text, nullable=False),
    Column("module_id", Text, nullable=False),
    Column("status", Text, nullable=False),
    Column("details_json", JSONB, nullable=False),
    Column("checked_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("correlation_id", Text, nullable=True),
    ForeignKeyConstraint(
        ["instance_id"],
        [f"{BOT_PLATFORM_SCHEMA}.bot_instances.instance_id"],
        name="bot_health_checks_instance_id_fk",
        ondelete="CASCADE",
    ),
    ForeignKeyConstraint(
        ["module_id"],
        [f"{BOT_PLATFORM_SCHEMA}.bot_modules.module_id"],
        name="bot_health_checks_module_id_fk",
        ondelete="RESTRICT",
    ),
    CheckConstraint("status in ('HEALTHY', 'DEGRADED', 'UNHEALTHY')", name="status_values"),
)
