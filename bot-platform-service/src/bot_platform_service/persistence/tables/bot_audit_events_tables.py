from __future__ import annotations

from sqlalchemy import Column, DateTime, ForeignKeyConstraint, Table, Text, func
from sqlalchemy.dialects.postgresql import JSONB

from bot_platform_service.persistence.tables.metadata import BOT_PLATFORM_SCHEMA, metadata

bot_audit_events = Table(
    "bot_audit_events",
    metadata,
    Column("event_id", Text, primary_key=True),
    Column("event_type", Text, nullable=False),
    Column("instance_id", Text, nullable=True),
    Column("module_id", Text, nullable=True),
    Column("actor_type", Text, nullable=False),
    Column("actor_id", Text, nullable=False),
    Column("payload_json", JSONB, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("correlation_id", Text, nullable=True),
    ForeignKeyConstraint(
        ["instance_id"],
        [f"{BOT_PLATFORM_SCHEMA}.bot_instances.instance_id"],
        name="bot_audit_events_instance_id_fk",
        ondelete="SET NULL",
    ),
    ForeignKeyConstraint(
        ["module_id"],
        [f"{BOT_PLATFORM_SCHEMA}.bot_modules.module_id"],
        name="bot_audit_events_module_id_fk",
        ondelete="SET NULL",
    ),
)
