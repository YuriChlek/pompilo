from __future__ import annotations

from sqlalchemy import CheckConstraint, Column, DateTime, Integer, Table, Text, func
from sqlalchemy.dialects.postgresql import JSONB

from bot_platform_service.persistence.tables.metadata import metadata

bot_modules = Table(
    "bot_modules",
    metadata,
    Column("module_id", Text, primary_key=True),
    Column("display_name", Text, nullable=False),
    Column("version", Text, nullable=False),
    Column("adapter_path", Text, nullable=False),
    Column("adapter_class", Text, nullable=True),
    Column("config_schema_version", Integer, nullable=True),
    Column("config_schema_json", JSONB, nullable=True),
    Column("status", Text, nullable=False),
    Column("manifest_json", JSONB, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    CheckConstraint("status in ('ACTIVE', 'DISABLED', 'DEPRECATED')", name="status_values"),
)
