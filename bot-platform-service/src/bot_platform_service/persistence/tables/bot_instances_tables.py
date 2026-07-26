from __future__ import annotations

from sqlalchemy import Boolean, CheckConstraint, Column, DateTime, ForeignKeyConstraint, Index, Integer, Table, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB

from bot_platform_service.persistence.tables.metadata import BOT_PLATFORM_SCHEMA, metadata

bot_instances = Table(
    "bot_instances",
    metadata,
    Column("instance_id", Text, primary_key=True),
    Column("module_id", Text, nullable=False),
    Column("tenant_id", Text, nullable=True),
    Column("name", Text, nullable=False),
    Column("mode", Text, nullable=False),
    Column("status", Text, nullable=False),
    Column("symbols", JSONB, nullable=False),
    Column("timeframes", JSONB, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    ForeignKeyConstraint(
        ["module_id"],
        [f"{BOT_PLATFORM_SCHEMA}.bot_modules.module_id"],
        name="bot_instances_module_id_fk",
        ondelete="RESTRICT",
    ),
    CheckConstraint("mode in ('dry_run', 'notification_only', 'signal_only')", name="mode_values"),
    CheckConstraint(
        "status in ('CREATED', 'VALIDATED', 'ENABLED', 'RUNNING', 'PAUSED', 'FAILED', 'DISABLED')",
        name="status_values",
    ),
)

bot_instance_configs = Table(
    "bot_instance_configs",
    metadata,
    Column("config_id", Text, primary_key=True),
    Column("instance_id", Text, nullable=False),
    Column("config_schema_version", Integer, nullable=False),
    Column("config_json", JSONB, nullable=False),
    Column("config_hash", Text, nullable=False),
    Column("is_active", Boolean, nullable=False, server_default="true"),
    Column("created_by_actor_type", Text, nullable=False),
    Column("created_by_actor_id", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("correlation_id", Text, nullable=True),
    ForeignKeyConstraint(
        ["instance_id"],
        [f"{BOT_PLATFORM_SCHEMA}.bot_instances.instance_id"],
        name="bot_instance_configs_instance_id_fk",
        ondelete="CASCADE",
    ),
    UniqueConstraint("instance_id", "config_hash", name="bot_instance_configs_instance_hash_uq"),
)

Index(
    "bot_instance_configs_one_active_config_uq",
    bot_instance_configs.c.instance_id,
    unique=True,
    postgresql_where=bot_instance_configs.c.is_active.is_(True),
)

bot_runtime_state = Table(
    "bot_runtime_state",
    metadata,
    Column("state_id", Text, primary_key=True),
    Column("instance_id", Text, nullable=False),
    Column("namespace", Text, nullable=False),
    Column("state_key", Text, nullable=False),
    Column("state_json", JSONB, nullable=False),
    Column("state_hash", Text, nullable=False),
    Column("version", Integer, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("correlation_id", Text, nullable=True),
    ForeignKeyConstraint(
        ["instance_id"],
        [f"{BOT_PLATFORM_SCHEMA}.bot_instances.instance_id"],
        name="bot_runtime_state_instance_id_fk",
        ondelete="CASCADE",
    ),
    UniqueConstraint("instance_id", "namespace", "state_key", name="bot_runtime_state_instance_namespace_key_uq"),
    CheckConstraint("version >= 1", name="version_positive"),
)

bot_permissions = Table(
    "bot_permissions",
    metadata,
    Column("permission_id", Text, primary_key=True),
    Column("instance_id", Text, nullable=False),
    Column("permission", Text, nullable=False),
    Column("enabled", Boolean, nullable=False, server_default="true"),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    ForeignKeyConstraint(
        ["instance_id"],
        [f"{BOT_PLATFORM_SCHEMA}.bot_instances.instance_id"],
        name="bot_permissions_instance_id_fk",
        ondelete="CASCADE",
    ),
    UniqueConstraint("instance_id", "permission", name="bot_permissions_instance_permission_uq"),
)

bot_secrets_refs = Table(
    "bot_secrets_refs",
    metadata,
    Column("secret_ref_id", Text, primary_key=True),
    Column("instance_id", Text, nullable=False),
    Column("secret_name", Text, nullable=False),
    Column("provider", Text, nullable=False),
    Column("provider_ref", Text, nullable=False),
    Column("status", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    ForeignKeyConstraint(
        ["instance_id"],
        [f"{BOT_PLATFORM_SCHEMA}.bot_instances.instance_id"],
        name="bot_secrets_refs_instance_id_fk",
        ondelete="CASCADE",
    ),
    UniqueConstraint("instance_id", "secret_name", name="bot_secrets_refs_instance_secret_name_uq"),
    CheckConstraint("status in ('ACTIVE', 'DISABLED', 'ROTATING')", name="status_values"),
)
