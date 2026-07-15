from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_create_bot_platform_schema"
down_revision = None
branch_labels = None
depends_on = None

BOT_PLATFORM_SCHEMA = "_bot_platform"


def upgrade() -> None:
    op.execute(sa.schema.CreateSchema(BOT_PLATFORM_SCHEMA, if_not_exists=True))

    op.create_table(
        "bot_modules",
        sa.Column("module_id", sa.Text(), primary_key=True),
        sa.Column("display_name", sa.Text(), nullable=False),
        sa.Column("version", sa.Text(), nullable=False),
        sa.Column("adapter_path", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("manifest_json", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("status in ('ACTIVE', 'DISABLED', 'DEPRECATED')", name="bot_modules_status_values_ck"),
        schema=BOT_PLATFORM_SCHEMA,
    )

    op.create_table(
        "bot_instances",
        sa.Column("instance_id", sa.Text(), primary_key=True),
        sa.Column("module_id", sa.Text(), nullable=False),
        sa.Column("tenant_id", sa.Text(), nullable=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("mode", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("symbols", postgresql.JSONB(), nullable=False),
        sa.Column("timeframes", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["module_id"], [f"{BOT_PLATFORM_SCHEMA}.bot_modules.module_id"], name="bot_instances_module_id_fk", ondelete="RESTRICT"),
        sa.CheckConstraint("mode in ('dry_run', 'notification_only', 'signal_only')", name="bot_instances_mode_values_ck"),
        sa.CheckConstraint(
            "status in ('CREATED', 'VALIDATED', 'ENABLED', 'RUNNING', 'PAUSED', 'FAILED', 'DISABLED')",
            name="bot_instances_status_values_ck",
        ),
        schema=BOT_PLATFORM_SCHEMA,
    )

    op.create_table(
        "bot_instance_configs",
        sa.Column("config_id", sa.Text(), primary_key=True),
        sa.Column("instance_id", sa.Text(), nullable=False),
        sa.Column("config_schema_version", sa.Integer(), nullable=False),
        sa.Column("config_json", postgresql.JSONB(), nullable=False),
        sa.Column("config_hash", sa.Text(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_by_actor_type", sa.Text(), nullable=False),
        sa.Column("created_by_actor_id", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("correlation_id", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["instance_id"], [f"{BOT_PLATFORM_SCHEMA}.bot_instances.instance_id"], name="bot_instance_configs_instance_id_fk", ondelete="CASCADE"),
        sa.UniqueConstraint("instance_id", "config_hash", name="bot_instance_configs_instance_hash_uq"),
        schema=BOT_PLATFORM_SCHEMA,
    )
    op.create_index(
        "bot_instance_configs_one_active_config_uq",
        "bot_instance_configs",
        ["instance_id"],
        unique=True,
        schema=BOT_PLATFORM_SCHEMA,
        postgresql_where=sa.text("is_active is true"),
    )

    op.create_table(
        "bot_runtime_state",
        sa.Column("state_id", sa.Text(), primary_key=True),
        sa.Column("instance_id", sa.Text(), nullable=False),
        sa.Column("namespace", sa.Text(), nullable=False),
        sa.Column("state_key", sa.Text(), nullable=False),
        sa.Column("state_json", postgresql.JSONB(), nullable=False),
        sa.Column("state_hash", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("correlation_id", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["instance_id"], [f"{BOT_PLATFORM_SCHEMA}.bot_instances.instance_id"], name="bot_runtime_state_instance_id_fk", ondelete="CASCADE"),
        sa.UniqueConstraint("instance_id", "namespace", "state_key", name="bot_runtime_state_instance_namespace_key_uq"),
        sa.CheckConstraint("version >= 1", name="bot_runtime_state_version_positive_ck"),
        schema=BOT_PLATFORM_SCHEMA,
    )

    op.create_table(
        "bot_permissions",
        sa.Column("permission_id", sa.Text(), primary_key=True),
        sa.Column("instance_id", sa.Text(), nullable=False),
        sa.Column("permission", sa.Text(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["instance_id"], [f"{BOT_PLATFORM_SCHEMA}.bot_instances.instance_id"], name="bot_permissions_instance_id_fk", ondelete="CASCADE"),
        sa.UniqueConstraint("instance_id", "permission", name="bot_permissions_instance_permission_uq"),
        schema=BOT_PLATFORM_SCHEMA,
    )

    op.create_table(
        "bot_secrets_refs",
        sa.Column("secret_ref_id", sa.Text(), primary_key=True),
        sa.Column("instance_id", sa.Text(), nullable=False),
        sa.Column("secret_name", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("provider_ref", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["instance_id"], [f"{BOT_PLATFORM_SCHEMA}.bot_instances.instance_id"], name="bot_secrets_refs_instance_id_fk", ondelete="CASCADE"),
        sa.UniqueConstraint("instance_id", "secret_name", name="bot_secrets_refs_instance_secret_name_uq"),
        sa.CheckConstraint("status in ('ACTIVE', 'DISABLED', 'ROTATING')", name="bot_secrets_refs_status_values_ck"),
        schema=BOT_PLATFORM_SCHEMA,
    )

    op.create_table(
        "bot_runs",
        sa.Column("run_id", sa.Text(), primary_key=True),
        sa.Column("instance_id", sa.Text(), nullable=False),
        sa.Column("module_id", sa.Text(), nullable=False),
        sa.Column("trigger_type", sa.Text(), nullable=False),
        sa.Column("trigger_event_id", sa.Text(), nullable=True),
        sa.Column("snapshot_id", sa.Text(), nullable=True),
        sa.Column("idempotency_key", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.Text(), nullable=True),
        sa.Column("error_message_redacted", sa.Text(), nullable=True),
        sa.Column("correlation_id", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["instance_id"], [f"{BOT_PLATFORM_SCHEMA}.bot_instances.instance_id"], name="bot_runs_instance_id_fk", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["module_id"], [f"{BOT_PLATFORM_SCHEMA}.bot_modules.module_id"], name="bot_runs_module_id_fk", ondelete="RESTRICT"),
        sa.UniqueConstraint("idempotency_key", name="bot_runs_idempotency_key_uq"),
        sa.CheckConstraint("trigger_type in ('manual', 'scheduler', 'event')", name="bot_runs_trigger_type_values_ck"),
        sa.CheckConstraint("status in ('RUNNING', 'COMPLETE', 'FAILED', 'CANCELLED')", name="bot_runs_status_values_ck"),
        schema=BOT_PLATFORM_SCHEMA,
    )

    op.create_table(
        "bot_run_events",
        sa.Column("event_id", sa.Text(), primary_key=True),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("instance_id", sa.Text(), nullable=False),
        sa.Column("module_id", sa.Text(), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("payload_json", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("correlation_id", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], [f"{BOT_PLATFORM_SCHEMA}.bot_runs.run_id"], name="bot_run_events_run_id_fk", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["instance_id"], [f"{BOT_PLATFORM_SCHEMA}.bot_instances.instance_id"], name="bot_run_events_instance_id_fk", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["module_id"], [f"{BOT_PLATFORM_SCHEMA}.bot_modules.module_id"], name="bot_run_events_module_id_fk", ondelete="RESTRICT"),
        schema=BOT_PLATFORM_SCHEMA,
    )

    op.create_table(
        "bot_signals",
        sa.Column("signal_id", sa.Text(), primary_key=True),
        sa.Column("signal_key", sa.Text(), nullable=False, unique=True),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("instance_id", sa.Text(), nullable=False),
        sa.Column("module_id", sa.Text(), nullable=False),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("timeframe", sa.Text(), nullable=False),
        sa.Column("snapshot_id", sa.Text(), nullable=False),
        sa.Column("signal_type", sa.Text(), nullable=False),
        sa.Column("side", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Numeric(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("payload_schema", sa.Text(), nullable=False),
        sa.Column("payload_schema_version", sa.Integer(), nullable=False),
        sa.Column("payload_hash", sa.Text(), nullable=False),
        sa.Column("payload_json", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("correlation_id", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], [f"{BOT_PLATFORM_SCHEMA}.bot_runs.run_id"], name="bot_signals_run_id_fk", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["instance_id"], [f"{BOT_PLATFORM_SCHEMA}.bot_instances.instance_id"], name="bot_signals_instance_id_fk", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["module_id"], [f"{BOT_PLATFORM_SCHEMA}.bot_modules.module_id"], name="bot_signals_module_id_fk", ondelete="RESTRICT"),
        sa.CheckConstraint("signal_type in ('entry', 'exit', 'rebalance', 'hold', 'alert')", name="bot_signals_signal_type_values_ck"),
        sa.CheckConstraint("side is null or side in ('buy', 'sell')", name="bot_signals_side_values_ck"),
        sa.CheckConstraint("status in ('CREATED', 'PUBLISHED', 'IGNORED')", name="bot_signals_status_values_ck"),
        schema=BOT_PLATFORM_SCHEMA,
    )

    op.create_table(
        "bot_health_checks",
        sa.Column("health_check_id", sa.Text(), primary_key=True),
        sa.Column("instance_id", sa.Text(), nullable=False),
        sa.Column("module_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("details_json", postgresql.JSONB(), nullable=False),
        sa.Column("checked_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("correlation_id", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["instance_id"], [f"{BOT_PLATFORM_SCHEMA}.bot_instances.instance_id"], name="bot_health_checks_instance_id_fk", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["module_id"], [f"{BOT_PLATFORM_SCHEMA}.bot_modules.module_id"], name="bot_health_checks_module_id_fk", ondelete="RESTRICT"),
        sa.CheckConstraint("status in ('HEALTHY', 'DEGRADED', 'UNHEALTHY')", name="bot_health_checks_status_values_ck"),
        schema=BOT_PLATFORM_SCHEMA,
    )

    op.create_table(
        "bot_audit_events",
        sa.Column("event_id", sa.Text(), primary_key=True),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("instance_id", sa.Text(), nullable=True),
        sa.Column("module_id", sa.Text(), nullable=True),
        sa.Column("actor_type", sa.Text(), nullable=False),
        sa.Column("actor_id", sa.Text(), nullable=False),
        sa.Column("payload_json", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("correlation_id", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["instance_id"], [f"{BOT_PLATFORM_SCHEMA}.bot_instances.instance_id"], name="bot_audit_events_instance_id_fk", ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["module_id"], [f"{BOT_PLATFORM_SCHEMA}.bot_modules.module_id"], name="bot_audit_events_module_id_fk", ondelete="SET NULL"),
        schema=BOT_PLATFORM_SCHEMA,
    )


def downgrade() -> None:
    op.drop_table("bot_audit_events", schema=BOT_PLATFORM_SCHEMA)
    op.drop_table("bot_health_checks", schema=BOT_PLATFORM_SCHEMA)
    op.drop_table("bot_signals", schema=BOT_PLATFORM_SCHEMA)
    op.drop_table("bot_run_events", schema=BOT_PLATFORM_SCHEMA)
    op.drop_table("bot_runs", schema=BOT_PLATFORM_SCHEMA)
    op.drop_table("bot_secrets_refs", schema=BOT_PLATFORM_SCHEMA)
    op.drop_table("bot_permissions", schema=BOT_PLATFORM_SCHEMA)
    op.drop_table("bot_runtime_state", schema=BOT_PLATFORM_SCHEMA)
    op.drop_index("bot_instance_configs_one_active_config_uq", table_name="bot_instance_configs", schema=BOT_PLATFORM_SCHEMA)
    op.drop_table("bot_instance_configs", schema=BOT_PLATFORM_SCHEMA)
    op.drop_table("bot_instances", schema=BOT_PLATFORM_SCHEMA)
    op.drop_table("bot_modules", schema=BOT_PLATFORM_SCHEMA)
    op.execute(sa.schema.DropSchema(BOT_PLATFORM_SCHEMA, if_exists=True))
