from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0007_create_sync_jobs"
down_revision = "0006_create_outbox_events"
branch_labels = None
depends_on = None

MARKET_DATA_SCHEMA = "_market_data"


def upgrade() -> None:
    op.create_table(
        "sync_jobs",
        sa.Column("idempotency_key", sa.Text(), primary_key=True),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("provider_symbol", sa.Text(), nullable=False),
        sa.Column("timeframe", sa.Text(), nullable=False),
        sa.Column("expected_close_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("scheduled_for", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint(
            "source",
            "provider_symbol",
            "timeframe",
            "expected_close_time",
            name="sync_jobs_logical_sync_uq",
        ),
        schema=MARKET_DATA_SCHEMA,
    )
    op.create_index(
        "sync_jobs_status_scheduled_for_idx",
        "sync_jobs",
        ["status", "scheduled_for"],
        schema=MARKET_DATA_SCHEMA,
    )


def downgrade() -> None:
    op.drop_index("sync_jobs_status_scheduled_for_idx", table_name="sync_jobs", schema=MARKET_DATA_SCHEMA)
    op.drop_table("sync_jobs", schema=MARKET_DATA_SCHEMA)
