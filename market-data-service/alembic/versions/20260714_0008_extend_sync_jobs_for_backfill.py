from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0008_extend_sync_backfill"
down_revision = "0007_create_sync_jobs"
branch_labels = None
depends_on = None

MARKET_DATA_SCHEMA = "_market_data"


def upgrade() -> None:
    op.add_column("sync_jobs", sa.Column("job_kind", sa.Text(), nullable=False, server_default="FRESH"), schema=MARKET_DATA_SCHEMA)
    op.add_column("sync_jobs", sa.Column("priority", sa.Integer(), nullable=False, server_default="10"), schema=MARKET_DATA_SCHEMA)
    op.add_column("sync_jobs", sa.Column("requested_from", sa.DateTime(timezone=True), nullable=True), schema=MARKET_DATA_SCHEMA)
    op.add_column("sync_jobs", sa.Column("requested_to", sa.DateTime(timezone=True), nullable=True), schema=MARKET_DATA_SCHEMA)
    op.drop_constraint("sync_jobs_logical_sync_uq", "sync_jobs", schema=MARKET_DATA_SCHEMA, type_="unique")
    op.create_unique_constraint(
        "sync_jobs_logical_sync_uq",
        "sync_jobs",
        ["source", "provider_symbol", "timeframe", "job_kind", "requested_from", "requested_to", "expected_close_time"],
        schema=MARKET_DATA_SCHEMA,
    )
    op.create_check_constraint(
        "sync_jobs_job_kind_ck",
        "sync_jobs",
        "job_kind in ('FRESH', 'BACKFILL')",
        schema=MARKET_DATA_SCHEMA,
    )
    op.create_check_constraint(
        "sync_jobs_priority_non_negative_ck",
        "sync_jobs",
        "priority >= 0",
        schema=MARKET_DATA_SCHEMA,
    )
    op.create_index(
        "sync_jobs_priority_scheduled_for_idx",
        "sync_jobs",
        ["priority", "scheduled_for"],
        schema=MARKET_DATA_SCHEMA,
    )


def downgrade() -> None:
    op.drop_index("sync_jobs_priority_scheduled_for_idx", table_name="sync_jobs", schema=MARKET_DATA_SCHEMA)
    op.drop_constraint("sync_jobs_priority_non_negative_ck", "sync_jobs", schema=MARKET_DATA_SCHEMA, type_="check")
    op.drop_constraint("sync_jobs_job_kind_ck", "sync_jobs", schema=MARKET_DATA_SCHEMA, type_="check")
    op.drop_constraint("sync_jobs_logical_sync_uq", "sync_jobs", schema=MARKET_DATA_SCHEMA, type_="unique")
    op.create_unique_constraint(
        "sync_jobs_logical_sync_uq",
        "sync_jobs",
        ["source", "provider_symbol", "timeframe", "expected_close_time"],
        schema=MARKET_DATA_SCHEMA,
    )
    op.drop_column("sync_jobs", "requested_to", schema=MARKET_DATA_SCHEMA)
    op.drop_column("sync_jobs", "requested_from", schema=MARKET_DATA_SCHEMA)
    op.drop_column("sync_jobs", "priority", schema=MARKET_DATA_SCHEMA)
    op.drop_column("sync_jobs", "job_kind", schema=MARKET_DATA_SCHEMA)
