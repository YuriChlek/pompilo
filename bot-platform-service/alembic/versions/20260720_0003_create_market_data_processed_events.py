from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0003_create_market_data_processed_events"
down_revision = "0002_extend_bot_module_metadata"
branch_labels = None
depends_on = None

BOT_PLATFORM_SCHEMA = "_bot_platform"


def upgrade() -> None:
    op.create_table(
        "market_data_processed_events",
        sa.Column("idempotency_key", sa.Text(), primary_key=True),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("contract_version", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("timeframe", sa.Text(), nullable=False),
        sa.Column("snapshot_id", sa.Text(), nullable=False),
        sa.Column("redis_message_id", sa.Text(), nullable=False),
        sa.Column("processing_status", sa.Text(), nullable=False),
        sa.Column("payload_json", postgresql.JSONB(), nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("redis_message_id", name="market_data_processed_events_redis_message_id_uq"),
        sa.CheckConstraint(
            "processing_status in ('PROCESSED', 'DUPLICATE', 'INVALID', 'PROCESSING', 'FAILED_RETRYABLE')",
            name="market_data_processed_events_processing_status_values_ck",
        ),
        schema=BOT_PLATFORM_SCHEMA,
    )


def downgrade() -> None:
    op.drop_table("market_data_processed_events", schema=BOT_PLATFORM_SCHEMA)
