from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0006_create_outbox_events"
down_revision = "0005_create_market_snapshots"
branch_labels = None
depends_on = None

MARKET_DATA_SCHEMA = "_market_data"


def upgrade() -> None:
    op.create_table(
        "outbox_events",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("aggregate_type", sa.Text(), nullable=False),
        sa.Column("aggregate_id", sa.Text(), nullable=False),
        sa.Column("payload_json", postgresql.JSONB(), nullable=False),
        sa.Column("idempotency_key", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("event_type", "idempotency_key", name="outbox_events_event_type_idempotency_key_uq"),
        schema=MARKET_DATA_SCHEMA,
    )
    op.create_index(
        "outbox_events_status_next_attempt_at_idx",
        "outbox_events",
        ["status", "next_attempt_at"],
        schema=MARKET_DATA_SCHEMA,
    )


def downgrade() -> None:
    op.drop_index("outbox_events_status_next_attempt_at_idx", table_name="outbox_events", schema=MARKET_DATA_SCHEMA)
    op.drop_table("outbox_events", schema=MARKET_DATA_SCHEMA)
