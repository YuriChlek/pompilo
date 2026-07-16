from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0004_create_market_data_batches"
down_revision = "0003_market_candles_partitions"
branch_labels = None
depends_on = None

MARKET_DATA_SCHEMA = "_market_data"


def upgrade() -> None:
    op.create_table(
        "market_data_batches",
        sa.Column("batch_id", sa.Text(), primary_key=True),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("canonical_symbol", sa.Text(), nullable=False),
        sa.Column("timeframe", sa.Text(), nullable=False),
        sa.Column("requested_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("requested_to", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expected_close_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("outbox_status", sa.Text(), nullable=False),
        sa.Column("rows_fetched", sa.Integer(), nullable=False),
        sa.Column("rows_inserted", sa.Integer(), nullable=False),
        sa.Column("rows_skipped_duplicate", sa.Integer(), nullable=False),
        sa.Column("rows_hash_mismatch", sa.Integer(), nullable=False),
        sa.Column("gap_count", sa.Integer(), nullable=False),
        sa.Column("first_open_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_close_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.Text(), nullable=True),
        sa.Column("error_message_redacted", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        schema=MARKET_DATA_SCHEMA,
    )
    op.create_index(
        "market_data_batches_lookup_idx",
        "market_data_batches",
        ["source", "canonical_symbol", "timeframe", "requested_to"],
        schema=MARKET_DATA_SCHEMA,
    )


def downgrade() -> None:
    op.drop_index("market_data_batches_lookup_idx", table_name="market_data_batches", schema=MARKET_DATA_SCHEMA)
    op.drop_table("market_data_batches", schema=MARKET_DATA_SCHEMA)
