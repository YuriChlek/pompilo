from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0009_collection_states_indexes"
down_revision = "e0fa34f7df6c"
branch_labels = None
depends_on = None

MARKET_DATA_SCHEMA = "_market_data"


def upgrade() -> None:
    op.create_table(
        "collection_states",
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("canonical_symbol", sa.Text(), nullable=False),
        sa.Column("provider_symbol", sa.Text(), nullable=False),
        sa.Column("timeframe", sa.Text(), nullable=False),
        sa.Column("bootstrap_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("bootstrap_to", sa.DateTime(timezone=True), nullable=False),
        sa.Column("bootstrap_next_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("bootstrap_completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_successful_close_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("source", "canonical_symbol", "timeframe", name="collection_states_pk"),
        schema=MARKET_DATA_SCHEMA,
    )
    op.create_index(
        "market_candles_source_symbol_timeframe_open_time_idx",
        "market_candles",
        ["source", "canonical_symbol", "timeframe", "open_time"],
        schema=MARKET_DATA_SCHEMA,
    )
    op.create_index(
        "market_snapshots_latest_complete_idx",
        "market_snapshots",
        [
            "source",
            "canonical_symbol",
            "timeframe",
            "completeness_status",
            sa.text("last_closed_candle_time DESC"),
            sa.text("snapshot_version DESC"),
            sa.text("created_at DESC"),
        ],
        schema=MARKET_DATA_SCHEMA,
    )


def downgrade() -> None:
    op.drop_index("market_snapshots_latest_complete_idx", table_name="market_snapshots", schema=MARKET_DATA_SCHEMA)
    op.drop_index(
        "market_candles_source_symbol_timeframe_open_time_idx",
        table_name="market_candles",
        schema=MARKET_DATA_SCHEMA,
    )
    op.drop_table("collection_states", schema=MARKET_DATA_SCHEMA)
