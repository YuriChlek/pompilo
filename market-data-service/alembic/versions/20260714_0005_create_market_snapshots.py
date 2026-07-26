from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0005_create_market_snapshots"
down_revision = "0004_create_market_data_batches"
branch_labels = None
depends_on = None

MARKET_DATA_SCHEMA = "_market_data"


def upgrade() -> None:
    op.create_table(
        "market_snapshots",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("canonical_symbol", sa.Text(), nullable=False),
        sa.Column("timeframe", sa.Text(), nullable=False),
        sa.Column("last_closed_candle_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lookback_start_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lookback_end_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("candle_count", sa.Integer(), nullable=False),
        sa.Column("data_hash", sa.Text(), nullable=False),
        sa.Column("batch_id", sa.Text(), nullable=False),
        sa.Column("completeness_status", sa.Text(), nullable=False),
        sa.Column("snapshot_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint(
            "source",
            "canonical_symbol",
            "timeframe",
            "lookback_start_time",
            "lookback_end_time",
            "data_hash",
            "completeness_status",
            name="market_snapshots_logical_data_hash_uq",
        ),
        sa.UniqueConstraint(
            "source",
            "canonical_symbol",
            "timeframe",
            "lookback_start_time",
            "lookback_end_time",
            "snapshot_version",
            name="market_snapshots_logical_version_uq",
        ),
        schema=MARKET_DATA_SCHEMA,
    )
    op.create_index(
        "market_snapshots_lookup_idx",
        "market_snapshots",
        ["source", "canonical_symbol", "timeframe", "lookback_end_time"],
        schema=MARKET_DATA_SCHEMA,
    )

    op.create_table(
        "market_snapshot_candles",
        sa.Column("snapshot_id", sa.Text(), nullable=False),
        sa.Column("candle_id", sa.Text(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("candle_hash_at_snapshot", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            [f"{MARKET_DATA_SCHEMA}.market_snapshots.id"],
            name="market_snapshot_candles_snapshot_id_fk",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("snapshot_id", "ordinal", name="market_snapshot_candles_pk"),
        sa.UniqueConstraint("snapshot_id", "candle_id", name="market_snapshot_candles_snapshot_id_candle_id_uq"),
        schema=MARKET_DATA_SCHEMA,
    )


def downgrade() -> None:
    op.drop_table("market_snapshot_candles", schema=MARKET_DATA_SCHEMA)
    op.drop_index("market_snapshots_lookup_idx", table_name="market_snapshots", schema=MARKET_DATA_SCHEMA)
    op.drop_table("market_snapshots", schema=MARKET_DATA_SCHEMA)
