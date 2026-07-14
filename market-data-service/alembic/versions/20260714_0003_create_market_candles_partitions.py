from __future__ import annotations

from alembic import op

revision = "0003_create_market_candles_partitions"
down_revision = "0002_seed_symbol_registry"
branch_labels = None
depends_on = None

MARKET_DATA_SCHEMA = "_market_data"
TIMEFRAME_PARTITIONS = {
    "1h": "market_candles_1h",
    "4h": "market_candles_4h",
    "1d": "market_candles_1d",
}


def upgrade() -> None:
    op.execute(
        f"""
        CREATE TABLE {MARKET_DATA_SCHEMA}.market_candles (
            candle_id TEXT NOT NULL,
            source TEXT NOT NULL,
            canonical_symbol TEXT NOT NULL,
            provider_symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            open_time TIMESTAMPTZ NOT NULL,
            close_time TIMESTAMPTZ NOT NULL,
            open NUMERIC NOT NULL,
            high NUMERIC NOT NULL,
            low NUMERIC NOT NULL,
            close NUMERIC NOT NULL,
            volume NUMERIC NOT NULL,
            quote_volume NUMERIC,
            taker_buy_base_volume NUMERIC,
            taker_buy_quote_volume NUMERIC,
            taker_sell_base_volume NUMERIC,
            taker_sell_quote_volume NUMERIC,
            trades_count INTEGER,
            is_closed BOOLEAN NOT NULL,
            provider_payload_hash TEXT NOT NULL,
            inserted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT market_candles_pk PRIMARY KEY (timeframe, candle_id),
            CONSTRAINT market_candles_natural_key_uq UNIQUE (source, canonical_symbol, timeframe, open_time)
        ) PARTITION BY LIST (timeframe);
        """
    )

    for timeframe, partition_name in TIMEFRAME_PARTITIONS.items():
        op.execute(
            f"""
            CREATE TABLE {MARKET_DATA_SCHEMA}.{partition_name}
            PARTITION OF {MARKET_DATA_SCHEMA}.market_candles
            FOR VALUES IN ('{timeframe}');
            """
        )

    op.create_index(
        "market_candles_symbol_timeframe_open_time_idx",
        "market_candles",
        ["canonical_symbol", "timeframe", "open_time"],
        schema=MARKET_DATA_SCHEMA,
    )


def downgrade() -> None:
    op.drop_index("market_candles_symbol_timeframe_open_time_idx", table_name="market_candles", schema=MARKET_DATA_SCHEMA)
    op.drop_table("market_candles", schema=MARKET_DATA_SCHEMA)
