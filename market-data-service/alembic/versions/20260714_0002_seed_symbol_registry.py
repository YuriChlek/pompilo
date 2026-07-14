from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002_seed_symbol_registry"
down_revision = "0001_create_market_data_schema_shell"
branch_labels = None
depends_on = None

MARKET_DATA_SCHEMA = "_market_data"
BINANCE_SPOT = "BINANCE_SPOT"
SUPPORTED_TIMEFRAMES = ["1h", "4h", "1d"]
MAX_BACKFILL_DAYS = 1095

MARKET_SYMBOL_ROWS = [
    {"canonical_symbol": "BTC/USDT", "base_asset": "BTC", "quote_asset": "USDT", "status": "ACTIVE"},
    {"canonical_symbol": "ETH/USDT", "base_asset": "ETH", "quote_asset": "USDT", "status": "ACTIVE"},
    {"canonical_symbol": "LTC/USDT", "base_asset": "LTC", "quote_asset": "USDT", "status": "ACTIVE"},
    {"canonical_symbol": "SOL/USDT", "base_asset": "SOL", "quote_asset": "USDT", "status": "ACTIVE"},
    {"canonical_symbol": "SUI/USDT", "base_asset": "SUI", "quote_asset": "USDT", "status": "ACTIVE"},
    {"canonical_symbol": "TAO/USDT", "base_asset": "TAO", "quote_asset": "USDT", "status": "ACTIVE"},
    {"canonical_symbol": "XRP/USDT", "base_asset": "XRP", "quote_asset": "USDT", "status": "ACTIVE"},
]

PROVIDER_SYMBOL_ROWS = [
    {
        "source": BINANCE_SPOT,
        "canonical_symbol": row["canonical_symbol"],
        "provider_symbol": row["canonical_symbol"].replace("/", ""),
        "status": "TRADING",
        "supported_timeframes": SUPPORTED_TIMEFRAMES,
        "min_available_time": None,
        "max_backfill_days": MAX_BACKFILL_DAYS,
        "metadata_json": {},
    }
    for row in MARKET_SYMBOL_ROWS
]


def _market_symbols_table() -> sa.Table:
    return sa.table(
        "market_symbols",
        sa.column("canonical_symbol", sa.Text()),
        sa.column("base_asset", sa.Text()),
        sa.column("quote_asset", sa.Text()),
        sa.column("status", sa.Text()),
        schema=MARKET_DATA_SCHEMA,
    )


def _provider_symbols_table() -> sa.Table:
    return sa.table(
        "provider_symbols",
        sa.column("source", sa.Text()),
        sa.column("canonical_symbol", sa.Text()),
        sa.column("provider_symbol", sa.Text()),
        sa.column("status", sa.Text()),
        sa.column("supported_timeframes", postgresql.ARRAY(sa.Text())),
        sa.column("min_available_time", sa.DateTime(timezone=True)),
        sa.column("max_backfill_days", sa.Integer()),
        sa.column("metadata_json", postgresql.JSONB()),
        schema=MARKET_DATA_SCHEMA,
    )


def upgrade() -> None:
    op.add_column("market_symbols", sa.Column("status", sa.Text(), nullable=True), schema=MARKET_DATA_SCHEMA)
    op.execute(
        f"""
        UPDATE {MARKET_DATA_SCHEMA}.market_symbols
        SET status = CASE WHEN is_active THEN 'ACTIVE' ELSE 'PAUSED' END
        """
    )
    op.alter_column("market_symbols", "status", nullable=False, schema=MARKET_DATA_SCHEMA)
    op.drop_column("market_symbols", "is_active", schema=MARKET_DATA_SCHEMA)

    op.add_column("provider_symbols", sa.Column("status", sa.Text(), nullable=True), schema=MARKET_DATA_SCHEMA)
    op.add_column(
        "provider_symbols",
        sa.Column("supported_timeframes", postgresql.ARRAY(sa.Text()), nullable=True),
        schema=MARKET_DATA_SCHEMA,
    )
    op.add_column(
        "provider_symbols",
        sa.Column("min_available_time", sa.DateTime(timezone=True), nullable=True),
        schema=MARKET_DATA_SCHEMA,
    )
    op.add_column(
        "provider_symbols",
        sa.Column("max_backfill_days", sa.Integer(), nullable=True),
        schema=MARKET_DATA_SCHEMA,
    )
    op.add_column(
        "provider_symbols",
        sa.Column("metadata_json", postgresql.JSONB(), nullable=True),
        schema=MARKET_DATA_SCHEMA,
    )
    op.execute(
        f"""
        UPDATE {MARKET_DATA_SCHEMA}.provider_symbols
        SET status = CASE WHEN is_active THEN 'TRADING' ELSE 'HALTED' END,
            supported_timeframes = ARRAY['1h','4h','1d'],
            metadata_json = '{{}}'::jsonb
        """
    )
    op.alter_column("provider_symbols", "status", nullable=False, schema=MARKET_DATA_SCHEMA)
    op.alter_column("provider_symbols", "supported_timeframes", nullable=False, schema=MARKET_DATA_SCHEMA)
    op.alter_column("provider_symbols", "metadata_json", nullable=False, schema=MARKET_DATA_SCHEMA)
    op.drop_column("provider_symbols", "is_active", schema=MARKET_DATA_SCHEMA)

    market_symbols_insert = postgresql.insert(_market_symbols_table()).values(MARKET_SYMBOL_ROWS)
    op.execute(
        market_symbols_insert.on_conflict_do_update(
            index_elements=["canonical_symbol"],
            set_={
                "base_asset": market_symbols_insert.excluded.base_asset,
                "quote_asset": market_symbols_insert.excluded.quote_asset,
                "status": market_symbols_insert.excluded.status,
            },
        )
    )

    provider_symbols_insert = postgresql.insert(_provider_symbols_table()).values(PROVIDER_SYMBOL_ROWS)
    op.execute(
        provider_symbols_insert.on_conflict_do_update(
            index_elements=["source", "canonical_symbol"],
            set_={
                "provider_symbol": provider_symbols_insert.excluded.provider_symbol,
                "status": provider_symbols_insert.excluded.status,
                "supported_timeframes": provider_symbols_insert.excluded.supported_timeframes,
                "min_available_time": provider_symbols_insert.excluded.min_available_time,
                "max_backfill_days": provider_symbols_insert.excluded.max_backfill_days,
                "metadata_json": provider_symbols_insert.excluded.metadata_json,
            },
        )
    )


def downgrade() -> None:
    provider_symbols = _provider_symbols_table()
    market_symbols = _market_symbols_table()

    op.execute(
        provider_symbols.delete().where(
            provider_symbols.c.source == BINANCE_SPOT,
            provider_symbols.c.provider_symbol.in_([row["provider_symbol"] for row in PROVIDER_SYMBOL_ROWS]),
        )
    )
    op.execute(
        market_symbols.delete().where(
            market_symbols.c.canonical_symbol.in_([row["canonical_symbol"] for row in MARKET_SYMBOL_ROWS]),
        )
    )

    op.add_column(
        "provider_symbols",
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        schema=MARKET_DATA_SCHEMA,
    )
    op.drop_column("provider_symbols", "metadata_json", schema=MARKET_DATA_SCHEMA)
    op.drop_column("provider_symbols", "max_backfill_days", schema=MARKET_DATA_SCHEMA)
    op.drop_column("provider_symbols", "min_available_time", schema=MARKET_DATA_SCHEMA)
    op.drop_column("provider_symbols", "supported_timeframes", schema=MARKET_DATA_SCHEMA)
    op.drop_column("provider_symbols", "status", schema=MARKET_DATA_SCHEMA)

    op.add_column(
        "market_symbols",
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        schema=MARKET_DATA_SCHEMA,
    )
    op.drop_column("market_symbols", "status", schema=MARKET_DATA_SCHEMA)
