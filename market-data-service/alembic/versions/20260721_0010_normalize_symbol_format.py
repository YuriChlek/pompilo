from __future__ import annotations

from alembic import op

revision = "0010_normalize_symbol_format"
down_revision = "0009_collection_states_indexes"
branch_labels = None
depends_on = None

MARKET_DATA_SCHEMA = "_market_data"


def upgrade() -> None:
    op.execute(
        f"""
        INSERT INTO {MARKET_DATA_SCHEMA}.market_symbols (canonical_symbol, base_asset, quote_asset, status)
        SELECT replace(canonical_symbol, '/', ''), base_asset, quote_asset, status
        FROM {MARKET_DATA_SCHEMA}.market_symbols
        WHERE canonical_symbol LIKE '%/%'
        ON CONFLICT (canonical_symbol) DO UPDATE
        SET base_asset = excluded.base_asset,
            quote_asset = excluded.quote_asset,
            status = excluded.status,
            updated_at = now()
        """
    )
    _normalize_provider_symbols()
    _normalize_canonical_symbol_column("market_candles", conflict_predicate="target.open_time = source.open_time")
    _normalize_canonical_symbol_column("market_data_batches", conflict_predicate="target.requested_to = source.requested_to")
    _normalize_canonical_symbol_column(
        "market_snapshots",
        conflict_predicate="""
            target.lookback_start_time = source.lookback_start_time
            AND target.lookback_end_time = source.lookback_end_time
            AND target.snapshot_version = source.snapshot_version
        """,
    )
    _normalize_canonical_symbol_column("collection_states", conflict_predicate="target.timeframe = source.timeframe")
    op.execute(
        f"""
        DELETE FROM {MARKET_DATA_SCHEMA}.market_symbols old_symbol
        USING {MARKET_DATA_SCHEMA}.market_symbols normalized_symbol
        WHERE old_symbol.canonical_symbol LIKE '%/%'
          AND normalized_symbol.canonical_symbol = replace(old_symbol.canonical_symbol, '/', '')
        """
    )
    op.execute(
        f"""
        UPDATE {MARKET_DATA_SCHEMA}.market_symbols
        SET canonical_symbol = replace(canonical_symbol, '/', ''),
            updated_at = now()
        WHERE canonical_symbol LIKE '%/%'
        """
    )


def downgrade() -> None:
    # Symbol format normalization is intentionally one-way. Reintroducing slash
    # symbols would split current candle/snapshot ownership.
    return None


def _normalize_provider_symbols() -> None:
    op.execute(
        f"""
        DELETE FROM {MARKET_DATA_SCHEMA}.provider_symbols old_symbol
        USING {MARKET_DATA_SCHEMA}.provider_symbols normalized_symbol
        WHERE old_symbol.canonical_symbol LIKE '%/%'
          AND normalized_symbol.source = old_symbol.source
          AND normalized_symbol.canonical_symbol = replace(old_symbol.canonical_symbol, '/', '')
        """
    )
    op.execute(
        f"""
        UPDATE {MARKET_DATA_SCHEMA}.provider_symbols
        SET canonical_symbol = replace(canonical_symbol, '/', ''),
            provider_symbol = replace(provider_symbol, '/', ''),
            updated_at = now()
        WHERE canonical_symbol LIKE '%/%'
        """
    )


def _normalize_canonical_symbol_column(table_name: str, *, conflict_predicate: str) -> None:
    op.execute(
        f"""
        DELETE FROM {MARKET_DATA_SCHEMA}.{table_name} source
        USING {MARKET_DATA_SCHEMA}.{table_name} target
        WHERE source.canonical_symbol LIKE '%/%'
          AND target.source = source.source
          AND target.canonical_symbol = replace(source.canonical_symbol, '/', '')
          AND target.timeframe = source.timeframe
          AND {conflict_predicate}
        """
    )
    op.execute(
        f"""
        UPDATE {MARKET_DATA_SCHEMA}.{table_name}
        SET canonical_symbol = replace(canonical_symbol, '/', '')
        WHERE canonical_symbol LIKE '%/%'
        """
    )
