from __future__ import annotations

from pathlib import Path


def test_collection_state_and_lookup_indexes_migration_exists() -> None:
    migration = Path(
        "market-data-service/alembic/versions/20260718_0009_create_collection_states_and_indexes.py"
    ).read_text()

    assert "collection_states" in migration
    assert "market_candles_source_symbol_timeframe_open_time_idx" in migration
    assert '"source", "canonical_symbol", "timeframe", "open_time"' in migration
    assert "market_snapshots_latest_complete_idx" in migration
    assert "last_closed_candle_time DESC" in migration
