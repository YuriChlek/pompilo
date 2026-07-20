from __future__ import annotations

from pathlib import Path


def test_collection_state_and_lookup_indexes_migration_exists() -> None:
    migration_file = Path(__file__).parents[2] / "alembic" / "versions" / "20260718_0009_create_collection_states_and_indexes.py"
    migration = migration_file.read_text()
    candle_table_file = Path(__file__).parents[2] / "src" / "market_data_service" / "persistence" / "tables" / "market_candles_tables.py"
    candle_table = candle_table_file.read_text()
    snapshot_table_file = Path(__file__).parents[2] / "src" / "market_data_service" / "persistence" / "tables" / "market_snapshots_tables.py"
    snapshot_table = snapshot_table_file.read_text()

    assert "collection_states" in migration
    assert "market_candles_natural_key_uq" in candle_table
    assert '"source", "canonical_symbol", "timeframe", "open_time"' in candle_table
    assert "market_candles_source_symbol_timeframe_open_time_idx" not in migration
    assert "market_snapshots_latest_complete_idx" in migration
    assert "last_closed_candle_time DESC" in migration
    assert 'PrimaryKeyConstraint("snapshot_id", "ordinal"' in snapshot_table
    assert 'UniqueConstraint("snapshot_id", "candle_id"' in snapshot_table
