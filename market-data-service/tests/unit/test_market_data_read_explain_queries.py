from __future__ import annotations

import pytest

from market_data_service.persistence.queries.market_data_read_explain_queries import (
    assert_no_sequential_scan,
    candle_range_lookup_explain_query,
    latest_complete_snapshot_explain_query,
    snapshot_membership_explain_query,
)


def test_stage_23_candle_range_explain_query_matches_natural_key_index_path() -> None:
    query = candle_range_lookup_explain_query()

    assert query.startswith("EXPLAIN (ANALYZE, BUFFERS)")
    assert "FROM _market_data.market_candles" in query
    assert "source = :source" in query
    assert "canonical_symbol = :canonical_symbol" in query
    assert "timeframe = :timeframe" in query
    assert "open_time >= :from_time" in query
    assert "open_time < :to_time" in query
    assert "ORDER BY open_time ASC" in query


def test_stage_23_latest_snapshot_explain_query_matches_latest_complete_index_path() -> None:
    query = latest_complete_snapshot_explain_query()

    assert query.startswith("EXPLAIN (ANALYZE, BUFFERS)")
    assert "FROM _market_data.market_snapshots" in query
    assert "completeness_status = 'COMPLETE'" in query
    assert "ORDER BY last_closed_candle_time DESC, snapshot_version DESC, created_at DESC" in query
    assert "LIMIT 1" in query


def test_stage_23_snapshot_membership_explain_query_uses_snapshot_id_ordered_membership() -> None:
    query = snapshot_membership_explain_query()

    assert query.startswith("EXPLAIN (ANALYZE, BUFFERS)")
    assert "FROM _market_data.market_snapshot_candles msc" in query
    assert "JOIN _market_data.market_candles mc" in query
    assert "msc.snapshot_id = :snapshot_id" in query
    assert "ORDER BY msc.ordinal ASC" in query


def test_stage_23_query_plan_guard_rejects_sequential_scans() -> None:
    assert_no_sequential_scan(("Index Scan using market_snapshots_latest_complete_idx",))

    with pytest.raises(AssertionError, match="sequential scan"):
        assert_no_sequential_scan(("Seq Scan on market_snapshots",))
