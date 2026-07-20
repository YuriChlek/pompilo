from __future__ import annotations

from collections.abc import Iterable


def candle_range_lookup_explain_query() -> str:
    """Return representative EXPLAIN query for indexed candle range lookup."""

    return """
EXPLAIN (ANALYZE, BUFFERS)
SELECT candle_id, open_time, close_time, open, high, low, close, volume
FROM _market_data.market_candles
WHERE source = :source
  AND canonical_symbol = :canonical_symbol
  AND timeframe = :timeframe
  AND open_time >= :from_time
  AND open_time < :to_time
ORDER BY open_time ASC
""".strip()


def latest_complete_snapshot_explain_query() -> str:
    """Return representative EXPLAIN query for latest complete snapshot lookup."""

    return """
EXPLAIN (ANALYZE, BUFFERS)
SELECT id, source, canonical_symbol, timeframe, last_closed_candle_time, snapshot_version, created_at
FROM _market_data.market_snapshots
WHERE source = :source
  AND canonical_symbol = :canonical_symbol
  AND timeframe = :timeframe
  AND completeness_status = 'COMPLETE'
ORDER BY last_closed_candle_time DESC, snapshot_version DESC, created_at DESC
LIMIT 1
""".strip()


def snapshot_membership_explain_query() -> str:
    """Return representative EXPLAIN query for ordered snapshot membership lookup."""

    return """
EXPLAIN (ANALYZE, BUFFERS)
SELECT mc.candle_id, mc.open_time, mc.close_time, mc.open, mc.high, mc.low, mc.close, mc.volume
FROM _market_data.market_snapshot_candles msc
JOIN _market_data.market_candles mc
  ON mc.candle_id = msc.candle_id
WHERE msc.snapshot_id = :snapshot_id
  AND mc.source = :source
  AND mc.canonical_symbol = :canonical_symbol
  AND mc.timeframe = :timeframe
ORDER BY msc.ordinal ASC
""".strip()


def assert_no_sequential_scan(plan_rows: Iterable[object]) -> None:
    """Raise when a representative EXPLAIN plan contains a sequential scan."""

    rendered_plan = "\n".join(_plan_row_to_text(row) for row in plan_rows)
    if "Seq Scan" in rendered_plan:
        raise AssertionError(f"Representative read query used a sequential scan:\n{rendered_plan}")


def _plan_row_to_text(row: object) -> str:
    if isinstance(row, str):
        return row
    if isinstance(row, tuple) and row:
        return str(row[0])
    return str(row)
