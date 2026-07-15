from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from bot_platform_service.infrastructure.market_data import (
    MarketDataServiceSnapshotProvider,
    SnapshotNotReadyError,
    SnapshotStaleError,
    TimeframeUnsupportedError,
)


class _Mappings:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def one_or_none(self):
        return self.rows[0] if self.rows else None

    def all(self):
        return self.rows


class _Result:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def mappings(self) -> _Mappings:
        return _Mappings(self.rows)


class _Connection:
    def __init__(self, results: list[list[dict[str, object]]]) -> None:
        self.results = list(results)
        self.statements: list[object] = []

    async def execute(self, statement):
        self.statements.append(statement)
        return _Result(self.results.pop(0))


def test_get_latest_complete_snapshot_reconstructs_candles_in_snapshot_order() -> None:
    connection = _Connection([[_snapshot_row()], [_candle_row("candle-1", 0), _candle_row("candle-2", 1)]])
    provider = MarketDataServiceSnapshotProvider(connection)

    snapshot = asyncio.run(
        provider.get_latest_complete_snapshot(
            source="BINANCE_SPOT",
            canonical_symbol="ETHUSDT",
            timeframe="H1",
        )
    )

    assert snapshot.snapshot_id == "snapshot-1"
    assert snapshot.timeframe == "1h"
    assert snapshot.completeness_status == "COMPLETE"
    assert [candle.close for candle in snapshot.candles] == [Decimal("101"), Decimal("102")]
    assert len(connection.statements) == 2


def test_get_latest_complete_snapshot_raises_not_ready_for_missing_or_incomplete_snapshot() -> None:
    connection = _Connection([[]])
    provider = MarketDataServiceSnapshotProvider(connection)

    with pytest.raises(SnapshotNotReadyError) as exc_info:
        asyncio.run(
            provider.get_latest_complete_snapshot(
                source="BINANCE_SPOT",
                canonical_symbol="ETHUSDT",
                timeframe="1h",
            )
        )

    assert exc_info.value.error_code == "SNAPSHOT_NOT_READY"
    assert len(connection.statements) == 1


def test_get_latest_complete_snapshot_raises_stale_when_candle_hash_changed() -> None:
    stale_candle = _candle_row("candle-1", 0)
    stale_candle["provider_payload_hash"] = "changed-hash"
    connection = _Connection([[_snapshot_row(candle_count=1)], [stale_candle]])
    provider = MarketDataServiceSnapshotProvider(connection)

    with pytest.raises(SnapshotStaleError) as exc_info:
        asyncio.run(
            provider.get_latest_complete_snapshot(
                source="BINANCE_SPOT",
                canonical_symbol="ETHUSDT",
                timeframe="1h",
            )
        )

    assert exc_info.value.error_code == "SNAPSHOT_STALE"


def test_get_latest_complete_snapshot_raises_stale_when_membership_count_mismatches() -> None:
    connection = _Connection([[_snapshot_row(candle_count=2)], [_candle_row("candle-1", 0)]])
    provider = MarketDataServiceSnapshotProvider(connection)

    with pytest.raises(SnapshotStaleError):
        asyncio.run(
            provider.get_latest_complete_snapshot(
                source="BINANCE_SPOT",
                canonical_symbol="ETHUSDT",
                timeframe="1h",
            )
        )


def test_get_latest_complete_snapshot_raises_typed_error_for_unsupported_timeframe() -> None:
    provider = MarketDataServiceSnapshotProvider(_Connection([]))

    with pytest.raises(TimeframeUnsupportedError) as exc_info:
        asyncio.run(
            provider.get_latest_complete_snapshot(
                source="BINANCE_SPOT",
                canonical_symbol="ETHUSDT",
                timeframe="15m",
            )
        )

    assert exc_info.value.error_code == "TIMEFRAME_UNSUPPORTED"


def test_build_context_normalizes_supporting_timeframe_aliases() -> None:
    connection = _Connection(
        [
            [_snapshot_row(snapshot_id="snapshot-1", timeframe="1h", candle_count=1)],
            [_candle_row("candle-1", 0, timeframe="1h")],
            [_snapshot_row(snapshot_id="snapshot-2", timeframe="4h", candle_count=1)],
            [_candle_row("candle-2", 0, timeframe="4h")],
            [_snapshot_row(snapshot_id="snapshot-3", timeframe="1d", candle_count=1)],
            [_candle_row("candle-3", 0, timeframe="1d")],
        ]
    )
    provider = MarketDataServiceSnapshotProvider(connection)

    context = asyncio.run(
        provider.build_context(
            source="BINANCE_SPOT",
            canonical_symbol="ETHUSDT",
            primary_timeframe="H1",
            supporting_timeframes=("H4", "D1"),
        )
    )

    assert context.primary_snapshot.timeframe == "1h"
    assert [snapshot.timeframe for snapshot in context.supporting_snapshots] == ["4h", "1d"]


def _snapshot_row(*, snapshot_id: str = "snapshot-1", timeframe: str = "1h", candle_count: int = 2) -> dict[str, object]:
    now = datetime(2026, 7, 14, tzinfo=UTC)
    return {
        "id": snapshot_id,
        "source": "BINANCE_SPOT",
        "canonical_symbol": "ETHUSDT",
        "timeframe": timeframe,
        "last_closed_candle_time": now,
        "lookback_start_time": now,
        "lookback_end_time": now,
        "candle_count": candle_count,
        "data_hash": "snapshot-hash",
        "batch_id": "batch-1",
        "completeness_status": "COMPLETE",
        "snapshot_version": 1,
        "created_at": now,
    }


def _candle_row(candle_id: str, ordinal: int, *, timeframe: str = "1h") -> dict[str, object]:
    now = datetime(2026, 7, 14, tzinfo=UTC)
    candle_hash = f"hash-{candle_id}"
    return {
        "candle_hash_at_snapshot": candle_hash,
        "candle_id": candle_id,
        "ordinal": ordinal,
        "source": "BINANCE_SPOT",
        "canonical_symbol": "ETHUSDT",
        "provider_symbol": "ETHUSDT",
        "timeframe": timeframe,
        "open_time": now,
        "close_time": now,
        "open": Decimal("100"),
        "high": Decimal("110"),
        "low": Decimal("90"),
        "close": Decimal(str(101 + ordinal)),
        "volume": Decimal("10"),
        "provider_payload_hash": candle_hash,
    }
