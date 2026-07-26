from __future__ import annotations

from datetime import UTC, datetime
import unittest

from market_data_service.domain.enums import CandleRangeStatus, MarketDataSource
from market_data_service.domain.events.candle_batch_ready import (
    CANDLE_BATCH_READY_EVENT_TYPE,
    CandleBatchReady,
    build_candle_batch_ready_idempotency_key,
)
from market_data_service.domain.snapshot_models import MarketSnapshot


class CandleBatchReadyEventTests(unittest.TestCase):
    def test_idempotency_key_uses_logical_snapshot_version_not_snapshot_id(self) -> None:
        first = _snapshot(snapshot_id="snapshot-a", snapshot_version=1)
        retry = _snapshot(snapshot_id="snapshot-b", snapshot_version=1)
        corrected = _snapshot(snapshot_id="snapshot-c", snapshot_version=2)

        self.assertEqual(
            build_candle_batch_ready_idempotency_key(first),
            build_candle_batch_ready_idempotency_key(retry),
        )
        self.assertNotEqual(
            build_candle_batch_ready_idempotency_key(first),
            build_candle_batch_ready_idempotency_key(corrected),
        )

    def test_event_payload_contains_replay_identifiers(self) -> None:
        event = CandleBatchReady.from_snapshot(
            _snapshot(snapshot_id="snapshot-a", snapshot_version=1),
            occurred_at=datetime(2026, 7, 14, 10, tzinfo=UTC),
        )

        payload = event.payload_json()

        self.assertEqual(event.event_type, CANDLE_BATCH_READY_EVENT_TYPE)
        self.assertEqual(payload["snapshot_id"], "snapshot-a")
        self.assertEqual(payload["snapshot_version"], 1)
        self.assertEqual(payload["last_closed_candle_time"], "2026-07-14T09:00:00+00:00")
        self.assertNotIn("candles", payload)


def _snapshot(*, snapshot_id: str, snapshot_version: int) -> MarketSnapshot:
    return MarketSnapshot(
        id=snapshot_id,
        source=MarketDataSource.BINANCE_SPOT,
        canonical_symbol="ETH/USDT",
        timeframe="1h",
        last_closed_candle_time=datetime(2026, 7, 14, 9, tzinfo=UTC),
        lookback_start_time=datetime(2026, 7, 14, 8, tzinfo=UTC),
        lookback_end_time=datetime(2026, 7, 14, 9, tzinfo=UTC),
        candle_count=1,
        data_hash="data-hash",
        batch_id="batch-1",
        completeness_status=CandleRangeStatus.COMPLETE,
        snapshot_version=snapshot_version,
        created_at=datetime(2026, 7, 14, 9, tzinfo=UTC),
    )
