from __future__ import annotations

from datetime import UTC, datetime
import unittest

from market_data_service.domain.enums import MarketDataSource
from market_data_service.domain.events.market_data_backfill_requested import (
    MARKET_DATA_BACKFILL_REQUESTED_EVENT_TYPE,
    MarketDataBackfillRequested,
    build_market_data_backfill_requested_idempotency_key,
)
from market_data_service.domain.scheduler_models import BACKFILL_SYNC_PRIORITY


class MarketDataBackfillRequestedEventTests(unittest.TestCase):
    def test_from_gap_builds_normalized_event_payload(self) -> None:
        event = MarketDataBackfillRequested.from_gap(
            source=MarketDataSource.BINANCE_SPOT,
            canonical_symbol="ETH/USDT",
            provider_symbol=" ethusdt ",
            timeframe=" 1H ",
            requested_from=datetime(2026, 7, 14, 9, tzinfo=UTC),
            requested_to=datetime(2026, 7, 14, 10, tzinfo=UTC),
            parent_batch_id="batch-1",
            occurred_at=datetime(2026, 7, 14, 10, 1, tzinfo=UTC),
        )

        self.assertEqual(event.event_type, MARKET_DATA_BACKFILL_REQUESTED_EVENT_TYPE)
        self.assertEqual(event.provider_symbol, "ETHUSDT")
        self.assertEqual(event.timeframe, "1h")
        self.assertEqual(event.reason, "GAP_DETECTED")
        self.assertEqual(event.priority, BACKFILL_SYNC_PRIORITY)
        self.assertEqual(
            event.idempotency_key,
            "BINANCE_SPOT|ETHUSDT|1h|2026-07-14T09:00:00+00:00|2026-07-14T10:00:00+00:00",
        )
        self.assertEqual(event.payload_json()["parent_batch_id"], "batch-1")

    def test_idempotency_key_is_stable(self) -> None:
        key = build_market_data_backfill_requested_idempotency_key(
            source=MarketDataSource.BINANCE_SPOT,
            provider_symbol="ethusdt",
            timeframe="1H",
            requested_from=datetime(2026, 7, 14, 9, tzinfo=UTC),
            requested_to=datetime(2026, 7, 14, 10, tzinfo=UTC),
        )

        self.assertEqual(key, "BINANCE_SPOT|ETHUSDT|1h|2026-07-14T09:00:00+00:00|2026-07-14T10:00:00+00:00")
