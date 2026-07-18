from __future__ import annotations

from datetime import UTC, datetime
import unittest

from market_data_service.domain.enums import CandleRangeStatus, MarketDataSource
from market_data_service.domain.events.market_data_candles_collected import (
    MARKET_DATA_CANDLES_COLLECTED_CONTRACT_VERSION,
    MARKET_DATA_CANDLES_COLLECTED_EVENT_TYPE,
    MarketDataCandlesCollectedEvent,
    build_market_data_candles_collected_idempotency_key,
)
from market_data_service.domain.snapshot_models import MarketSnapshot


class MarketDataCandlesCollectedEventTests(unittest.TestCase):
    def test_idempotency_key_format(self) -> None:
        key = build_market_data_candles_collected_idempotency_key(
            source=MarketDataSource.BINANCE_SPOT,
            provider_symbol="BTCUSDT",
            timeframe="1h",
            closed_at=datetime(2026, 7, 16, 1, 0, 0, tzinfo=UTC),
        )
        self.assertEqual(key, "BINANCE_SPOT:BTCUSDT:1h:2026-07-16T01:00:00Z")

    def test_event_payload_conforms_to_contract(self) -> None:
        snapshot = MarketSnapshot(
            id="snapshot-123",
            source=MarketDataSource.BINANCE_SPOT,
            canonical_symbol="BTC/USDT",
            timeframe="1h",
            last_closed_candle_time=datetime(2026, 7, 16, 1, 0, 0, tzinfo=UTC),
            lookback_start_time=datetime(2026, 7, 16, 0, 0, 0, tzinfo=UTC),
            lookback_end_time=datetime(2026, 7, 16, 1, 0, 0, tzinfo=UTC),
            candle_count=1,
            data_hash="abc",
            batch_id="batch-456",
            completeness_status=CandleRangeStatus.COMPLETE,
            snapshot_version=1,
            created_at=datetime(2026, 7, 16, 1, 5, 0, tzinfo=UTC),
        )

        event = MarketDataCandlesCollectedEvent.from_snapshot(
            snapshot,
            provider_symbol="BTCUSDT",
            occurred_at=datetime(2026, 7, 16, 1, 10, 0, tzinfo=UTC),
        )

        payload = event.payload_json()

        self.assertEqual(payload["event_type"], MARKET_DATA_CANDLES_COLLECTED_EVENT_TYPE)
        self.assertEqual(payload["contract_version"], MARKET_DATA_CANDLES_COLLECTED_CONTRACT_VERSION)
        self.assertEqual(payload["source"], "BINANCE_SPOT")
        self.assertEqual(payload["symbol"], "BTC/USDT")
        self.assertEqual(payload["provider_symbol"], "BTCUSDT")
        self.assertEqual(payload["timeframe"], "1h")
        self.assertEqual(payload["from"], "2026-07-16T00:00:00Z")
        self.assertEqual(payload["to"], "2026-07-16T01:00:00Z")
        self.assertEqual(payload["batch_id"], "batch-456")
        self.assertEqual(payload["snapshot_id"], "snapshot-123")
        self.assertEqual(payload["closed_at"], "2026-07-16T01:00:00Z")
        self.assertEqual(payload["idempotency_key"], "BINANCE_SPOT:BTCUSDT:1h:2026-07-16T01:00:00Z")
