from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
import unittest

from market_data_service.domain.candle_models import CanonicalCandle
from market_data_service.domain.enums import MarketDataSource
from market_data_service.domain.snapshot_hash import build_snapshot_membership, calculate_snapshot_data_hash


def _candle(open_time: datetime, *, candle_id: str, provider_payload_hash: str) -> CanonicalCandle:
    return CanonicalCandle(
        candle_id=candle_id,
        source=MarketDataSource.BINANCE_SPOT,
        canonical_symbol="ETH/USDT",
        provider_symbol="ETHUSDT",
        timeframe="1h",
        open_time=open_time,
        close_time=open_time + timedelta(hours=1),
        open=Decimal("100"),
        high=Decimal("110"),
        low=Decimal("90"),
        close=Decimal("105"),
        volume=Decimal("10"),
        quote_volume=Decimal("1050"),
        taker_buy_base_volume=None,
        taker_buy_quote_volume=None,
        taker_sell_base_volume=None,
        taker_sell_quote_volume=None,
        trades_count=None,
        is_closed=True,
        provider_payload_hash=provider_payload_hash,
    )


class SnapshotHashTests(unittest.TestCase):
    def test_snapshot_hash_is_stable_for_same_ordered_candle_membership(self) -> None:
        candles = (
            _candle(datetime(2026, 7, 14, 8, tzinfo=UTC), candle_id="candle-1", provider_payload_hash="hash-1"),
            _candle(datetime(2026, 7, 14, 9, tzinfo=UTC), candle_id="candle-2", provider_payload_hash="hash-2"),
        )

        self.assertEqual(calculate_snapshot_data_hash(candles), calculate_snapshot_data_hash(candles))

    def test_snapshot_hash_changes_when_candle_hash_changes(self) -> None:
        original = (
            _candle(datetime(2026, 7, 14, 8, tzinfo=UTC), candle_id="candle-1", provider_payload_hash="hash-1"),
        )
        corrected = (
            _candle(datetime(2026, 7, 14, 8, tzinfo=UTC), candle_id="candle-1", provider_payload_hash="hash-2"),
        )

        self.assertNotEqual(calculate_snapshot_data_hash(original), calculate_snapshot_data_hash(corrected))

    def test_snapshot_hash_includes_order(self) -> None:
        first = _candle(datetime(2026, 7, 14, 8, tzinfo=UTC), candle_id="candle-1", provider_payload_hash="hash-1")
        second = _candle(datetime(2026, 7, 14, 9, tzinfo=UTC), candle_id="candle-2", provider_payload_hash="hash-2")

        self.assertNotEqual(calculate_snapshot_data_hash((first, second)), calculate_snapshot_data_hash((second, first)))

    def test_membership_records_snapshot_hash_and_ordinal(self) -> None:
        candles = (
            _candle(datetime(2026, 7, 14, 8, tzinfo=UTC), candle_id="candle-1", provider_payload_hash="hash-1"),
            _candle(datetime(2026, 7, 14, 9, tzinfo=UTC), candle_id="candle-2", provider_payload_hash="hash-2"),
        )

        membership = build_snapshot_membership("snapshot-1", candles)

        self.assertEqual(membership[0].ordinal, 1)
        self.assertEqual(membership[1].ordinal, 2)
        self.assertEqual(membership[0].candle_hash_at_snapshot, "hash-1")
        self.assertEqual(membership[1].candle_id, "candle-2")
