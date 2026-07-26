from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
import unittest

from market_data_service.application.services.snapshot_read_service import (
    SNAPSHOT_CONTRACT_VERSION,
    LatestSnapshotQuery,
    SnapshotReadService,
)
from market_data_service.domain.candle_models import CanonicalCandle
from market_data_service.domain.enums import CandleRangeStatus, MarketDataSource, ProviderSymbolStatus
from market_data_service.domain.snapshot_models import MarketSnapshot
from market_data_service.domain.symbol_registry_models import ProviderSymbol


class SnapshotReadServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_latest_complete_snapshot_returns_versioned_ready_contract(self) -> None:
        snapshot = _snapshot(last_closed_candle_time=datetime(2026, 7, 14, 11, tzinfo=UTC))
        service = SnapshotReadService(
            symbol_registry=FakeSymbolRegistry(),
            snapshot_reader=FakeSnapshotReader(snapshot=snapshot, candles=(_candle(),)),
            now_provider=lambda: datetime(2026, 7, 14, 12, tzinfo=UTC),
        )

        result = await service.latest_complete_snapshot(_query(max_age_seconds=7200))

        self.assertEqual(result.contract_version, SNAPSHOT_CONTRACT_VERSION)
        self.assertEqual(result.status, "ready")
        payload = result.as_payload()
        self.assertEqual(payload["contract_version"], "market-snapshot.v1")
        self.assertEqual(payload["snapshot"]["snapshot_version"], 3)
        self.assertEqual(payload["candles"][0]["open"], "100")

    async def test_latest_complete_snapshot_returns_not_ready_without_snapshot(self) -> None:
        service = SnapshotReadService(
            symbol_registry=FakeSymbolRegistry(),
            snapshot_reader=FakeSnapshotReader(snapshot=None, candles=()),
            now_provider=lambda: datetime(2026, 7, 14, 12, tzinfo=UTC),
        )

        result = await service.latest_complete_snapshot(_query(max_age_seconds=7200))

        self.assertEqual(result.status, "not_ready")
        self.assertIsNone(result.snapshot)
        self.assertEqual(result.candles, ())

    async def test_latest_complete_snapshot_returns_stale_without_candles(self) -> None:
        snapshot = _snapshot(last_closed_candle_time=datetime(2026, 7, 14, 8, tzinfo=UTC))
        service = SnapshotReadService(
            symbol_registry=FakeSymbolRegistry(),
            snapshot_reader=FakeSnapshotReader(snapshot=snapshot, candles=(_candle(),)),
            now_provider=lambda: datetime(2026, 7, 14, 12, tzinfo=UTC),
        )

        result = await service.latest_complete_snapshot(_query(max_age_seconds=3600))

        self.assertEqual(result.status, "stale")
        self.assertEqual(result.candles, ())

    async def test_snapshot_by_id_returns_ready_contract_without_latest_lookup(self) -> None:
        snapshot = _snapshot(last_closed_candle_time=datetime(2026, 7, 14, 11, tzinfo=UTC))
        reader = FakeSnapshotReader(snapshot=snapshot, candles=(_candle(),))
        service = SnapshotReadService(
            symbol_registry=FakeSymbolRegistry(),
            snapshot_reader=reader,
            now_provider=lambda: datetime(2026, 7, 14, 12, tzinfo=UTC),
        )

        result = await service.snapshot_by_id("snapshot-1")

        self.assertEqual(result.status, "ready")
        self.assertEqual(result.snapshot.id, "snapshot-1")
        self.assertEqual(result.candles[0].candle_id, "candle-1")
        self.assertEqual(reader.snapshot_ids, ["snapshot-1"])
        self.assertEqual(reader.latest_calls, 0)

    async def test_snapshot_by_id_returns_not_ready_for_missing_snapshot(self) -> None:
        service = SnapshotReadService(
            symbol_registry=FakeSymbolRegistry(),
            snapshot_reader=FakeSnapshotReader(snapshot=None, candles=()),
            now_provider=lambda: datetime(2026, 7, 14, 12, tzinfo=UTC),
        )

        result = await service.snapshot_by_id("missing-snapshot")

        self.assertEqual(result.status, "not_ready")
        self.assertEqual(result.reason, "snapshot is not available")


def _query(*, max_age_seconds: int | None) -> LatestSnapshotQuery:
    return LatestSnapshotQuery(
        source=MarketDataSource.BINANCE_SPOT,
        provider_symbol="ETHUSDT",
        timeframe="1h",
        max_age_seconds=max_age_seconds,
    )


def _snapshot(*, last_closed_candle_time: datetime) -> MarketSnapshot:
    return MarketSnapshot(
        id="snapshot-1",
        source=MarketDataSource.BINANCE_SPOT,
        canonical_symbol="ETH/USDT",
        timeframe="1h",
        last_closed_candle_time=last_closed_candle_time,
        lookback_start_time=last_closed_candle_time - timedelta(hours=2),
        lookback_end_time=last_closed_candle_time,
        candle_count=2,
        data_hash="hash",
        batch_id="batch-1",
        completeness_status=CandleRangeStatus.COMPLETE,
        snapshot_version=3,
        created_at=last_closed_candle_time,
    )


def _candle() -> CanonicalCandle:
    return CanonicalCandle(
        candle_id="candle-1",
        source=MarketDataSource.BINANCE_SPOT,
        canonical_symbol="ETH/USDT",
        provider_symbol="ETHUSDT",
        timeframe="1h",
        open_time=datetime(2026, 7, 14, 10, tzinfo=UTC),
        close_time=datetime(2026, 7, 14, 11, tzinfo=UTC),
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
        trades_count=42,
        is_closed=True,
        provider_payload_hash="payload-hash",
    )


class FakeSymbolRegistry:
    async def get_provider_symbol(self, source: MarketDataSource, provider_symbol: str) -> ProviderSymbol:
        return ProviderSymbol(
            source=source,
            canonical_symbol="ETH/USDT",
            provider_symbol=provider_symbol,
            status=ProviderSymbolStatus.TRADING,
            supported_timeframes=("1h",),
        )


class FakeSnapshotReader:
    def __init__(self, *, snapshot: MarketSnapshot | None, candles: tuple[CanonicalCandle, ...]) -> None:
        self.snapshot = snapshot
        self.candles = candles
        self.snapshot_ids: list[str] = []
        self.latest_calls = 0

    async def get_snapshot(self, snapshot_id: str) -> MarketSnapshot | None:
        self.snapshot_ids.append(snapshot_id)
        return self.snapshot

    async def get_latest_complete_snapshot(self, **kwargs) -> MarketSnapshot | None:
        self.latest_calls += 1
        return self.snapshot

    async def read_snapshot_candles(self, snapshot_id: str) -> list[CanonicalCandle]:
        return list(self.candles)
