from __future__ import annotations

from datetime import UTC, datetime
import unittest

from market_data_service.application.services.gap_scan_service import GapScanCommand, GapScanService
from market_data_service.domain.enums import CandleRangeStatus, MarketDataSource, ProviderSymbolStatus
from market_data_service.domain.symbol_registry_models import ProviderSymbol


class GapScanServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_scan_reports_gaps_without_creating_backfill_by_default(self) -> None:
        backfill_requester = FakeBackfillRequester()
        service = GapScanService(
            symbol_registry=FakeSymbolRegistry(),
            candle_reader=FakeCandleReader(
                open_times=(datetime(2026, 7, 14, 0, tzinfo=UTC), datetime(2026, 7, 14, 2, tzinfo=UTC))
            ),
            backfill_requester=backfill_requester,
        )

        result = await service.scan(_command(create_backfill=False))

        self.assertFalse(result.create_backfill)
        self.assertEqual(result.total_gap_count, 1)
        self.assertEqual(result.total_backfill_requested_count, 0)
        self.assertEqual(backfill_requester.events, [])
        report = result.reports[0]
        self.assertEqual(report.status, CandleRangeStatus.GAP_DETECTED)
        self.assertEqual(report.expected_count, 3)
        self.assertEqual(report.actual_count, 2)
        self.assertEqual(report.missing_intervals[0].open_time, datetime(2026, 7, 14, 1, tzinfo=UTC))

    async def test_scan_creates_backfill_only_when_explicitly_requested(self) -> None:
        backfill_requester = FakeBackfillRequester()
        service = GapScanService(
            symbol_registry=FakeSymbolRegistry(),
            candle_reader=FakeCandleReader(open_times=(datetime(2026, 7, 14, 0, tzinfo=UTC),)),
            backfill_requester=backfill_requester,
        )

        result = await service.scan(_command(create_backfill=True))

        self.assertTrue(result.create_backfill)
        self.assertEqual(result.total_gap_count, 2)
        self.assertEqual(result.total_backfill_requested_count, 2)
        self.assertEqual(len(backfill_requester.events), 2)
        self.assertEqual(backfill_requester.events[0].provider_symbol, "ETHUSDT")
        self.assertEqual(backfill_requester.events[0].requested_from, datetime(2026, 7, 14, 1, tzinfo=UTC))


def _command(*, create_backfill: bool) -> GapScanCommand:
    return GapScanCommand(
        source=MarketDataSource.BINANCE_SPOT,
        provider_symbols=("ETHUSDT",),
        timeframes=("1h",),
        from_time=datetime(2026, 7, 14, 0, tzinfo=UTC),
        to_time=datetime(2026, 7, 14, 3, tzinfo=UTC),
        create_backfill=create_backfill,
    )


class FakeSymbolRegistry:
    async def get_provider_symbol(self, source: MarketDataSource, provider_symbol: str) -> ProviderSymbol:
        return ProviderSymbol(
            source=source,
            canonical_symbol="ETH/USDT",
            provider_symbol=provider_symbol,
            status=ProviderSymbolStatus.TRADING,
            supported_timeframes=("1h", "4h", "1d"),
            max_backfill_days=1095,
            metadata={},
        )


class FakeCandleReader:
    def __init__(self, *, open_times: tuple[datetime, ...]) -> None:
        self.open_times = open_times

    async def list_closed_candle_open_times(self, **kwargs) -> tuple[datetime, ...]:
        return self.open_times


class FakeBackfillRequester:
    def __init__(self) -> None:
        self.events = []

    async def request_backfill(self, event) -> bool:
        self.events.append(event)
        return True
