from __future__ import annotations

from datetime import UTC, datetime
import unittest

from market_data_service.application.services.backfill_planning_service import BackfillPlanningService
from market_data_service.domain.candle_gap_detector import MissingInterval
from market_data_service.domain.enums import MarketDataSource
from market_data_service.domain.events.market_data_backfill_requested import MarketDataBackfillRequested


class FakeBackfillRequester:
    def __init__(self, outcomes: tuple[bool, ...]) -> None:
        self.outcomes = list(outcomes)
        self.events: list[MarketDataBackfillRequested] = []

    async def request_backfill(self, event: MarketDataBackfillRequested) -> bool:
        self.events.append(event)
        return self.outcomes.pop(0)


class BackfillPlanningServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_backfill_for_gaps_creates_one_request_per_missing_interval(self) -> None:
        requester = FakeBackfillRequester(outcomes=(True, False))
        service = BackfillPlanningService(requester)

        result = await service.request_backfill_for_gaps(
            source=MarketDataSource.BINANCE_SPOT,
            canonical_symbol="ETH/USDT",
            provider_symbol="ETHUSDT",
            timeframe="1h",
            parent_batch_id="batch-1",
            missing_intervals=(
                MissingInterval(
                    open_time=datetime(2026, 7, 14, 9, tzinfo=UTC),
                    close_time=datetime(2026, 7, 14, 10, tzinfo=UTC),
                ),
                MissingInterval(
                    open_time=datetime(2026, 7, 14, 11, tzinfo=UTC),
                    close_time=datetime(2026, 7, 14, 12, tzinfo=UTC),
                ),
            ),
        )

        self.assertEqual(result.requested_count, 1)
        self.assertEqual(result.skipped_duplicate_count, 1)
        self.assertEqual(len(requester.events), 2)
        self.assertEqual(requester.events[0].requested_from, datetime(2026, 7, 14, 9, tzinfo=UTC))
