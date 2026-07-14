from __future__ import annotations

from dataclasses import dataclass

from market_data_service.application.backfill_ports import BackfillRequestPort
from market_data_service.domain.candle_gap_detector import MissingInterval
from market_data_service.domain.enums import MarketDataSource
from market_data_service.domain.events.market_data_backfill_requested import MarketDataBackfillRequested


@dataclass(frozen=True, slots=True)
class BackfillPlanningResult:
    requested_count: int
    skipped_duplicate_count: int


class BackfillPlanningService:
    def __init__(self, backfill_requester: BackfillRequestPort) -> None:
        self.backfill_requester = backfill_requester

    async def request_backfill_for_gaps(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        provider_symbol: str,
        timeframe: str,
        parent_batch_id: str,
        missing_intervals: tuple[MissingInterval, ...],
    ) -> BackfillPlanningResult:
        requested_count = 0
        skipped_duplicate_count = 0
        for interval in missing_intervals:
            event = MarketDataBackfillRequested.from_gap(
                source=source,
                canonical_symbol=canonical_symbol,
                provider_symbol=provider_symbol,
                timeframe=timeframe,
                requested_from=interval.open_time,
                requested_to=interval.close_time,
                parent_batch_id=parent_batch_id,
            )
            if await self.backfill_requester.request_backfill(event):
                requested_count += 1
            else:
                skipped_duplicate_count += 1
        return BackfillPlanningResult(
            requested_count=requested_count,
            skipped_duplicate_count=skipped_duplicate_count,
        )
