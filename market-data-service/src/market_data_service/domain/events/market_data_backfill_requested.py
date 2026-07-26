from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from market_data_service.domain.enums import MarketDataSource
from market_data_service.domain.scheduler_models import BACKFILL_SYNC_PRIORITY

MARKET_DATA_BACKFILL_REQUESTED_EVENT_TYPE = "MarketDataBackfillRequested"


@dataclass(frozen=True, slots=True)
class MarketDataBackfillRequested:
    event_id: str
    event_type: str
    occurred_at: datetime
    source: MarketDataSource
    canonical_symbol: str
    provider_symbol: str
    timeframe: str
    requested_from: datetime
    requested_to: datetime
    parent_batch_id: str
    reason: str
    priority: int
    idempotency_key: str

    @classmethod
    def from_gap(
        cls,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        provider_symbol: str,
        timeframe: str,
        requested_from: datetime,
        requested_to: datetime,
        parent_batch_id: str,
        occurred_at: datetime | None = None,
    ) -> "MarketDataBackfillRequested":
        normalized_provider_symbol = provider_symbol.strip().upper()
        normalized_timeframe = timeframe.strip().lower()
        event_time = occurred_at or datetime.now(UTC)
        return cls(
            event_id=str(uuid4()),
            event_type=MARKET_DATA_BACKFILL_REQUESTED_EVENT_TYPE,
            occurred_at=event_time,
            source=source,
            canonical_symbol=canonical_symbol,
            provider_symbol=normalized_provider_symbol,
            timeframe=normalized_timeframe,
            requested_from=requested_from,
            requested_to=requested_to,
            parent_batch_id=parent_batch_id,
            reason="GAP_DETECTED",
            priority=BACKFILL_SYNC_PRIORITY,
            idempotency_key=build_market_data_backfill_requested_idempotency_key(
                source=source,
                provider_symbol=normalized_provider_symbol,
                timeframe=normalized_timeframe,
                requested_from=requested_from,
                requested_to=requested_to,
            ),
        )

    def payload_json(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "occurred_at": self.occurred_at.isoformat(),
            "source": self.source.value,
            "canonical_symbol": self.canonical_symbol,
            "provider_symbol": self.provider_symbol,
            "timeframe": self.timeframe,
            "requested_from": self.requested_from.isoformat(),
            "requested_to": self.requested_to.isoformat(),
            "parent_batch_id": self.parent_batch_id,
            "reason": self.reason,
            "priority": self.priority,
            "idempotency_key": self.idempotency_key,
        }


def build_market_data_backfill_requested_idempotency_key(
    *,
    source: MarketDataSource,
    provider_symbol: str,
    timeframe: str,
    requested_from: datetime,
    requested_to: datetime,
) -> str:
    return "|".join(
        (
            source.value,
            provider_symbol.strip().upper(),
            timeframe.strip().lower(),
            requested_from.astimezone(UTC).isoformat(),
            requested_to.astimezone(UTC).isoformat(),
        )
    )
