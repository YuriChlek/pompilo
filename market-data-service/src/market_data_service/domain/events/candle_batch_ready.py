from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from market_data_service.domain.enums import CandleRangeStatus, MarketDataSource
from market_data_service.domain.snapshot_models import MarketSnapshot

CANDLE_BATCH_READY_EVENT_TYPE = "CandleBatchReady"


@dataclass(frozen=True, slots=True)
class CandleBatchReady:
    event_id: str
    event_type: str
    occurred_at: datetime
    source: MarketDataSource
    canonical_symbol: str
    timeframe: str
    batch_id: str
    snapshot_id: str
    last_closed_candle_time: datetime
    completeness_status: CandleRangeStatus
    idempotency_key: str
    snapshot_version: int

    @classmethod
    def from_snapshot(
        cls,
        snapshot: MarketSnapshot,
        *,
        occurred_at: datetime | None = None,
    ) -> "CandleBatchReady":
        event_time = occurred_at or datetime.now(UTC)
        return cls(
            event_id=str(uuid4()),
            event_type=CANDLE_BATCH_READY_EVENT_TYPE,
            occurred_at=event_time,
            source=snapshot.source,
            canonical_symbol=snapshot.canonical_symbol,
            timeframe=snapshot.timeframe,
            batch_id=snapshot.batch_id,
            snapshot_id=snapshot.id,
            last_closed_candle_time=snapshot.last_closed_candle_time,
            completeness_status=snapshot.completeness_status,
            idempotency_key=build_candle_batch_ready_idempotency_key(snapshot),
            snapshot_version=snapshot.snapshot_version,
        )

    def payload_json(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "occurred_at": self.occurred_at.isoformat(),
            "source": self.source.value,
            "canonical_symbol": self.canonical_symbol,
            "timeframe": self.timeframe,
            "batch_id": self.batch_id,
            "snapshot_id": self.snapshot_id,
            "last_closed_candle_time": self.last_closed_candle_time.isoformat(),
            "completeness_status": self.completeness_status.value,
            "idempotency_key": self.idempotency_key,
            "snapshot_version": self.snapshot_version,
        }


def build_candle_batch_ready_idempotency_key(snapshot: MarketSnapshot) -> str:
    parts = (
        snapshot.source.value,
        snapshot.canonical_symbol,
        snapshot.timeframe,
        snapshot.last_closed_candle_time.isoformat(),
        str(snapshot.snapshot_version),
    )
    return "|".join(parts)
