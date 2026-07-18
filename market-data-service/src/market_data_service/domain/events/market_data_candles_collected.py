from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from market_data_service.domain.enums import MarketDataSource
from market_data_service.domain.snapshot_models import MarketSnapshot

MARKET_DATA_CANDLES_COLLECTED_EVENT_TYPE = "market_data.candles_collected"
MARKET_DATA_CANDLES_COLLECTED_CONTRACT_VERSION = "market-data-event.v1"


@dataclass(frozen=True, slots=True)
class MarketDataCandlesCollectedEvent:
    event_id: str
    event_type: str
    contract_version: str
    occurred_at: datetime
    source: MarketDataSource
    symbol: str
    provider_symbol: str
    timeframe: str
    from_time: datetime
    to_time: datetime
    batch_id: str
    snapshot_id: str
    closed_at: datetime
    idempotency_key: str

    @classmethod
    def from_snapshot(
        cls,
        snapshot: MarketSnapshot,
        provider_symbol: str,
        *,
        occurred_at: datetime | None = None,
    ) -> "MarketDataCandlesCollectedEvent":
        event_time = occurred_at or datetime.now(UTC)
        idempotency_key = build_market_data_candles_collected_idempotency_key(
            source=snapshot.source,
            provider_symbol=provider_symbol,
            timeframe=snapshot.timeframe,
            closed_at=snapshot.last_closed_candle_time,
        )
        return cls(
            event_id=str(uuid4()),
            event_type=MARKET_DATA_CANDLES_COLLECTED_EVENT_TYPE,
            contract_version=MARKET_DATA_CANDLES_COLLECTED_CONTRACT_VERSION,
            occurred_at=event_time,
            source=snapshot.source,
            symbol=snapshot.canonical_symbol,
            provider_symbol=provider_symbol,
            timeframe=snapshot.timeframe,
            from_time=snapshot.lookback_start_time,
            to_time=snapshot.lookback_end_time,
            batch_id=snapshot.batch_id,
            snapshot_id=snapshot.id,
            closed_at=snapshot.last_closed_candle_time,
            idempotency_key=idempotency_key,
        )

    def payload_json(self) -> dict[str, object]:
        from_time_utc = self.from_time.astimezone(UTC) if self.from_time.tzinfo else self.from_time.replace(tzinfo=UTC)
        to_time_utc = self.to_time.astimezone(UTC) if self.to_time.tzinfo else self.to_time.replace(tzinfo=UTC)
        closed_at_utc = self.closed_at.astimezone(UTC) if self.closed_at.tzinfo else self.closed_at.replace(tzinfo=UTC)

        from_str = from_time_utc.isoformat().replace("+00:00", "Z")
        to_str = to_time_utc.isoformat().replace("+00:00", "Z")
        closed_str = closed_at_utc.isoformat().replace("+00:00", "Z")

        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "contract_version": self.contract_version,
            "occurred_at": self.occurred_at.astimezone(UTC).isoformat().replace("+00:00", "Z") if self.occurred_at.tzinfo else self.occurred_at.replace(tzinfo=UTC).isoformat().replace("+00:00", "Z"),
            "source": self.source.value,
            "symbol": self.symbol,
            "provider_symbol": self.provider_symbol,
            "timeframe": self.timeframe,
            "from": from_str,
            "to": to_str,
            "batch_id": self.batch_id,
            "snapshot_id": self.snapshot_id,
            "closed_at": closed_str,
            "idempotency_key": self.idempotency_key,
        }


def build_market_data_candles_collected_idempotency_key(
    *,
    source: MarketDataSource,
    provider_symbol: str,
    timeframe: str,
    closed_at: datetime,
) -> str:
    closed_at_utc = closed_at.astimezone(UTC) if closed_at.tzinfo else closed_at.replace(tzinfo=UTC)
    closed_at_str = closed_at_utc.isoformat().replace("+00:00", "Z")
    return f"{source.value}:{provider_symbol.strip().upper()}:{timeframe.strip().lower()}:{closed_at_str}"
