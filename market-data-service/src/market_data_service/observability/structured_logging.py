from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Protocol

from market_data_service.application.sync_models import SyncClosedCandlesResult

REQUIRED_MARKET_DATA_LOG_FIELDS = (
    "source",
    "canonical_symbol",
    "timeframe",
    "batch_id",
    "snapshot_id",
    "last_closed_candle_time",
    "status",
    "gap_count",
    "correlation_id",
)


@dataclass(frozen=True, slots=True)
class StructuredLogEvent:
    event_name: str
    level: str
    message: str
    fields: Mapping[str, object | None] = field(default_factory=dict)


class StructuredLogger(Protocol):
    def emit(self, event: StructuredLogEvent) -> None: ...


class InMemoryStructuredLogger:
    def __init__(self) -> None:
        self.events: list[StructuredLogEvent] = []

    def emit(self, event: StructuredLogEvent) -> None:
        self.events.append(event)


def build_sync_completed_log(
    result: SyncClosedCandlesResult,
    *,
    last_closed_candle_time: datetime | None,
    correlation_id: str | None,
) -> StructuredLogEvent:
    return StructuredLogEvent(
        event_name="market_data.sync.completed",
        level="INFO",
        message="market data sync completed",
        fields={
            "source": result.source.value,
            "canonical_symbol": result.canonical_symbol,
            "provider_symbol": result.provider_symbol,
            "timeframe": result.timeframe,
            "batch_id": result.batch_id,
            "snapshot_id": result.snapshot_id,
            "last_closed_candle_time": last_closed_candle_time.isoformat() if last_closed_candle_time else None,
            "status": result.batch_status.value,
            "range_status": result.range_status.value,
            "gap_count": result.gap_count,
            "correlation_id": correlation_id,
            "rows_fetched": result.fetched_count,
            "rows_inserted": result.inserted_count,
            "rows_skipped_duplicate": result.skipped_duplicate_count,
        },
    )


def build_sync_failed_log(
    *,
    source: str,
    provider_symbol: str,
    timeframe: str,
    batch_id: str | None,
    error_code: str,
    correlation_id: str | None,
) -> StructuredLogEvent:
    return StructuredLogEvent(
        event_name="market_data.sync.failed",
        level="ERROR",
        message="market data sync failed",
        fields={
            "source": source,
            "canonical_symbol": None,
            "provider_symbol": provider_symbol,
            "timeframe": timeframe,
            "batch_id": batch_id,
            "snapshot_id": None,
            "last_closed_candle_time": None,
            "status": "FAILED",
            "gap_count": None,
            "correlation_id": correlation_id,
            "error_code": error_code,
        },
    )


def build_outbox_publish_log(
    *,
    fetched_count: int,
    published_count: int,
    retry_count: int,
    failed_count: int,
    publisher_lag_seconds: float,
    correlation_id: str | None,
) -> StructuredLogEvent:
    status = "COMPLETE" if failed_count == 0 else "DEGRADED"
    return StructuredLogEvent(
        event_name="market_data.outbox.publish_once",
        level="INFO" if failed_count == 0 else "WARNING",
        message="market data outbox publish pass completed",
        fields={
            "source": None,
            "canonical_symbol": None,
            "timeframe": None,
            "batch_id": None,
            "snapshot_id": None,
            "last_closed_candle_time": None,
            "status": status,
            "gap_count": None,
            "correlation_id": correlation_id,
            "fetched_count": fetched_count,
            "published_count": published_count,
            "retry_count": retry_count,
            "failed_count": failed_count,
            "publisher_lag_seconds": publisher_lag_seconds,
        },
    )


def has_required_market_data_fields(event: StructuredLogEvent) -> bool:
    return all(field_name in event.fields for field_name in REQUIRED_MARKET_DATA_LOG_FIELDS)
