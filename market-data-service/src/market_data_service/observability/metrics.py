from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from market_data_service.domain.enums import MarketDataSource
from market_data_service.domain.snapshot_models import MarketSnapshot

MARKET_DATA_SYNC_DURATION_SECONDS = "market_data_sync_duration_seconds"
MARKET_DATA_SYNC_ROWS_FETCHED_TOTAL = "market_data_sync_rows_fetched_total"
MARKET_DATA_SYNC_ROWS_INSERTED_TOTAL = "market_data_sync_rows_inserted_total"
MARKET_DATA_SYNC_ROWS_SKIPPED_DUPLICATE_TOTAL = "market_data_sync_rows_skipped_duplicate_total"
MARKET_DATA_GAP_COUNT = "market_data_gap_count"
MARKET_DATA_SNAPSHOT_AGE_SECONDS = "market_data_snapshot_age_seconds"
MARKET_DATA_BATCH_STATUS_TOTAL = "market_data_batch_status_total"
MARKET_DATA_PROVIDER_ERRORS_TOTAL = "market_data_provider_errors_total"
MARKET_DATA_PROVIDER_RATE_LIMITED_TOTAL = "market_data_provider_rate_limited_total"
MARKET_DATA_QUEUE_LAG_SECONDS = "market_data_queue_lag_seconds"
MARKET_DATA_OUTBOX_LAG_SECONDS = "market_data_outbox_lag_seconds"
MARKET_DATA_SYNC_JOBS_STUCK_TOTAL = "market_data_sync_jobs_stuck_total"
MARKET_DATA_SCHEDULER_TICKS_TOTAL = "market_data_scheduler_ticks_total"
MARKET_DATA_OUTBOX_EVENTS_FETCHED_TOTAL = "market_data_outbox_events_fetched_total"
MARKET_DATA_OUTBOX_EVENTS_PUBLISHED_TOTAL = "market_data_outbox_events_published_total"
MARKET_DATA_OUTBOX_EVENTS_FAILED_TOTAL = "market_data_outbox_events_failed_total"

STABLE_METRIC_NAMES = (
    MARKET_DATA_SYNC_DURATION_SECONDS,
    MARKET_DATA_SYNC_ROWS_FETCHED_TOTAL,
    MARKET_DATA_SYNC_ROWS_INSERTED_TOTAL,
    MARKET_DATA_SYNC_ROWS_SKIPPED_DUPLICATE_TOTAL,
    MARKET_DATA_GAP_COUNT,
    MARKET_DATA_SNAPSHOT_AGE_SECONDS,
    MARKET_DATA_BATCH_STATUS_TOTAL,
    MARKET_DATA_PROVIDER_ERRORS_TOTAL,
    MARKET_DATA_PROVIDER_RATE_LIMITED_TOTAL,
    MARKET_DATA_QUEUE_LAG_SECONDS,
    MARKET_DATA_OUTBOX_LAG_SECONDS,
    MARKET_DATA_SYNC_JOBS_STUCK_TOTAL,
    MARKET_DATA_SCHEDULER_TICKS_TOTAL,
    MARKET_DATA_OUTBOX_EVENTS_FETCHED_TOTAL,
    MARKET_DATA_OUTBOX_EVENTS_PUBLISHED_TOTAL,
    MARKET_DATA_OUTBOX_EVENTS_FAILED_TOTAL,
)


@dataclass(frozen=True, slots=True)
class MetricSample:
    name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    kind: str = "gauge"


class MetricsRecorder(Protocol):
    def record(self, sample: MetricSample) -> None: ...


class InMemoryMetricsRecorder:
    def __init__(self) -> None:
        self.samples: list[MetricSample] = []

    def record(self, sample: MetricSample) -> None:
        self.samples.append(sample)


def record_sync_metrics(
    recorder: MetricsRecorder | None,
    *,
    source: MarketDataSource,
    canonical_symbol: str,
    timeframe: str,
    status: str,
    duration_seconds: float,
    rows_fetched: int,
    rows_inserted: int,
    rows_skipped_duplicate: int,
    gap_count: int,
) -> None:
    if recorder is None:
        return
    labels = {
        "source": source.value,
        "canonical_symbol": canonical_symbol,
        "timeframe": timeframe,
        "status": status,
    }
    recorder.record(MetricSample(MARKET_DATA_SYNC_DURATION_SECONDS, duration_seconds, labels, "histogram"))
    recorder.record(MetricSample(MARKET_DATA_SYNC_ROWS_FETCHED_TOTAL, float(rows_fetched), labels, "counter"))
    recorder.record(MetricSample(MARKET_DATA_SYNC_ROWS_INSERTED_TOTAL, float(rows_inserted), labels, "counter"))
    recorder.record(
        MetricSample(MARKET_DATA_SYNC_ROWS_SKIPPED_DUPLICATE_TOTAL, float(rows_skipped_duplicate), labels, "counter")
    )
    recorder.record(MetricSample(MARKET_DATA_GAP_COUNT, float(gap_count), labels, "gauge"))
    recorder.record(MetricSample(MARKET_DATA_BATCH_STATUS_TOTAL, 1.0, labels, "counter"))


def record_provider_error(
    recorder: MetricsRecorder | None,
    *,
    source: MarketDataSource,
    provider: str,
    error_code: str,
    rate_limited: bool = False,
) -> None:
    if recorder is None:
        return
    labels = {"source": source.value, "provider": provider, "error_code": error_code}
    recorder.record(MetricSample(MARKET_DATA_PROVIDER_ERRORS_TOTAL, 1.0, labels, "counter"))
    if rate_limited:
        recorder.record(MetricSample(MARKET_DATA_PROVIDER_RATE_LIMITED_TOTAL, 1.0, labels, "counter"))


def record_outbox_lag(recorder: MetricsRecorder | None, *, lag_seconds: float) -> None:
    if recorder is None:
        return
    recorder.record(MetricSample(MARKET_DATA_OUTBOX_LAG_SECONDS, lag_seconds, {}, "gauge"))


def record_queue_lag(recorder: MetricsRecorder | None, *, queue_name: str, lag_seconds: float) -> None:
    if recorder is None:
        return
    recorder.record(MetricSample(MARKET_DATA_QUEUE_LAG_SECONDS, lag_seconds, {"queue_name": queue_name}, "gauge"))


def record_snapshot_age(recorder: MetricsRecorder | None, *, snapshot: MarketSnapshot, now: datetime) -> None:
    if recorder is None:
        return
    labels = {
        "source": snapshot.source.value,
        "canonical_symbol": snapshot.canonical_symbol,
        "timeframe": snapshot.timeframe,
        "snapshot_id": snapshot.id,
    }
    age_seconds = max(0.0, (now - snapshot.last_closed_candle_time.astimezone(now.tzinfo)).total_seconds())
    recorder.record(MetricSample(MARKET_DATA_SNAPSHOT_AGE_SECONDS, age_seconds, labels, "gauge"))


def record_sync_jobs_stuck(recorder: MetricsRecorder | None, *, stuck_count: int) -> None:
    if recorder is None:
        return
    recorder.record(MetricSample(MARKET_DATA_SYNC_JOBS_STUCK_TOTAL, float(stuck_count), {}, "gauge"))


def record_scheduler_tick(recorder: MetricsRecorder | None, *, status: str) -> None:
    if recorder is None:
        return
    recorder.record(MetricSample(MARKET_DATA_SCHEDULER_TICKS_TOTAL, 1.0, {"status": status}, "counter"))


def record_outbox_publish_batch(
    recorder: MetricsRecorder | None,
    *,
    fetched_count: int,
    published_count: int,
    failed_count: int,
    status: str,
) -> None:
    if recorder is None:
        return
    labels = {"status": status}
    recorder.record(MetricSample(MARKET_DATA_OUTBOX_EVENTS_FETCHED_TOTAL, float(fetched_count), labels, "counter"))
    recorder.record(MetricSample(MARKET_DATA_OUTBOX_EVENTS_PUBLISHED_TOTAL, float(published_count), labels, "counter"))
    recorder.record(MetricSample(MARKET_DATA_OUTBOX_EVENTS_FAILED_TOTAL, float(failed_count), labels, "counter"))
