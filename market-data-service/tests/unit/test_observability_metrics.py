from __future__ import annotations

from datetime import UTC, datetime
import unittest

from market_data_service.domain.enums import CandleRangeStatus, MarketDataSource
from market_data_service.domain.snapshot_models import MarketSnapshot
from market_data_service.observability.metrics import (
    MARKET_DATA_GAP_COUNT,
    MARKET_DATA_OUTBOX_LAG_SECONDS,
    MARKET_DATA_PROVIDER_ERRORS_TOTAL,
    MARKET_DATA_PROVIDER_RATE_LIMITED_TOTAL,
    MARKET_DATA_QUEUE_LAG_SECONDS,
    MARKET_DATA_SNAPSHOT_AGE_SECONDS,
    MARKET_DATA_SYNC_DURATION_SECONDS,
    STABLE_METRIC_NAMES,
    InMemoryMetricsRecorder,
    record_outbox_lag,
    record_provider_error,
    record_queue_lag,
    record_snapshot_age,
    record_sync_metrics,
)


class ObservabilityMetricsTests(unittest.TestCase):
    def test_stable_metric_names_match_plan_contract(self) -> None:
        self.assertIn(MARKET_DATA_SYNC_DURATION_SECONDS, STABLE_METRIC_NAMES)
        self.assertIn(MARKET_DATA_GAP_COUNT, STABLE_METRIC_NAMES)
        self.assertIn(MARKET_DATA_SNAPSHOT_AGE_SECONDS, STABLE_METRIC_NAMES)
        self.assertIn(MARKET_DATA_PROVIDER_ERRORS_TOTAL, STABLE_METRIC_NAMES)
        self.assertIn(MARKET_DATA_PROVIDER_RATE_LIMITED_TOTAL, STABLE_METRIC_NAMES)
        self.assertIn(MARKET_DATA_QUEUE_LAG_SECONDS, STABLE_METRIC_NAMES)
        self.assertIn(MARKET_DATA_OUTBOX_LAG_SECONDS, STABLE_METRIC_NAMES)

    def test_record_sync_metrics_emits_rows_gap_status_and_duration(self) -> None:
        recorder = InMemoryMetricsRecorder()

        record_sync_metrics(
            recorder,
            source=MarketDataSource.BINANCE_SPOT,
            canonical_symbol="ETH/USDT",
            timeframe="1h",
            status="INCOMPLETE",
            duration_seconds=1.25,
            rows_fetched=10,
            rows_inserted=8,
            rows_skipped_duplicate=2,
            gap_count=1,
        )

        names = {sample.name for sample in recorder.samples}
        self.assertIn(MARKET_DATA_SYNC_DURATION_SECONDS, names)
        self.assertIn(MARKET_DATA_GAP_COUNT, names)
        self.assertEqual(recorder.samples[0].labels["canonical_symbol"], "ETH/USDT")

    def test_record_provider_error_emits_rate_limit_metric_when_applicable(self) -> None:
        recorder = InMemoryMetricsRecorder()

        record_provider_error(
            recorder,
            source=MarketDataSource.BINANCE_SPOT,
            provider="BINANCE_SPOT",
            error_code="429",
            rate_limited=True,
        )

        self.assertEqual(
            [sample.name for sample in recorder.samples],
            [MARKET_DATA_PROVIDER_ERRORS_TOTAL, MARKET_DATA_PROVIDER_RATE_LIMITED_TOTAL],
        )

    def test_record_snapshot_age_and_outbox_lag_emit_gauges(self) -> None:
        recorder = InMemoryMetricsRecorder()

        record_snapshot_age(
            recorder,
            snapshot=_snapshot(),
            now=datetime(2026, 7, 14, 10, tzinfo=UTC),
        )
        record_outbox_lag(recorder, lag_seconds=42.0)

        self.assertEqual(recorder.samples[0].name, MARKET_DATA_SNAPSHOT_AGE_SECONDS)
        self.assertEqual(recorder.samples[0].value, 3600.0)
        self.assertEqual(recorder.samples[1].name, MARKET_DATA_OUTBOX_LAG_SECONDS)
        self.assertEqual(recorder.samples[1].value, 42.0)

    def test_record_queue_lag_emits_queue_label(self) -> None:
        recorder = InMemoryMetricsRecorder()

        record_queue_lag(recorder, queue_name="market-data-events", lag_seconds=15.0)

        self.assertEqual(recorder.samples[0].name, MARKET_DATA_QUEUE_LAG_SECONDS)
        self.assertEqual(recorder.samples[0].labels["queue_name"], "market-data-events")
        self.assertEqual(recorder.samples[0].value, 15.0)


def _snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        id="snapshot-1",
        source=MarketDataSource.BINANCE_SPOT,
        canonical_symbol="ETH/USDT",
        timeframe="1h",
        last_closed_candle_time=datetime(2026, 7, 14, 9, tzinfo=UTC),
        lookback_start_time=datetime(2026, 7, 14, 8, tzinfo=UTC),
        lookback_end_time=datetime(2026, 7, 14, 9, tzinfo=UTC),
        candle_count=1,
        data_hash="data-hash",
        batch_id="batch-1",
        completeness_status=CandleRangeStatus.COMPLETE,
        snapshot_version=1,
        created_at=datetime(2026, 7, 14, 9, tzinfo=UTC),
    )
