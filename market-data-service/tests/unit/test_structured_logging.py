from __future__ import annotations

from datetime import UTC, datetime
import unittest

from market_data_service.application.sync_models import SyncClosedCandlesResult
from market_data_service.domain.enums import CandleRangeStatus, MarketDataBatchStatus, MarketDataSource
from market_data_service.observability.structured_logging import (
    InMemoryStructuredLogger,
    build_outbox_publish_log,
    build_sync_completed_log,
    has_required_market_data_fields,
)


class StructuredLoggingTests(unittest.TestCase):
    def test_sync_completed_log_contains_required_market_data_fields(self) -> None:
        event = build_sync_completed_log(
            _sync_result(),
            last_closed_candle_time=datetime(2026, 7, 14, 9, tzinfo=UTC),
            correlation_id="corr-1",
        )

        self.assertTrue(has_required_market_data_fields(event))
        self.assertEqual(event.fields["batch_id"], "batch-1")
        self.assertEqual(event.fields["snapshot_id"], "snapshot-1")
        self.assertEqual(event.fields["correlation_id"], "corr-1")

    def test_outbox_publish_log_uses_degraded_status_when_failures_exist(self) -> None:
        event = build_outbox_publish_log(
            fetched_count=2,
            published_count=1,
            retry_count=0,
            failed_count=1,
            publisher_lag_seconds=300.0,
            correlation_id="corr-2",
        )

        self.assertTrue(has_required_market_data_fields(event))
        self.assertEqual(event.level, "WARNING")
        self.assertEqual(event.fields["status"], "DEGRADED")

    def test_in_memory_structured_logger_records_events(self) -> None:
        logger = InMemoryStructuredLogger()
        event = build_sync_completed_log(_sync_result(), last_closed_candle_time=None, correlation_id=None)

        logger.emit(event)

        self.assertEqual(logger.events, [event])


def _sync_result() -> SyncClosedCandlesResult:
    return SyncClosedCandlesResult(
        batch_id="batch-1",
        source=MarketDataSource.BINANCE_SPOT,
        canonical_symbol="ETH/USDT",
        provider_symbol="ETHUSDT",
        timeframe="1h",
        fetched_count=1,
        inserted_count=1,
        skipped_duplicate_count=0,
        range_status=CandleRangeStatus.COMPLETE,
        batch_status=MarketDataBatchStatus.COMPLETE,
        gap_count=0,
        dry_run=False,
        snapshot_id="snapshot-1",
        snapshot_created=True,
    )
