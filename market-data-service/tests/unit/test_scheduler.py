from __future__ import annotations

from datetime import UTC, datetime, timedelta
import unittest

from market_data_service.domain.enums import MarketDataSource
from market_data_service.domain.scheduler import (
    build_sync_job_idempotency_key,
    build_sync_jobs,
    due_closed_candle_time,
    latest_closed_candle_time,
)


class SchedulerDomainTests(unittest.TestCase):
    def test_latest_closed_candle_time_uses_utc_boundaries(self) -> None:
        now = datetime(2026, 7, 14, 10, 31, 45, tzinfo=UTC)

        self.assertEqual(latest_closed_candle_time(now, "1h"), datetime(2026, 7, 14, 10, tzinfo=UTC))
        self.assertEqual(latest_closed_candle_time(now, "4h"), datetime(2026, 7, 14, 8, tzinfo=UTC))
        self.assertEqual(latest_closed_candle_time(now, "1d"), datetime(2026, 7, 14, tzinfo=UTC))

    def test_due_closed_candle_time_respects_safety_delay(self) -> None:
        close_time = due_closed_candle_time(
            now=datetime(2026, 7, 14, 10, 0, 10, tzinfo=UTC),
            timeframe="1h",
            safety_delay=timedelta(seconds=30),
        )

        self.assertEqual(close_time, datetime(2026, 7, 14, 9, tzinfo=UTC))

    def test_due_closed_candle_time_returns_latest_close_after_delay(self) -> None:
        close_time = due_closed_candle_time(
            now=datetime(2026, 7, 14, 10, 0, 31, tzinfo=UTC),
            timeframe="1h",
            safety_delay=timedelta(seconds=30),
        )

        self.assertEqual(close_time, datetime(2026, 7, 14, 10, tzinfo=UTC))

    def test_build_sync_jobs_creates_per_symbol_jobs_with_jitter(self) -> None:
        jobs = build_sync_jobs(
            source=MarketDataSource.BINANCE_SPOT,
            provider_symbols=("ethusdt", "btcusdt"),
            timeframes=("1h", "4h"),
            now=datetime(2026, 7, 14, 12, 1, tzinfo=UTC),
            safety_delay_by_timeframe={"1h": timedelta(seconds=30), "4h": timedelta(seconds=45)},
            jitter_seconds=10,
            random_seed=1,
        )

        self.assertEqual(len(jobs), 4)
        self.assertEqual({job.provider_symbol for job in jobs}, {"ETHUSDT", "BTCUSDT"})
        self.assertEqual({job.timeframe for job in jobs}, {"1h", "4h"})
        self.assertTrue(all(job.scheduled_for >= job.expected_close_time for job in jobs))
        self.assertEqual(len({job.idempotency_key for job in jobs}), 4)

    def test_idempotency_key_is_stable_and_normalized(self) -> None:
        key = build_sync_job_idempotency_key(
            source=MarketDataSource.BINANCE_SPOT,
            provider_symbol=" ethusdt ",
            timeframe=" 1H ",
            expected_close_time=datetime(2026, 7, 14, 10, tzinfo=UTC),
        )

        self.assertEqual(key, "BINANCE_SPOT|ETHUSDT|1h|2026-07-14T10:00:00+00:00")
