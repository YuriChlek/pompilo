from __future__ import annotations

from datetime import UTC, datetime, timedelta
import unittest

from market_data_service.domain.enums import MarketDataSource
from market_data_service.domain.scheduler import build_sync_jobs


class LoadShapeTests(unittest.TestCase):
    def test_scheduler_creates_unique_jobs_for_large_symbol_timeframe_universe(self) -> None:
        provider_symbols = tuple(f"ASSET{index}USDT" for index in range(250))

        jobs = build_sync_jobs(
            source=MarketDataSource.BINANCE_SPOT,
            provider_symbols=provider_symbols,
            timeframes=("1h", "4h", "1d"),
            now=datetime(2026, 7, 14, 12, 2, tzinfo=UTC),
            safety_delay_by_timeframe={
                "1h": timedelta(seconds=30),
                "4h": timedelta(seconds=45),
                "1d": timedelta(seconds=90),
            },
            jitter_seconds=60,
            random_seed=42,
        )

        self.assertEqual(len(jobs), 750)
        self.assertEqual(len({job.idempotency_key for job in jobs}), 750)
        self.assertEqual({job.timeframe for job in jobs}, {"1h", "4h", "1d"})
        self.assertTrue(all(job.scheduled_for >= job.expected_close_time for job in jobs))
