from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class Stage16LoadRateLimitHardeningSmokeTests(unittest.TestCase):
    def test_load_and_rate_limit_hardening_artifacts_exist(self) -> None:
        limiter = (SERVICE_ROOT / "src/market_data_service/infrastructure/concurrency_limiter.py").read_text(
            encoding="utf-8"
        )
        breaker = (
            SERVICE_ROOT / "src/market_data_service/infrastructure/providers/circuit_breaker.py"
        ).read_text(encoding="utf-8")
        document = (SERVICE_ROOT / "docs/load_rate_limit_hardening.md").read_text(encoding="utf-8")

        self.assertIn("AsyncConcurrencyLimiter", limiter)
        self.assertIn("CircuitBreaker", breaker)
        self.assertIn("BINANCE_MAX_CONCURRENT_REQUESTS", document)
        self.assertIn("market_data_provider_rate_limited_total", document)
        self.assertIn("market_data_queue_lag_seconds", document)
        self.assertIn("market_data_outbox_lag_seconds", document)

    def test_stage_16_has_automated_load_rate_limit_and_lag_coverage(self) -> None:
        load_test = (SERVICE_ROOT / "tests/unit/test_load_shape.py").read_text(encoding="utf-8")
        adapter_test = (SERVICE_ROOT / "tests/unit/test_binance_spot_adapter.py").read_text(encoding="utf-8")
        outbox_test = (SERVICE_ROOT / "tests/unit/test_outbox_publisher_service.py").read_text(encoding="utf-8")

        self.assertIn("test_scheduler_creates_unique_jobs_for_large_symbol_timeframe_universe", load_test)
        self.assertIn("test_fetch_closed_candles_respects_concurrency_limiter", adapter_test)
        self.assertIn("test_fetch_closed_candles_opens_circuit_breaker_after_rate_limit_failures", adapter_test)
        self.assertIn("MARKET_DATA_PROVIDER_RATE_LIMITED_TOTAL", adapter_test)
        self.assertIn("test_publish_once_reports_publisher_lag_metric", outbox_test)
