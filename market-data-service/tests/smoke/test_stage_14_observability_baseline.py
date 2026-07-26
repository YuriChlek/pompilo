from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class Stage14ObservabilityBaselineSmokeTests(unittest.TestCase):
    def test_metrics_contract_contains_plan_metrics(self) -> None:
        metrics = (SERVICE_ROOT / "src/market_data_service/observability/metrics.py").read_text(encoding="utf-8")

        self.assertIn("market_data_sync_duration_seconds", metrics)
        self.assertIn("market_data_sync_rows_fetched_total", metrics)
        self.assertIn("market_data_sync_rows_inserted_total", metrics)
        self.assertIn("market_data_sync_rows_skipped_duplicate_total", metrics)
        self.assertIn("market_data_gap_count", metrics)
        self.assertIn("market_data_snapshot_age_seconds", metrics)
        self.assertIn("market_data_batch_status_total", metrics)
        self.assertIn("market_data_provider_errors_total", metrics)
        self.assertIn("market_data_provider_rate_limited_total", metrics)
        self.assertIn("market_data_queue_lag_seconds", metrics)
        self.assertIn("market_data_outbox_lag_seconds", metrics)

    def test_alerts_cover_stale_gaps_provider_outbox_and_stuck_jobs(self) -> None:
        alerts = (SERVICE_ROOT / "src/market_data_service/observability/alerts.py").read_text(encoding="utf-8")

        self.assertIn("market_data_snapshot_stale", alerts)
        self.assertIn("market_data_gap_detected", alerts)
        self.assertIn("market_data_provider_errors_high", alerts)
        self.assertIn("market_data_outbox_lag_high", alerts)
        self.assertIn("market_data_sync_jobs_stuck", alerts)

    def test_structured_logs_have_required_identifiers(self) -> None:
        structured_logging = (
            SERVICE_ROOT / "src/market_data_service/observability/structured_logging.py"
        ).read_text(encoding="utf-8")

        self.assertIn("source", structured_logging)
        self.assertIn("canonical_symbol", structured_logging)
        self.assertIn("timeframe", structured_logging)
        self.assertIn("batch_id", structured_logging)
        self.assertIn("snapshot_id", structured_logging)
        self.assertIn("last_closed_candle_time", structured_logging)
        self.assertIn("status", structured_logging)
        self.assertIn("gap_count", structured_logging)
        self.assertIn("correlation_id", structured_logging)

    def test_dashboard_and_alert_baseline_documented(self) -> None:
        document = (SERVICE_ROOT / "docs/observability_baseline.md").read_text(encoding="utf-8")

        self.assertIn("Dashboard Panels", document)
        self.assertIn("Alerts", document)
        self.assertIn("Outbox lag", document)
        self.assertIn("Provider errors", document)
