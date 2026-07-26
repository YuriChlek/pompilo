from __future__ import annotations

import unittest

from market_data_service.observability.alerts import DEFAULT_ALERT_RULES, evaluate_alerts
from market_data_service.observability.metrics import (
    MARKET_DATA_GAP_COUNT,
    MARKET_DATA_OUTBOX_LAG_SECONDS,
    MARKET_DATA_PROVIDER_ERRORS_TOTAL,
    MARKET_DATA_SYNC_JOBS_STUCK_TOTAL,
    MetricSample,
)


class ObservabilityAlertsTests(unittest.TestCase):
    def test_default_alert_rules_cover_stage_14_failure_modes(self) -> None:
        names = {rule.name for rule in DEFAULT_ALERT_RULES}

        self.assertIn("market_data_snapshot_stale", names)
        self.assertIn("market_data_gap_detected", names)
        self.assertIn("market_data_provider_errors_high", names)
        self.assertIn("market_data_outbox_lag_high", names)
        self.assertIn("market_data_sync_jobs_stuck", names)

    def test_evaluate_alerts_returns_only_threshold_breaches(self) -> None:
        alerts = evaluate_alerts(
            [
                MetricSample(MARKET_DATA_GAP_COUNT, 1.0, {"timeframe": "1h"}),
                MetricSample(MARKET_DATA_PROVIDER_ERRORS_TOTAL, 4.0, {}),
                MetricSample(MARKET_DATA_OUTBOX_LAG_SECONDS, 301.0, {}),
                MetricSample(MARKET_DATA_SYNC_JOBS_STUCK_TOTAL, 2.0, {}),
            ]
        )

        self.assertEqual(
            {alert.name for alert in alerts},
            {"market_data_gap_detected", "market_data_outbox_lag_high", "market_data_sync_jobs_stuck"},
        )
