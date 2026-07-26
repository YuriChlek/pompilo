from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class Stage15FailureRecoverySmokeTests(unittest.TestCase):
    def test_documented_failure_modes_have_automated_coverage(self) -> None:
        test_module = (SERVICE_ROOT / "tests/unit/test_failure_recovery.py").read_text(encoding="utf-8")

        self.assertIn("test_provider_timeout_is_retryable_and_observable", test_module)
        self.assertIn("test_partial_response_becomes_incomplete_and_has_no_ready_snapshot_input", test_module)
        self.assertIn("test_duplicate_job_can_be_retried_as_noop_after_unique_insert", test_module)
        self.assertIn("test_async_transaction_boundary_rolls_back_on_commit_failure", test_module)
        self.assertIn("test_outbox_publisher_restart_redelivers_pending_event", test_module)
        self.assertIn("test_snapshot_membership_is_immutable_after_candle_correction", test_module)

    def test_failure_recovery_uses_existing_layer_boundaries(self) -> None:
        test_module = (SERVICE_ROOT / "tests/unit/test_failure_recovery.py").read_text(encoding="utf-8")

        self.assertIn("BinanceSpotAdapter", test_module)
        self.assertIn("detect_candle_range_status", test_module)
        self.assertIn("_transaction_boundary", test_module)
        self.assertIn("OutboxPublisherService", test_module)
        self.assertIn("build_snapshot_membership", test_module)
