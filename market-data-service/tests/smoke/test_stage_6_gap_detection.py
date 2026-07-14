from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class Stage6GapDetectionSmokeTests(unittest.TestCase):
    def test_gap_detector_declares_required_statuses_and_fields(self) -> None:
        detector = (SERVICE_ROOT / "src/market_data_service/domain/candle_gap_detector.py").read_text(encoding="utf-8")
        enums = (SERVICE_ROOT / "src/market_data_service/domain/enums.py").read_text(encoding="utf-8")

        for status in ("COMPLETE", "INCOMPLETE", "GAP_DETECTED", "STALE"):
            self.assertIn(status, enums)

        self.assertIn("MissingInterval", detector)
        self.assertIn("gap_count", detector)
        self.assertIn("duplicate_count", detector)
        self.assertIn("wrong_duration_count", detector)

    def test_sync_result_includes_range_status_without_readiness(self) -> None:
        sync_models = (SERVICE_ROOT / "src/market_data_service/application/sync_models.py").read_text(encoding="utf-8")
        sync_service = (
            SERVICE_ROOT / "src/market_data_service/application/services/single_symbol_sync_service.py"
        ).read_text(encoding="utf-8")

        self.assertIn("range_status", sync_models)
        self.assertIn("gap_count", sync_models)
        self.assertIn("detect_candle_range_status", sync_service)
        self.assertNotIn("StrategyMarketDataReady", sync_service)
        self.assertNotIn("outbox", sync_service.lower())
