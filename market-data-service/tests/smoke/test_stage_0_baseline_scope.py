from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class Stage0BaselineScopeTests(unittest.TestCase):
    def test_baseline_documents_no_existing_application_runtime_switch(self) -> None:
        baseline = (SERVICE_ROOT / "docs/stage_0_baseline.md").read_text(encoding="utf-8")

        self.assertIn("Initial Symbol Universe", baseline)
        self.assertIn("Supported Timeframes", baseline)
        self.assertIn("No existing application runtime switch", baseline)
        self.assertNotIn("MARKET_DATA_SERVICE_ENABLED", baseline)
        self.assertNotIn("spot_grid_bot", baseline)
        self.assertNotIn("spot-greenwich-bot", baseline)


class StandardsScopeTests(unittest.TestCase):
    def test_standards_match_autonomous_market_data_scope(self) -> None:
        standards = (SERVICE_ROOT / "STANDARDS.md").read_text(encoding="utf-8")

        self.assertIn("CandleBatchReady", standards)
        self.assertNotIn("StrategyMarketDataReady", standards)
        self.assertNotIn("readiness_repository", standards)
        self.assertNotIn("Strategy Readiness Rules", standards)
