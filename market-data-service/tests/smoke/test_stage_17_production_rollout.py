from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class Stage17ProductionRolloutSmokeTests(unittest.TestCase):
    def test_rollout_document_covers_scope_alerts_and_rollback(self) -> None:
        document = (SERVICE_ROOT / "docs/production_rollout.md").read_text(encoding="utf-8")

        self.assertIn("production data-ingestion layer only", document)
        self.assertIn("does not connect bots", document)
        self.assertIn("one provider symbol group", document)
        self.assertIn("one timeframe group", document)
        self.assertIn("Alerts To Check", document)
        self.assertIn("Rollback Decision Points", document)
        self.assertIn("Schema rollback is not required", document)
        self.assertIn("consumer integration plan", document)

    def test_rollout_service_has_promotion_hold_and_rollback_decisions(self) -> None:
        service = (
            SERVICE_ROOT / "src/market_data_service/application/services/production_rollout_service.py"
        ).read_text(encoding="utf-8")

        self.assertIn("build_symbol_group_rollout_scopes", service)
        self.assertIn("build_timeframe_group_rollout_scopes", service)
        self.assertIn("RolloutDecisionStatus.PROMOTE", service)
        self.assertIn("RolloutDecisionStatus.HOLD", service)
        self.assertIn("RolloutDecisionStatus.ROLLBACK", service)
        self.assertIn("ready_for_consumer_integration_plan=True", service)

    def test_stage_17_does_not_add_runtime_bot_or_consumer_integration(self) -> None:
        service_root_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (SERVICE_ROOT / "src/market_data_service").rglob("*.py")
        )

        self.assertNotIn("spot_grid_bot", service_root_text)
        self.assertNotIn("spot-greenwich-bot", service_root_text)
        self.assertNotIn("StrategyMarketDataReady", service_root_text)
