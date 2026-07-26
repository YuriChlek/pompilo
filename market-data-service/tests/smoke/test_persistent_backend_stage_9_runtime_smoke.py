from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class PersistentBackendStage9RuntimeSmokeTests(unittest.TestCase):
    def test_docker_runtime_smoke_script_covers_health_db_and_redis_without_legacy_bots(self) -> None:
        script = (SERVICE_ROOT / "scripts/docker_runtime_smoke.sh").read_text(encoding="utf-8")

        for required_text in (
            "market_data_migrate",
            "collect --once",
            "compose up --build -d market_data",
            "python -m market_data_service.main healthcheck",
            "/metrics",
            "_market_data.sync_jobs",
            "_market_data.market_data_batches",
            "_market_data.market_candles",
            "_market_data.market_snapshots",
            "_market_data.outbox_events",
            "MARKET_DATA_PROVIDER_MODE",
            "redis-cli",
            "XLEN",
            "Smoke failed:",
        ):
            self.assertIn(required_text, script)

        self.assertNotIn("spot_grid_bot", script)
        self.assertNotIn("spot-greenwich-bot", script)

    def test_runtime_smoke_document_has_command_and_failure_guidance(self) -> None:
        document = (SERVICE_ROOT / "docs/docker_runtime_smoke.md").read_text(encoding="utf-8")

        self.assertIn("bash market-data-service/scripts/docker_runtime_smoke.sh", document)
        self.assertIn("does not use or inspect legacy bot directories", document)
        self.assertIn("market_candles", document)
        self.assertIn("market_snapshots", document)
        self.assertIn("Redis Stream", document)
