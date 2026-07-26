from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class PersistentBackendStage11BackfillCommandSmokeTests(unittest.TestCase):
    def test_backfill_is_internal_and_not_exposed_as_public_cli_command(self) -> None:
        main = (SERVICE_ROOT / "src/market_data_service/main.py").read_text(encoding="utf-8")
        service = (
            SERVICE_ROOT / "src/market_data_service/application/services/backfill_command_service.py"
        ).read_text(encoding="utf-8")
        maintenance = (SERVICE_ROOT / "src/market_data_service/runtime/maintenance.py").read_text(encoding="utf-8")
        readme = (SERVICE_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertNotIn('add_parser("backfill"', main)
        self.assertNotIn("run_backfill", main)
        self.assertNotIn("run_backfill", maintenance)
        self.assertNotIn("python -m market_data_service.main backfill", readme)
        self.assertIn("BackfillCommandService", service)
        self.assertIn("MarketDataBackfillRequested.from_gap", service)
        self.assertIn("request_backfill(event)", service)
        self.assertNotIn("run_http_server", maintenance)

    def test_backfill_has_batch_size_and_concurrency_limits(self) -> None:
        settings = (SERVICE_ROOT / "src/market_data_service/config/settings.py").read_text(encoding="utf-8")
        service = (
            SERVICE_ROOT / "src/market_data_service/application/services/backfill_command_service.py"
        ).read_text(encoding="utf-8")

        self.assertIn("MARKET_DATA_BACKFILL_BATCH_CANDLES", settings)
        self.assertIn("MARKET_DATA_BACKFILL_MAX_CONCURRENCY", settings)
        self.assertIn("batch_size_candles", service)
        self.assertIn("max_concurrency", service)

    def test_docker_compose_no_longer_exposes_backfill_as_production_job(self) -> None:
        compose = (SERVICE_ROOT.parent / "infra/compose/docker-compose.platform.yaml").read_text(encoding="utf-8")

        self.assertNotIn("market_data_backfill:", compose)
        self.assertNotIn("MARKET_DATA_BACKFILL_SYMBOL", compose)
        self.assertNotIn("MARKET_DATA_BACKFILL_FROM", compose)
        self.assertNotIn("MARKET_DATA_BACKFILL_TO", compose)
        self.assertIn('command: ["python", "-m", "market_data_service.main", "collect"]', compose)
