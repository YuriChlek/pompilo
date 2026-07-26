from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class PersistentBackendStage1CliSkeletonSmokeTests(unittest.TestCase):
    def test_market_data_main_entrypoint_exists_with_required_commands(self) -> None:
        entrypoint = (SERVICE_ROOT / "src/market_data_service/main.py").read_text(encoding="utf-8")

        for command in (
            "serve",
            "collect",
            "candles:get",
            "healthcheck",
            "db:check",
            "outbox:cleanup",
        ):
            self.assertIn(command, entrypoint)

        for removed_command in (
            "scheduler",
            "outbox-publisher",
            "db:revision",
            "scheduler:run-once",
            "sync:run-next",
            "outbox:publish-once",
            "symbols:sync",
            "backfill",
            "gaps:scan",
            "outbox:replay",
        ):
            self.assertNotIn(f'"{removed_command}"', entrypoint)

        self.assertIn("make_url", entrypoint)
        self.assertIn("Database configuration is valid", entrypoint)

    def test_market_data_migrate_job_still_runs_alembic_directly(self) -> None:
        compose = (SERVICE_ROOT.parent / "infra/compose/docker-compose.platform.yaml").read_text(encoding="utf-8")

        self.assertIn("market_data_migrate:", compose)
        self.assertIn('command: ["alembic", "upgrade", "head"]', compose)
        self.assertNotIn('market_data_service.main", "db:migrate"', compose)
