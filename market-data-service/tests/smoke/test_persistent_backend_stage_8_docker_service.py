from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class PersistentBackendStage8DockerServiceSmokeTests(unittest.TestCase):
    def test_market_data_service_is_long_running_and_keeps_migration_job_separate(self) -> None:
        compose = (SERVICE_ROOT.parent / "infra/compose/docker-compose.platform.yaml").read_text(encoding="utf-8")

        self.assertIn("market_data_migrate:", compose)
        self.assertIn('command: ["alembic", "upgrade", "head"]', compose)
        self.assertIn("market_data:", compose)
        self.assertIn('command: ["python", "-m", "market_data_service.main", "collect"]', compose)
        self.assertIn("market_data_migrate:", compose)
        self.assertIn("condition: service_completed_successfully", compose)
        self.assertIn("condition: service_healthy", compose)
        self.assertIn('test: ["CMD", "python", "-m", "market_data_service.main", "healthcheck"]', compose)
        self.assertIn("MARKET_DATA_REDIS_URL: redis://redis:6379/0", compose)
        self.assertIn("MARKET_DATA_OUTBOX_STREAM: ${MARKET_DATA_OUTBOX_STREAM:-market-data-events}", compose)
        self.assertIn("MARKET_DATA_OUTBOX_STREAM_MAXLEN", compose)
        self.assertIn("MARKET_DATA_REDIS_EVENT_RETENTION_DAYS", compose)
        self.assertIn("MARKET_DATA_OUTBOX_RETENTION_DAYS", compose)
        self.assertIn("MARKET_DATA_HTTP_PORT: 8010", compose)
        self.assertIn("MARKET_DATA_COLLECT_POLL_INTERVAL_SECONDS", compose)
        self.assertIn("MARKET_DATA_COLLECT_MAX_JOBS_PER_TICK", compose)
        self.assertIn("MARKET_DATA_PROVIDER_PRIORITY", compose)
        self.assertNotIn('market_data_service.main", "db:migrate"', compose)
        self.assertNotIn("market_data_symbols_sync:", compose)
        self.assertNotIn("market_data_backfill:", compose)

    def test_market_data_http_port_is_exposed_only_in_dev_override(self) -> None:
        platform_compose = (SERVICE_ROOT.parent / "infra/compose/docker-compose.platform.yaml").read_text(
            encoding="utf-8"
        )
        dev_compose = (SERVICE_ROOT.parent / "infra/compose/docker-compose.dev.yaml").read_text(encoding="utf-8")

        market_data_block = platform_compose.split("  market_data:", 1)[1].split("  bot_platform_migrate:", 1)[0]
        self.assertNotIn("ports:", market_data_block)
        self.assertIn("market_data:", dev_compose)
        self.assertIn("${MARKET_DATA_HTTP_HOST_PORT:-8010}:8010", dev_compose)
