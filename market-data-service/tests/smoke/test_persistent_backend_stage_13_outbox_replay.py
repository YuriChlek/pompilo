from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class PersistentBackendStage13OutboxReplaySmokeTests(unittest.TestCase):
    def test_outbox_replay_command_has_recovery_arguments(self) -> None:
        main = (SERVICE_ROOT / "src/market_data_service/main.py").read_text(encoding="utf-8")
        maintenance = (SERVICE_ROOT / "src/market_data_service/runtime/maintenance.py").read_text(encoding="utf-8")

        self.assertIn("outbox:replay", main)
        self.assertIn("--from-id", main)
        self.assertIn("--to-id", main)
        self.assertIn("--dry-run", main)
        self.assertIn("Outbox replay report", main)
        self.assertIn("run_outbox_replay", maintenance)
        self.assertNotIn("run_http_server", maintenance)

    def test_replay_service_is_duplicate_safe_by_skipping_published_events(self) -> None:
        service = (
            SERVICE_ROOT / "src/market_data_service/application/services/outbox_replay_service.py"
        ).read_text(encoding="utf-8")
        repository = (
            SERVICE_ROOT / "src/market_data_service/persistence/repositories/outbox_repository.py"
        ).read_text(encoding="utf-8")

        self.assertIn("event.status != OutboxStatus.PUBLISHED", service)
        self.assertIn("if command.dry_run", service)
        self.assertIn("broker.publish(event)", service)
        self.assertIn("mark_published", service)
        self.assertIn("list_replay_events", repository)
