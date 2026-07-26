from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class PersistentBackendStage10SymbolsSyncSmokeTests(unittest.TestCase):
    def test_symbols_sync_is_not_exposed_as_public_cli_command(self) -> None:
        main = (SERVICE_ROOT / "src/market_data_service/main.py").read_text(encoding="utf-8")
        maintenance = (SERVICE_ROOT / "src/market_data_service/runtime/maintenance.py").read_text(encoding="utf-8")
        readme = (SERVICE_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertNotIn("symbols:sync", main)
        self.assertNotIn("run_symbols_sync", main)
        self.assertNotIn("run_symbols_sync", maintenance)
        self.assertNotIn("symbols:sync", readme)
        self.assertIn("build_runtime_container", maintenance)
        self.assertNotIn("run_http_server", maintenance)
        self.assertNotIn("scheduler_lifecycle.start()", maintenance)
        self.assertNotIn("outbox_publisher_lifecycle.start()", maintenance)

    def test_symbols_sync_repository_uses_postgres_upsert_for_idempotency(self) -> None:
        repository = (
            SERVICE_ROOT / "src/market_data_service/persistence/repositories/symbol_registry_repository.py"
        ).read_text(encoding="utf-8")

        self.assertIn("on_conflict_do_update", repository)
        self.assertIn('index_elements=["canonical_symbol"]', repository)
        self.assertIn('index_elements=["source", "canonical_symbol"]', repository)

    def test_docker_compose_no_longer_exposes_symbols_sync_as_production_job(self) -> None:
        compose = (SERVICE_ROOT.parent / "infra/compose/docker-compose.platform.yaml").read_text(encoding="utf-8")

        self.assertNotIn("market_data_symbols_sync:", compose)
        self.assertNotIn('command: ["python", "-m", "market_data_service.main", "symbols:sync"]', compose)
        self.assertIn("market_data_migrate:", compose)
        self.assertIn('command: ["python", "-m", "market_data_service.main", "collect"]', compose)
        self.assertIn("condition: service_completed_successfully", compose)
