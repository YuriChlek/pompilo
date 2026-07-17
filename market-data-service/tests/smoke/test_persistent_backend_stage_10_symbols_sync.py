from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class PersistentBackendStage10SymbolsSyncSmokeTests(unittest.TestCase):
    def test_symbols_sync_command_uses_runtime_container_without_http_server(self) -> None:
        main = (SERVICE_ROOT / "src/market_data_service/main.py").read_text(encoding="utf-8")
        maintenance = (SERVICE_ROOT / "src/market_data_service/runtime/maintenance.py").read_text(encoding="utf-8")

        self.assertIn("symbols:sync", main)
        self.assertIn("run_symbols_sync", main)
        self.assertIn("build_runtime_container", maintenance)
        self.assertIn("symbol_registry_sync.sync_default_symbols()", maintenance)
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

    def test_docker_compose_exposes_symbols_sync_as_one_shot_job(self) -> None:
        compose = (SERVICE_ROOT.parent / "infra/compose/docker-compose.platform.yaml").read_text(encoding="utf-8")

        self.assertIn("market_data_symbols_sync:", compose)
        self.assertIn('command: ["python", "-m", "market_data_service.main", "symbols:sync"]', compose)
        self.assertIn("market_data_migrate:", compose)
        self.assertIn("condition: service_completed_successfully", compose)
