from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class Stage5SingleSymbolSyncSmokeTests(unittest.TestCase):
    def test_single_symbol_sync_service_has_expected_boundaries(self) -> None:
        service = (
            SERVICE_ROOT / "src/market_data_service/application/services/single_symbol_sync_service.py"
        ).read_text(encoding="utf-8")

        self.assertIn("assert_sync_mapping_active", service)
        self.assertIn("acquire_sync_lock", service)
        self.assertIn("fetch_closed_candles", service)
        self.assertIn("insert_closed_candles", service)
        self.assertIn("dry_run", service)
        self.assertNotIn("outbox", service.lower())
        self.assertNotIn("market_candles_", service)
        self.assertNotIn("sqlalchemy", service.lower())

    def test_advisory_lock_repository_uses_transaction_scoped_postgres_lock(self) -> None:
        repository = (
            SERVICE_ROOT / "src/market_data_service/persistence/repositories/advisory_lock_repository.py"
        ).read_text(encoding="utf-8")

        self.assertIn("pg_try_advisory_xact_lock", repository)
        self.assertIn("hashtextextended", repository)
        self.assertIn("market_data_sync:", repository)
