from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class Stage8SnapshotModelSmokeTests(unittest.TestCase):
    def test_snapshot_tables_and_migration_exist(self) -> None:
        tables = (SERVICE_ROOT / "src/market_data_service/persistence/tables/market_snapshots_tables.py").read_text(
            encoding="utf-8"
        )
        migration = (SERVICE_ROOT / "alembic/versions/20260714_0005_create_market_snapshots.py").read_text(
            encoding="utf-8"
        )

        for table_name in ("market_snapshots", "market_snapshot_candles"):
            self.assertIn(table_name, tables)
            self.assertIn(table_name, migration)

        for field_name in (
            "data_hash",
            "snapshot_version",
            "last_closed_candle_time",
            "lookback_start_time",
            "lookback_end_time",
            "candle_hash_at_snapshot",
        ):
            self.assertIn(field_name, tables)
            self.assertIn(field_name, migration)

        self.assertIn("market_snapshots_logical_data_hash_uq", migration)
        self.assertIn("market_snapshot_candles_pk", migration)

    def test_snapshot_repository_reads_only_through_membership(self) -> None:
        repository = (
            SERVICE_ROOT / "src/market_data_service/persistence/repositories/snapshot_repository.py"
        ).read_text(encoding="utf-8")

        self.assertIn("create_snapshot_if_changed", repository)
        self.assertIn("calculate_snapshot_data_hash", repository)
        self.assertIn("build_snapshot_membership", repository)
        self.assertIn("read_snapshot_candles", repository)
        self.assertIn("market_snapshot_candles.join", repository)
        self.assertIn("market_snapshot_candles.c.snapshot_id == snapshot_id", repository)

    def test_snapshot_hash_is_pure_domain_logic(self) -> None:
        module = (SERVICE_ROOT / "src/market_data_service/domain/snapshot_hash.py").read_text(encoding="utf-8")

        self.assertIn("calculate_snapshot_data_hash", module)
        self.assertIn("provider_payload_hash", module)
        self.assertNotIn("sqlalchemy", module.lower())
        self.assertNotIn("persistence", module.lower())
