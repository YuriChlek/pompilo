from __future__ import annotations

from pathlib import Path
import unittest


class MigrationContractTests(unittest.TestCase):
    def test_migration_files_are_ordered_and_present_for_runtime_state_changes(self) -> None:
        migration_names = sorted(path.name for path in (Path(__file__).resolve().parent.parent / "migrations").glob("*.sql"))

        self.assertEqual(
            migration_names,
            [
                "001_create_d1_candles.sql",
                "002_create_spot_ledger.sql",
                "003_create_4h_candles.sql",
                "004_add_no_loss_audit.sql",
                "005_add_entry_count.sql",
                "006_add_position_exit_state.sql",
            ],
        )

    def test_runtime_state_migrations_are_idempotent(self) -> None:
        migrations_dir = Path(__file__).resolve().parent.parent / "migrations"
        add_entry_count_sql = (migrations_dir / "005_add_entry_count.sql").read_text(encoding="utf-8")
        add_position_exit_state_sql = (migrations_dir / "006_add_position_exit_state.sql").read_text(encoding="utf-8")
        add_no_loss_audit_sql = (migrations_dir / "004_add_no_loss_audit.sql").read_text(encoding="utf-8")

        self.assertIn("ADD COLUMN IF NOT EXISTS entry_count", add_entry_count_sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS _spot_trading_bot.position_exit_state", add_position_exit_state_sql)
        self.assertIn("ADD COLUMN IF NOT EXISTS realized_pnl_usdt", add_no_loss_audit_sql)
        self.assertIn("CREATE INDEX IF NOT EXISTS order_ledger_signal_dedupe_idx", add_no_loss_audit_sql)
