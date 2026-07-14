from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class Stage2SymbolRegistrySeedTests(unittest.TestCase):
    def test_seed_migration_adds_registry_status_and_timeframes(self) -> None:
        migration = (
            SERVICE_ROOT / "alembic/versions/20260714_0002_seed_symbol_registry.py"
        ).read_text(encoding="utf-8")

        self.assertIn('revision = "0002_seed_symbol_registry"', migration)
        self.assertIn('down_revision = "0001_create_market_data_schema_shell"', migration)
        self.assertIn('SUPPORTED_TIMEFRAMES = ["1h", "4h", "1d"]', migration)
        self.assertIn('Column("status"', migration)
        self.assertIn('Column("supported_timeframes"', migration)
        self.assertIn("on_conflict_do_update", migration)

    def test_seed_migration_covers_current_bot_symbols(self) -> None:
        migration = (
            SERVICE_ROOT / "alembic/versions/20260714_0002_seed_symbol_registry.py"
        ).read_text(encoding="utf-8")

        for provider_symbol in ("BTCUSDT", "ETHUSDT", "LTCUSDT", "SOLUSDT", "SUIUSDT", "TAOUSDT", "XRPUSDT"):
            canonical_symbol = provider_symbol.replace("USDT", "/USDT")
            self.assertIn(canonical_symbol, migration)

    def test_metadata_matches_seeded_registry_contract(self) -> None:
        market_symbols_module = (
            SERVICE_ROOT / "src/market_data_service/persistence/tables/market_symbols_tables.py"
        ).read_text(encoding="utf-8")
        provider_symbols_module = (
            SERVICE_ROOT / "src/market_data_service/persistence/tables/provider_symbols_tables.py"
        ).read_text(encoding="utf-8")

        self.assertIn('Column("status"', market_symbols_module)
        self.assertNotIn('Column("is_active"', market_symbols_module)
        self.assertIn('Column("status"', provider_symbols_module)
        self.assertIn('Column("supported_timeframes"', provider_symbols_module)
        self.assertIn('Column("metadata_json"', provider_symbols_module)
        self.assertNotIn('Column("is_active"', provider_symbols_module)
