from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class Stage1SchemaShellTests(unittest.TestCase):
    def test_project_declares_required_market_data_stack(self) -> None:
        pyproject = (SERVICE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('"SQLAlchemy>=2.0"', pyproject)
        self.assertIn('"asyncpg>=0.29"', pyproject)
        self.assertIn('"alembic>=1.13"', pyproject)

    def test_metadata_declares_market_data_schema_and_shell_tables(self) -> None:
        metadata_module = (
            SERVICE_ROOT / "src/market_data_service/persistence/tables/metadata.py"
        ).read_text(encoding="utf-8")
        market_symbols_module = (
            SERVICE_ROOT / "src/market_data_service/persistence/tables/market_symbols_tables.py"
        ).read_text(encoding="utf-8")
        provider_symbols_module = (
            SERVICE_ROOT / "src/market_data_service/persistence/tables/provider_symbols_tables.py"
        ).read_text(encoding="utf-8")

        self.assertIn('MARKET_DATA_SCHEMA = "_market_data"', metadata_module)
        self.assertIn('"market_symbols"', market_symbols_module)
        self.assertIn('"provider_symbols"', provider_symbols_module)
        self.assertIn("provider_symbols_source_canonical_symbol_uq", provider_symbols_module)
        self.assertIn("provider_symbols_source_provider_symbol_uq", provider_symbols_module)

    def test_alembic_environment_uses_service_metadata(self) -> None:
        env_module = (SERVICE_ROOT / "alembic/env.py").read_text(encoding="utf-8")

        self.assertIn("from market_data_service.config.database_config import get_database_url", env_module)
        self.assertIn("from market_data_service.persistence.tables import metadata as target_metadata", env_module)
        self.assertIn("include_schemas=True", env_module)
        self.assertIn("async_engine_from_config", env_module)

    def test_initial_migration_creates_and_drops_only_market_data_shell(self) -> None:
        migration = (
            SERVICE_ROOT / "alembic/versions/20260713_0001_create_market_data_schema_shell.py"
        ).read_text(encoding="utf-8")

        self.assertIn('MARKET_DATA_SCHEMA = "_market_data"', migration)
        self.assertIn("CreateSchema(MARKET_DATA_SCHEMA, if_not_exists=True)", migration)
        self.assertIn('op.create_table(\n        "market_symbols"', migration)
        self.assertIn('op.create_table(\n        "provider_symbols"', migration)
        self.assertIn('op.drop_table("provider_symbols", schema=MARKET_DATA_SCHEMA)', migration)
        self.assertIn('op.drop_table("market_symbols", schema=MARKET_DATA_SCHEMA)', migration)
        self.assertIn("DropSchema(MARKET_DATA_SCHEMA, if_exists=True)", migration)
        self.assertNotIn("_candles_trading_data", migration)
