from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class Stage3MarketCandlesTests(unittest.TestCase):
    def test_candle_metadata_declares_partitioned_parent_contract(self) -> None:
        module = (
            SERVICE_ROOT / "src/market_data_service/persistence/tables/market_candles_tables.py"
        ).read_text(encoding="utf-8")

        self.assertIn('"market_candles"', module)
        self.assertIn('postgresql_partition_by="LIST (timeframe)"', module)
        self.assertIn('PrimaryKeyConstraint("timeframe", "candle_id"', module)
        self.assertIn('UniqueConstraint("source", "canonical_symbol", "timeframe", "open_time"', module)
        self.assertIn('"taker_buy_base_volume"', module)
        self.assertIn('"taker_sell_quote_volume"', module)

    def test_candle_migration_creates_parent_and_timeframe_partitions(self) -> None:
        migration = (
            SERVICE_ROOT / "alembic/versions/20260714_0003_create_market_candles_partitions.py"
        ).read_text(encoding="utf-8")

        self.assertIn('revision = "0003_market_candles_partitions"', migration)
        self.assertIn('down_revision = "0002_seed_symbol_registry"', migration)
        self.assertIn("PARTITION BY LIST (timeframe)", migration)
        self.assertIn("market_candles_1h", migration)
        self.assertIn("market_candles_4h", migration)
        self.assertIn("market_candles_1d", migration)
        self.assertIn("CONSTRAINT market_candles_natural_key_uq UNIQUE", migration)

    def test_candle_repository_uses_on_conflict_do_nothing(self) -> None:
        repository = (
            SERVICE_ROOT / "src/market_data_service/persistence/repositories/candle_repository.py"
        ).read_text(encoding="utf-8")

        self.assertIn(".on_conflict_do_nothing(", repository)
        self.assertIn('index_elements=["source", "canonical_symbol", "timeframe", "open_time"]', repository)
