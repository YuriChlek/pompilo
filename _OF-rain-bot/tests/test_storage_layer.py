from __future__ import annotations

import unittest

from trading.infrastructure.storage import (
    DatabaseMigrationRunner,
    QueuedRuntimeEventRepository,
    RuntimeEventRepository,
    get_index_statements,
    get_schema_name,
    get_schema_statement,
    get_table_statements,
)
from utils.config import APP_CONFIG


class StorageLayerTests(unittest.TestCase):
    def test_schema_helpers_use_canonical_app_config(self) -> None:
        schema_name = get_schema_name()

        self.assertEqual(schema_name, APP_CONFIG.orderflow.db_schema)
        self.assertIn(schema_name, get_schema_statement())
        self.assertIn(schema_name, get_table_statements()["order_events"])
        self.assertIn(schema_name, get_index_statements()["order_events_symbol_created_at_idx"])

    def test_canonical_storage_classes_are_exposed(self) -> None:
        self.assertTrue(issubclass(QueuedRuntimeEventRepository, RuntimeEventRepository))
        self.assertEqual(RuntimeEventRepository.__name__, "RuntimeEventRepository")
        self.assertEqual(QueuedRuntimeEventRepository.__name__, "QueuedRuntimeEventRepository")
        self.assertEqual(DatabaseMigrationRunner.__name__, "DatabaseMigrationRunner")
