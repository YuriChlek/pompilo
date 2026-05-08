from __future__ import annotations

import unittest

from utils.db_actions import CREATE_SPOT_ORDER_LEDGER_SQL, CREATE_SPOT_POSITION_EXIT_STATE_SQL, CREATE_SPOT_POSITION_STATE_SQL


class DatabaseSchemaContractTests(unittest.TestCase):
    def test_position_state_schema_includes_entry_count(self) -> None:
        self.assertIn("entry_count INTEGER NOT NULL DEFAULT 0", CREATE_SPOT_POSITION_STATE_SQL)

    def test_position_exit_state_schema_is_declared_in_create_tables(self) -> None:
        self.assertIn("CREATE TABLE IF NOT EXISTS {schema}.position_exit_state", CREATE_SPOT_POSITION_EXIT_STATE_SQL)
        self.assertIn("first_take_profit_done BOOLEAN NOT NULL DEFAULT FALSE", CREATE_SPOT_POSITION_EXIT_STATE_SQL)
        self.assertIn("first_take_profit_order_id TEXT", CREATE_SPOT_POSITION_EXIT_STATE_SQL)

    def test_order_ledger_schema_includes_no_loss_and_signal_audit_fields(self) -> None:
        self.assertIn("realized_pnl_usdt NUMERIC", CREATE_SPOT_ORDER_LEDGER_SQL)
        self.assertIn("realized_pnl_pct NUMERIC", CREATE_SPOT_ORDER_LEDGER_SQL)
        self.assertIn("no_loss_check_price NUMERIC", CREATE_SPOT_ORDER_LEDGER_SQL)
        self.assertIn("signal_timeframe TEXT", CREATE_SPOT_ORDER_LEDGER_SQL)
        self.assertIn("signal_candle_id TEXT", CREATE_SPOT_ORDER_LEDGER_SQL)
