import unittest
from dataclasses import replace

from domain.allocation import calculate_symbol_entry_budget
from domain.models import InventorySnapshot, RiskDecision
from domain.strategy_config import DEFAULT_STRATEGY_CONFIG


def _risk_decision(**overrides) -> RiskDecision:
    decision = RiskDecision(
        can_trade=True,
        pause_entries=False,
        force_risk_off=False,
        cancel_entries=False,
        allow_exit_only=False,
    )
    return replace(decision, **overrides)


class AllocationRefactorTests(unittest.TestCase):
    def test_budget_scales_in_usdt_not_coin_units_for_low_priced_assets(self):
        config = replace(
            DEFAULT_STRATEGY_CONFIG,
            risk=replace(
                DEFAULT_STRATEGY_CONFIG.risk,
                max_symbol_inventory_pct_of_equity=0.03,
                max_symbol_new_entry_pct_of_free_quote=0.05,
                max_symbol_notional_cap=400.0,
                min_symbol_entry_notional=25.0,
            ),
        )
        inventory = InventorySnapshot(base_balance=0.0, quote_balance=10_000.0, reserved_quote=0.0, mark_price=0.6)

        budget = calculate_symbol_entry_budget(inventory, _risk_decision(), config)

        self.assertEqual(budget, 300.0)

    def test_budget_scales_in_usdt_not_coin_units_for_higher_priced_assets(self):
        config = replace(
            DEFAULT_STRATEGY_CONFIG,
            risk=replace(
                DEFAULT_STRATEGY_CONFIG.risk,
                max_symbol_inventory_pct_of_equity=0.03,
                max_symbol_new_entry_pct_of_free_quote=0.05,
                max_symbol_notional_cap=400.0,
                min_symbol_entry_notional=25.0,
            ),
        )
        inventory = InventorySnapshot(base_balance=0.0, quote_balance=10_000.0, reserved_quote=0.0, mark_price=82.68)

        budget = calculate_symbol_entry_budget(inventory, _risk_decision(), config)

        self.assertEqual(budget, 300.0)

    def test_budget_is_zero_when_symbol_inventory_cap_is_already_used(self):
        config = replace(
            DEFAULT_STRATEGY_CONFIG,
            risk=replace(
                DEFAULT_STRATEGY_CONFIG.risk,
                max_symbol_inventory_pct_of_equity=0.03,
                max_symbol_new_entry_pct_of_free_quote=0.05,
                max_symbol_notional_cap=400.0,
                min_symbol_entry_notional=25.0,
            ),
        )
        inventory = InventorySnapshot(base_balance=3.0, quote_balance=700.0, reserved_quote=0.0, mark_price=100.0)

        budget = calculate_symbol_entry_budget(inventory, _risk_decision(), config)

        self.assertEqual(budget, 0.0)
