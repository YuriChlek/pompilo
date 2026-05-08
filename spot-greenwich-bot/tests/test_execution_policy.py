from __future__ import annotations

import unittest
from decimal import Decimal

from domain.execution import apply_portfolio_position_limit, decide_spot_execution
from domain.models import ExecutionDecision, PositionState, SpotSignal


class ExecutionPolicyTests(unittest.TestCase):
    def test_first_buy_is_allowed_without_existing_position(self) -> None:
        signal = SpotSignal("ETHUSDT", "buy", Decimal("100"), "2026-01-01", "test")
        state = PositionState("ETHUSDT", Decimal("0"), Decimal("0"), Decimal("0"))
        decision = decide_spot_execution(signal, state, Decimal("500"))
        self.assertEqual(decision.action, "buy")
        self.assertEqual(decision.quote_amount, Decimal("25"))
        self.assertEqual(decision.quantity, Decimal("0.25000000"))

    def test_repeat_buy_requires_better_price_than_average_entry(self) -> None:
        signal = SpotSignal("ETHUSDT", "buy", Decimal("101"), "2026-01-01", "test")
        state = PositionState("ETHUSDT", Decimal("1"), Decimal("100"), Decimal("100"))
        decision = decide_spot_execution(signal, state, Decimal("500"))
        self.assertEqual(decision.action, "skip")
        self.assertEqual(decision.reason, "buy_price_not_better_than_avg_entry")

    def test_second_buy_uses_sixty_percent_of_base_size(self) -> None:
        signal = SpotSignal("ETHUSDT", "buy", Decimal("90"), "2026-01-01", "test")
        state = PositionState("ETHUSDT", Decimal("1"), Decimal("100"), Decimal("100"), entry_count=1)
        decision = decide_spot_execution(signal, state, Decimal("500"))
        self.assertEqual(decision.action, "buy")
        self.assertEqual(decision.quote_amount, Decimal("15"))
        self.assertEqual(decision.quantity, Decimal("0.16666666"))

    def test_third_buy_uses_thirty_percent_of_base_size(self) -> None:
        signal = SpotSignal("ETHUSDT", "buy", Decimal("90"), "2026-01-01", "test")
        state = PositionState("ETHUSDT", Decimal("1"), Decimal("100"), Decimal("100"), entry_count=2)
        decision = decide_spot_execution(signal, state, Decimal("500"))
        self.assertEqual(decision.action, "buy")
        self.assertEqual(decision.quote_amount, Decimal("7.5"))
        self.assertEqual(decision.quantity, Decimal("0.08333333"))

    def test_fourth_buy_is_blocked_by_max_entry_count(self) -> None:
        signal = SpotSignal("ETHUSDT", "buy", Decimal("90"), "2026-01-01", "test")
        state = PositionState("ETHUSDT", Decimal("1"), Decimal("100"), Decimal("100"), entry_count=3)
        decision = decide_spot_execution(signal, state, Decimal("500"))
        self.assertEqual(decision.action, "skip")
        self.assertEqual(decision.reason, "max_entry_count_reached")

    def test_buy_is_skipped_when_quote_balance_is_empty(self) -> None:
        signal = SpotSignal("ETHUSDT", "buy", Decimal("100"), "2026-01-01", "test")
        state = PositionState("ETHUSDT", Decimal("0"), Decimal("0"), Decimal("0"))
        decision = decide_spot_execution(signal, state, Decimal("0"))
        self.assertEqual(decision.action, "skip")
        self.assertEqual(decision.reason, "insufficient_quote_balance")

    def test_sell_requires_profitable_price(self) -> None:
        signal = SpotSignal("ETHUSDT", "sell", Decimal("99"), "2026-01-01", "test")
        state = PositionState("ETHUSDT", Decimal("1"), Decimal("100"), Decimal("100"))
        decision = decide_spot_execution(signal, state, Decimal("500"))
        self.assertEqual(decision.action, "skip")
        self.assertEqual(decision.reason, "sell_price_not_profitable")

    def test_sell_requires_default_one_percent_profit_buffer(self) -> None:
        signal = SpotSignal("ETHUSDT", "sell", Decimal("100.50"), "2026-01-01", "test")
        state = PositionState("ETHUSDT", Decimal("1"), Decimal("100"), Decimal("100"))
        decision = decide_spot_execution(signal, state, Decimal("500"))
        self.assertEqual(decision.action, "skip")
        self.assertEqual(decision.reason, "sell_price_not_profitable")

    def test_sell_uses_full_position_when_profitable(self) -> None:
        signal = SpotSignal("ETHUSDT", "sell", Decimal("110"), "2026-01-01", "test")
        state = PositionState("ETHUSDT", Decimal("1.25"), Decimal("100"), Decimal("125"))
        decision = decide_spot_execution(signal, state, Decimal("500"))
        self.assertEqual(decision.action, "sell")
        self.assertEqual(decision.quantity, Decimal("1.25"))

    def test_take_profit_upper1_sells_half_position(self) -> None:
        signal = SpotSignal("ETHUSDT", "sell", Decimal("120"), "2026-01-01", "greenwich_take_profit_upper1")
        state = PositionState("ETHUSDT", Decimal("1.25"), Decimal("100"), Decimal("125"))
        decision = decide_spot_execution(signal, state, Decimal("500"))
        self.assertEqual(decision.action, "sell")
        self.assertEqual(decision.reason, "greenwich_take_profit_upper1")
        self.assertEqual(decision.quantity, Decimal("0.62500000"))

    def test_buy_uses_smaller_quote_amount_when_atr_multiplier_is_below_one(self) -> None:
        signal = SpotSignal("ETHUSDT", "buy", Decimal("100"), "2026-01-01", "test")
        state = PositionState("ETHUSDT", Decimal("0"), Decimal("0"), Decimal("0"))
        decision = decide_spot_execution(signal, state, Decimal("500"), atr_size_multiplier=Decimal("0.5"))
        self.assertEqual(decision.action, "buy")
        self.assertEqual(decision.quote_amount, Decimal("12.5"))
        self.assertEqual(decision.quantity, Decimal("0.12500000"))

    def test_buy_uses_larger_quote_amount_when_atr_multiplier_is_above_one(self) -> None:
        signal = SpotSignal("ETHUSDT", "buy", Decimal("100"), "2026-01-01", "test")
        state = PositionState("ETHUSDT", Decimal("0"), Decimal("0"), Decimal("0"))
        decision = decide_spot_execution(signal, state, Decimal("500"), atr_size_multiplier=Decimal("1.5"))
        self.assertEqual(decision.action, "buy")
        self.assertEqual(decision.quote_amount, Decimal("37.5"))
        self.assertEqual(decision.quantity, Decimal("0.37500000"))

    def test_portfolio_limit_blocks_new_buy_when_cap_is_reached(self) -> None:
        decisions = {
            "ETHUSDT": ExecutionDecision("buy", "ETHUSDT", Decimal("100"), Decimal("1"), Decimal("100"), "test"),
            "BTCUSDT": ExecutionDecision("buy", "BTCUSDT", Decimal("100"), Decimal("1"), Decimal("100"), "test"),
        }
        position_states = {
            "ETHUSDT": PositionState("ETHUSDT", Decimal("0"), Decimal("0"), Decimal("0")),
            "BTCUSDT": PositionState("BTCUSDT", Decimal("0"), Decimal("0"), Decimal("0")),
            "SOLUSDT": PositionState("SOLUSDT", Decimal("1"), Decimal("100"), Decimal("100")),
            "SUIUSDT": PositionState("SUIUSDT", Decimal("1"), Decimal("100"), Decimal("100")),
            "XRPUSDT": PositionState("XRPUSDT", Decimal("1"), Decimal("100"), Decimal("100")),
        }

        constrained = apply_portfolio_position_limit(
            decisions,
            position_states,
            position_limit=3,
            priority_symbols=("BTCUSDT", "ETHUSDT"),
        )

        self.assertEqual(constrained["ETHUSDT"].action, "skip")
        self.assertEqual(constrained["ETHUSDT"].reason, "portfolio_position_limit_reached")
        self.assertEqual(constrained["BTCUSDT"].action, "skip")
        self.assertEqual(constrained["BTCUSDT"].reason, "portfolio_position_limit_reached")

    def test_portfolio_limit_keeps_priority_symbol_when_slots_are_limited(self) -> None:
        decisions = {
            "BTCUSDT": ExecutionDecision("buy", "BTCUSDT", Decimal("100"), Decimal("1"), Decimal("100"), "test"),
            "SUIUSDT": ExecutionDecision("buy", "SUIUSDT", Decimal("100"), Decimal("1"), Decimal("100"), "test"),
            "ETHUSDT": ExecutionDecision("buy", "ETHUSDT", Decimal("100"), Decimal("1"), Decimal("100"), "test"),
        }
        position_states = {
            "BTCUSDT": PositionState("BTCUSDT", Decimal("0"), Decimal("0"), Decimal("0")),
            "SUIUSDT": PositionState("SUIUSDT", Decimal("0"), Decimal("0"), Decimal("0")),
            "ETHUSDT": PositionState("ETHUSDT", Decimal("0"), Decimal("0"), Decimal("0")),
            "SOLUSDT": PositionState("SOLUSDT", Decimal("1"), Decimal("100"), Decimal("100")),
            "XRPUSDT": PositionState("XRPUSDT", Decimal("1"), Decimal("100"), Decimal("100")),
        }

        constrained = apply_portfolio_position_limit(
            decisions,
            position_states,
            position_limit=3,
            priority_symbols=("BTCUSDT", "ETHUSDT"),
        )

        self.assertEqual(constrained["BTCUSDT"].action, "buy")
        self.assertEqual(constrained["ETHUSDT"].action, "skip")
        self.assertEqual(constrained["ETHUSDT"].reason, "portfolio_position_limit_priority_blocked")
        self.assertEqual(constrained["SUIUSDT"].action, "skip")
        self.assertEqual(constrained["SUIUSDT"].reason, "portfolio_position_limit_priority_blocked")

    def test_portfolio_limit_does_not_block_averaging_existing_position(self) -> None:
        decisions = {
            "ETHUSDT": ExecutionDecision("buy", "ETHUSDT", Decimal("90"), Decimal("1"), Decimal("90"), "test"),
            "BTCUSDT": ExecutionDecision("buy", "BTCUSDT", Decimal("100"), Decimal("1"), Decimal("100"), "test"),
        }
        position_states = {
            "ETHUSDT": PositionState("ETHUSDT", Decimal("1"), Decimal("100"), Decimal("100"), entry_count=1),
            "BTCUSDT": PositionState("BTCUSDT", Decimal("0"), Decimal("0"), Decimal("0")),
            "SOLUSDT": PositionState("SOLUSDT", Decimal("1"), Decimal("100"), Decimal("100")),
            "SUIUSDT": PositionState("SUIUSDT", Decimal("1"), Decimal("100"), Decimal("100")),
        }

        constrained = apply_portfolio_position_limit(
            decisions,
            position_states,
            position_limit=3,
            priority_symbols=("BTCUSDT", "ETHUSDT"),
        )

        self.assertEqual(constrained["ETHUSDT"].action, "buy")
        self.assertEqual(constrained["BTCUSDT"].action, "skip")
        self.assertEqual(constrained["BTCUSDT"].reason, "portfolio_position_limit_reached")

    def test_portfolio_limit_preserves_existing_skip_reason_from_averaging_limit(self) -> None:
        decisions = {
            "ETHUSDT": ExecutionDecision("skip", "ETHUSDT", Decimal("90"), Decimal("0"), Decimal("0"), "max_entry_count_reached"),
            "BTCUSDT": ExecutionDecision("buy", "BTCUSDT", Decimal("100"), Decimal("1"), Decimal("100"), "test"),
        }
        position_states = {
            "ETHUSDT": PositionState("ETHUSDT", Decimal("1"), Decimal("100"), Decimal("100"), entry_count=3),
            "BTCUSDT": PositionState("BTCUSDT", Decimal("0"), Decimal("0"), Decimal("0")),
            "SOLUSDT": PositionState("SOLUSDT", Decimal("1"), Decimal("100"), Decimal("100")),
            "SUIUSDT": PositionState("SUIUSDT", Decimal("1"), Decimal("100"), Decimal("100")),
            "XRPUSDT": PositionState("XRPUSDT", Decimal("1"), Decimal("100"), Decimal("100")),
        }

        constrained = apply_portfolio_position_limit(
            decisions,
            position_states,
            position_limit=3,
            priority_symbols=("BTCUSDT", "ETHUSDT"),
        )

        self.assertEqual(constrained["ETHUSDT"].action, "skip")
        self.assertEqual(constrained["ETHUSDT"].reason, "max_entry_count_reached")
        self.assertEqual(constrained["BTCUSDT"].action, "skip")
        self.assertEqual(constrained["BTCUSDT"].reason, "portfolio_position_limit_reached")
