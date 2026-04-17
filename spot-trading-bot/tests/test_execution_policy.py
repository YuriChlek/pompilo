from __future__ import annotations

import unittest
from decimal import Decimal

from trading.domain.execution import decide_spot_execution
from trading.domain.models import PositionState, SpotSignal


class ExecutionPolicyTests(unittest.TestCase):
    def test_first_buy_is_allowed_without_existing_position(self) -> None:
        signal = SpotSignal("ETHUSDT", "buy", Decimal("100"), "2026-01-01", "test")
        state = PositionState("ETHUSDT", Decimal("0"), Decimal("0"), Decimal("0"))
        decision = decide_spot_execution(signal, state)
        self.assertEqual(decision.action, "buy")

    def test_repeat_buy_requires_better_price_than_average_entry(self) -> None:
        signal = SpotSignal("ETHUSDT", "buy", Decimal("101"), "2026-01-01", "test")
        state = PositionState("ETHUSDT", Decimal("1"), Decimal("100"), Decimal("100"))
        decision = decide_spot_execution(signal, state)
        self.assertEqual(decision.action, "skip")
        self.assertEqual(decision.reason, "buy_price_not_better_than_avg_entry")

    def test_sell_requires_profitable_price(self) -> None:
        signal = SpotSignal("ETHUSDT", "sell", Decimal("99"), "2026-01-01", "test")
        state = PositionState("ETHUSDT", Decimal("1"), Decimal("100"), Decimal("100"))
        decision = decide_spot_execution(signal, state)
        self.assertEqual(decision.action, "skip")
        self.assertEqual(decision.reason, "sell_price_not_profitable")

    def test_sell_uses_full_position_when_profitable(self) -> None:
        signal = SpotSignal("ETHUSDT", "sell", Decimal("110"), "2026-01-01", "test")
        state = PositionState("ETHUSDT", Decimal("1.25"), Decimal("100"), Decimal("125"))
        decision = decide_spot_execution(signal, state)
        self.assertEqual(decision.action, "sell")
        self.assertEqual(decision.quantity, Decimal("1.25"))
