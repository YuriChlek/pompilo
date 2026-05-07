from __future__ import annotations

import unittest
from decimal import Decimal
from unittest.mock import patch

from domain.models import PositionState, SpotSignal
from domain.planner import MultiTimeframeSpotPlanner


class MultiTimeframeSpotPlannerTests(unittest.TestCase):
    def _plan_with(self, *, d1_blocked: bool, h4_signal: SpotSignal):
        class _Planner(MultiTimeframeSpotPlanner):
            def _is_d1_buy_blocked(self, d1_candles_df) -> bool:
                return d1_blocked

        planner = _Planner()
        state = PositionState("ETHUSDT", Decimal("0"), Decimal("0"), Decimal("0"))
        candles = {"d1": object(), "h4": object()}
        with patch("domain.planner.generate_spot_signal", return_value=h4_signal):
            return planner.plan("ETHUSDT", candles, state, Decimal("500"))

    def test_d1_blocked_converts_h4_buy_to_hold(self) -> None:
        h4_signal = SpotSignal("ETHUSDT", "buy", Decimal("100"), "2026-01-01", "h4_buy")

        plan = self._plan_with(d1_blocked=True, h4_signal=h4_signal)

        self.assertEqual(plan.signal.signal_type, "hold")
        self.assertEqual(plan.signal.reason, "d1_regime_blocks_h4_buy")
        self.assertEqual(plan.decision.action, "skip")

    def test_d1_allowed_keeps_h4_buy(self) -> None:
        h4_signal = SpotSignal("ETHUSDT", "buy", Decimal("100"), "2026-01-01", "h4_buy")

        plan = self._plan_with(d1_blocked=False, h4_signal=h4_signal)

        self.assertEqual(plan.signal.signal_type, "buy")
        self.assertEqual(plan.decision.action, "buy")

    def test_d1_blocked_keeps_h4_sell(self) -> None:
        h4_signal = SpotSignal("ETHUSDT", "sell", Decimal("110"), "2026-01-01", "h4_sell")
        state = PositionState("ETHUSDT", Decimal("1"), Decimal("100"), Decimal("100"))

        class _Planner(MultiTimeframeSpotPlanner):
            def _is_d1_buy_blocked(self, d1_candles_df) -> bool:
                return True

        with patch("domain.planner.generate_spot_signal", return_value=h4_signal):
            plan = _Planner().plan("ETHUSDT", {"d1": object(), "h4": object()}, state, Decimal("500"))

        self.assertEqual(plan.signal.signal_type, "sell")
        self.assertEqual(plan.decision.action, "sell")

    def test_d1_allowed_keeps_h4_hold(self) -> None:
        h4_signal = SpotSignal("ETHUSDT", "hold", Decimal("100"), "2026-01-01", "h4_hold")

        plan = self._plan_with(d1_blocked=False, h4_signal=h4_signal)

        self.assertEqual(plan.signal.signal_type, "hold")
        self.assertEqual(plan.decision.action, "skip")
