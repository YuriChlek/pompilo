from __future__ import annotations

import unittest
from decimal import Decimal
from unittest.mock import patch

import pandas as pd

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
            with patch("domain.planner.CONFIRMATION_CANDLE_ENABLED", False):
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

    def test_upper1_take_profit_generates_partial_sell_when_position_is_open(self) -> None:
        snapshot = type(
            "Snapshot",
            (),
            {
                "signal_price": Decimal("120"),
                "signal_high": Decimal("120"),
                "upper1": Decimal("110"),
                "close_time": "2026-01-01",
            },
        )()
        state = PositionState("ETHUSDT", Decimal("2"), Decimal("100"), Decimal("200"))

        with patch("domain.planner.build_greenwich_signal_snapshot", return_value=snapshot):
            with patch("domain.planner.generate_spot_signal", return_value=SpotSignal("ETHUSDT", "hold", Decimal("120"), "2026-01-01", "h4_hold", "h4", "2026-01-01")):
                with patch("domain.planner.CONFIRMATION_CANDLE_ENABLED", False):
                    plan = MultiTimeframeSpotPlanner(d1_regime_filter_enabled=False).plan("ETHUSDT", {"d1": object(), "h4": object()}, state, Decimal("500"))

        self.assertEqual(plan.signal.reason, "greenwich_take_profit_upper1")
        self.assertEqual(plan.decision.action, "sell")
        self.assertEqual(plan.decision.quantity, Decimal("1.00000000"))

    def test_upper1_take_profit_is_not_repeated_after_first_take_profit(self) -> None:
        snapshot = type(
            "Snapshot",
            (),
            {
                "signal_price": Decimal("120"),
                "signal_high": Decimal("120"),
                "upper1": Decimal("110"),
                "close_time": "2026-01-01",
            },
        )()
        state = PositionState("ETHUSDT", Decimal("2"), Decimal("100"), Decimal("200"), first_take_profit_done=True)

        with patch("domain.planner.build_greenwich_signal_snapshot", return_value=snapshot):
            with patch("domain.planner.generate_spot_signal", return_value=SpotSignal("ETHUSDT", "hold", Decimal("120"), "2026-01-01", "h4_hold", "h4", "2026-01-01")):
                with patch("domain.planner.CONFIRMATION_CANDLE_ENABLED", False):
                    plan = MultiTimeframeSpotPlanner(d1_regime_filter_enabled=False).plan("ETHUSDT", {"d1": object(), "h4": object()}, state, Decimal("500"))

        self.assertEqual(plan.signal.signal_type, "hold")
        self.assertEqual(plan.decision.action, "skip")

    def test_h4_previous_buy_is_confirmed_on_next_candle(self) -> None:
        planner = MultiTimeframeSpotPlanner(d1_regime_filter_enabled=False)
        state = PositionState("ETHUSDT", Decimal("0"), Decimal("0"), Decimal("0"))
        h4 = pd.DataFrame({"close": [100] * 119 + [111], "volume": [1000] * 120})
        current_hold = SpotSignal("ETHUSDT", "hold", Decimal("111"), "2026-01-02", "no_greenwich_signal", "h4", "2026-01-02")
        previous_buy = SpotSignal("ETHUSDT", "buy", Decimal("100"), "2026-01-01", "greenwich_buy_recovery", "h4", "2026-01-01")
        snapshot = type("Snapshot", (), {"lower3": Decimal("100"), "close_time": "2026-01-02"})()

        with patch("domain.planner.generate_spot_signal", side_effect=[current_hold, previous_buy]):
            with patch("domain.planner.build_greenwich_signal_snapshot", return_value=snapshot):
                with patch("domain.planner.ANTI_CRASH_BUY_BLOCK_ENABLED", False):
                    plan = planner.plan("ETHUSDT", {"d1": object(), "h4": h4}, state, Decimal("500"))

        self.assertEqual(plan.signal.signal_type, "buy")
        self.assertEqual(plan.signal.reason, "greenwich_buy_confirmation")
        self.assertEqual(plan.decision.action, "buy")
