from __future__ import annotations

import unittest
from decimal import Decimal
from unittest.mock import patch

import pandas as pd

from domain.models import PositionState
from domain.planner import GreenwichSpotPlanner, _resolve_atr_size_multiplier


def _base_frame() -> pd.DataFrame:
    rows = []
    for index in range(120):
        rows.append(
            {
                "open_time": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
                "close_time": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index, hours=23),
                "symbol": "ETHUSDT",
                "open": 100 + index * 0.1,
                "high": 102 + index * 0.1,
                "low": 98 + index * 0.1,
                "close": 100 + index * 0.1,
                "volume": 1000 + index,
                "cvd": index,
            }
        )
    return pd.DataFrame(rows)


class GreenwichSpotPlannerTests(unittest.TestCase):
    def test_planner_returns_buy_signal_and_buy_decision(self) -> None:
        planner = GreenwichSpotPlanner()
        df = _base_frame()
        df.loc[df.index[-2], "low"] = 80
        df.loc[df.index[-1], "low"] = 100
        state = PositionState("ETHUSDT", Decimal("0"), Decimal("0"), Decimal("0"))

        with patch("domain.planner.CONFIRMATION_CANDLE_ENABLED", False):
            plan = planner.plan("ETHUSDT", df, state, Decimal("500"))

        self.assertEqual(plan.signal.signal_type, "buy")
        self.assertEqual(plan.decision.action, "buy")

    def test_planner_blocks_buy_after_sharp_drop(self) -> None:
        planner = GreenwichSpotPlanner()
        df = _base_frame()
        df.loc[df.index[-4], "close"] = 120
        df.loc[df.index[-3], "close"] = 115
        df.loc[df.index[-2], "close"] = 110
        df.loc[df.index[-1], "close"] = 100
        state = PositionState("ETHUSDT", Decimal("0"), Decimal("0"), Decimal("0"))

        with patch("domain.planner.generate_spot_signal", return_value=type("Signal", (), {
            "symbol": "ETHUSDT",
            "signal_type": "buy",
            "signal_price": Decimal("100"),
            "close_time": "2026-01-01",
            "reason": "mock_buy",
            "timeframe": "d1",
            "candle_id": "2026-01-01",
        })()):
            with patch("domain.planner.CONFIRMATION_CANDLE_ENABLED", False):
                plan = planner.plan("ETHUSDT", df, state, Decimal("500"))

        self.assertEqual(plan.signal.signal_type, "hold")
        self.assertEqual(plan.signal.reason, "buy_anti_crash_blocked")
        self.assertEqual(plan.decision.action, "skip")

    def test_planner_waits_for_confirmation_on_initial_buy_signal(self) -> None:
        planner = GreenwichSpotPlanner()
        df = _base_frame()
        state = PositionState("ETHUSDT", Decimal("0"), Decimal("0"), Decimal("0"))
        current_buy = type("Signal", (), {
            "symbol": "ETHUSDT",
            "signal_type": "buy",
            "signal_price": Decimal("100"),
            "close_time": "2026-01-02",
            "reason": "greenwich_buy_recovery",
            "timeframe": "d1",
            "candle_id": "2026-01-02",
        })()
        previous_hold = type("Signal", (), {
            "symbol": "ETHUSDT",
            "signal_type": "hold",
            "signal_price": Decimal("99"),
            "close_time": "2026-01-01",
            "reason": "no_greenwich_signal",
            "timeframe": "d1",
            "candle_id": "2026-01-01",
        })()

        with patch("domain.planner.generate_spot_signal", side_effect=[current_buy, previous_hold]):
            plan = planner.plan("ETHUSDT", df, state, Decimal("500"))

        self.assertEqual(plan.signal.signal_type, "hold")
        self.assertEqual(plan.signal.reason, "buy_waiting_confirmation")
        self.assertEqual(plan.decision.action, "skip")

    def test_planner_confirms_previous_buy_on_next_candle(self) -> None:
        planner = GreenwichSpotPlanner()
        df = _base_frame()
        df.loc[df.index[-1], "close"] = 111
        state = PositionState("ETHUSDT", Decimal("0"), Decimal("0"), Decimal("0"))
        current_hold = type("Signal", (), {
            "symbol": "ETHUSDT",
            "signal_type": "hold",
            "signal_price": Decimal("111"),
            "close_time": "2026-01-02",
            "reason": "no_greenwich_signal",
            "timeframe": "d1",
            "candle_id": "2026-01-02",
        })()
        previous_buy = type("Signal", (), {
            "symbol": "ETHUSDT",
            "signal_type": "buy",
            "signal_price": Decimal("100"),
            "close_time": "2026-01-01",
            "reason": "greenwich_buy_recovery",
            "timeframe": "d1",
            "candle_id": "2026-01-01",
        })()
        snapshot = type("Snapshot", (), {"lower3": Decimal("100"), "close_time": "2026-01-02"})()

        with patch("domain.planner.generate_spot_signal", side_effect=[current_hold, previous_buy]):
            with patch("domain.planner.build_greenwich_signal_snapshot", return_value=snapshot):
                with patch("domain.planner.ANTI_CRASH_BUY_BLOCK_ENABLED", False):
                    plan = planner.plan("ETHUSDT", df, state, Decimal("500"))

        self.assertEqual(plan.signal.signal_type, "buy")
        self.assertEqual(plan.signal.reason, "greenwich_buy_confirmation")
        self.assertEqual(plan.decision.action, "buy")

    def test_planner_skips_buy_when_confirmation_fails(self) -> None:
        planner = GreenwichSpotPlanner()
        df = _base_frame()
        df.loc[df.index[-1], "close"] = 111
        state = PositionState("ETHUSDT", Decimal("0"), Decimal("0"), Decimal("0"))
        current_hold = type("Signal", (), {
            "symbol": "ETHUSDT",
            "signal_type": "hold",
            "signal_price": Decimal("111"),
            "close_time": "2026-01-02",
            "reason": "no_greenwich_signal",
            "timeframe": "d1",
            "candle_id": "2026-01-02",
        })()
        previous_buy = type("Signal", (), {
            "symbol": "ETHUSDT",
            "signal_type": "buy",
            "signal_price": Decimal("100"),
            "close_time": "2026-01-01",
            "reason": "greenwich_buy_recovery",
            "timeframe": "d1",
            "candle_id": "2026-01-01",
        })()
        snapshot = type("Snapshot", (), {"lower3": Decimal("120"), "close_time": "2026-01-02"})()

        with patch("domain.planner.generate_spot_signal", side_effect=[current_hold, previous_buy]):
            with patch("domain.planner.build_greenwich_signal_snapshot", return_value=snapshot):
                with patch("domain.planner.ANTI_CRASH_BUY_BLOCK_ENABLED", False):
                    plan = planner.plan("ETHUSDT", df, state, Decimal("500"))

        self.assertEqual(plan.signal.signal_type, "hold")
        self.assertEqual(plan.signal.reason, "buy_confirmation_failed")
        self.assertEqual(plan.decision.action, "skip")

    def test_atr_size_multiplier_decreases_in_high_volatility(self) -> None:
        df = _base_frame()
        df.loc[df.index[-60]:, "high"] = df.loc[df.index[-60]:, "close"] + 25
        df.loc[df.index[-60]:, "low"] = df.loc[df.index[-60]:, "close"] - 25

        multiplier = _resolve_atr_size_multiplier(df)

        self.assertLess(multiplier, Decimal("1"))
        self.assertGreaterEqual(multiplier, Decimal("0.5"))

    def test_atr_size_multiplier_increases_in_low_volatility(self) -> None:
        df = _base_frame()
        df.loc[df.index[-60]:, "high"] = df.loc[df.index[-60]:, "close"] + 0.5
        df.loc[df.index[-60]:, "low"] = df.loc[df.index[-60]:, "close"] - 0.5

        multiplier = _resolve_atr_size_multiplier(df)

        self.assertGreater(multiplier, Decimal("1"))
        self.assertLessEqual(multiplier, Decimal("1.5"))
