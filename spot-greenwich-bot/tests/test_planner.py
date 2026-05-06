from __future__ import annotations

import unittest
from decimal import Decimal

import pandas as pd

from domain.models import PositionState
from domain.planner import GreenwichSpotPlanner


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

        plan = planner.plan("ETHUSDT", df, state, Decimal("500"))

        self.assertEqual(plan.signal.signal_type, "buy")
        self.assertEqual(plan.decision.action, "buy")
