from __future__ import annotations

import unittest

import pandas as pd

from indicators.grinvich import _crossover, _crossunder, build_greenwich_snapshot


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


class GreenwichSignalTests(unittest.TestCase):
    def test_crossover_matches_pine_boundary_semantics(self) -> None:
        left = pd.Series([95.0, 96.0, 97.0])
        right = pd.Series([96.0, 96.0, 96.0])
        result = _crossover(left, right)
        self.assertEqual(result.tolist(), [False, False, True])

    def test_crossunder_matches_pine_boundary_semantics(self) -> None:
        left = pd.Series([105.0, 104.0, 103.0])
        right = pd.Series([104.0, 104.0, 104.0])
        result = _crossunder(left, right)
        self.assertEqual(result.tolist(), [False, False, True])

    def test_buy_signal_detects_recovery_above_lower3(self) -> None:
        df = _base_frame()
        df.loc[df.index[-2], "low"] = 80
        df.loc[df.index[-1], "low"] = 100
        snapshot = build_greenwich_snapshot(df)
        self.assertTrue(snapshot.buy_signal)
        self.assertFalse(snapshot.sell_signal)

    def test_sell_signal_detects_drop_below_upper2(self) -> None:
        df = _base_frame()
        snapshot_before = build_greenwich_snapshot(df)
        df.loc[df.index[-2], "close"] = float(snapshot_before.upper2) + 10
        df.loc[df.index[-1], "close"] = float(snapshot_before.upper2) - 5
        snapshot = build_greenwich_snapshot(df)
        self.assertTrue(snapshot.sell_signal)
