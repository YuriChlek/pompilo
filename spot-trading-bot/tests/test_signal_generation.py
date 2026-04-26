from __future__ import annotations

import unittest
from decimal import Decimal
from unittest.mock import patch

from domain.signals import generate_spot_signal


class SignalGenerationTests(unittest.TestCase):
    def test_generate_spot_signal_returns_buy_when_snapshot_has_buy_signal(self) -> None:
        snapshot = type(
            "Snapshot",
            (),
            {
                "buy_signal": True,
                "sell_signal": False,
                "signal_price": Decimal("100"),
                "close_time": "2026-01-01",
            },
        )()

        with patch("domain.signals.build_greenwich_signal_snapshot", return_value=snapshot):
            signal = generate_spot_signal("ETHUSDT", candles_df=object())

        self.assertEqual(signal.signal_type, "buy")
        self.assertEqual(signal.reason, "greenwich_buy_recovery")

    def test_generate_spot_signal_returns_sell_when_snapshot_has_sell_signal(self) -> None:
        snapshot = type(
            "Snapshot",
            (),
            {
                "buy_signal": False,
                "sell_signal": True,
                "signal_price": Decimal("95"),
                "close_time": "2026-01-01",
            },
        )()

        with patch("domain.signals.build_greenwich_signal_snapshot", return_value=snapshot):
            signal = generate_spot_signal("ETHUSDT", candles_df=object())

        self.assertEqual(signal.signal_type, "sell")
        self.assertEqual(signal.reason, "greenwich_sell_fade")

    def test_generate_spot_signal_returns_hold_when_snapshot_has_no_entry_or_exit(self) -> None:
        snapshot = type(
            "Snapshot",
            (),
            {
                "buy_signal": False,
                "sell_signal": False,
                "signal_price": Decimal("98"),
                "close_time": "2026-01-01",
            },
        )()

        with patch("domain.signals.build_greenwich_signal_snapshot", return_value=snapshot):
            signal = generate_spot_signal("ETHUSDT", candles_df=object())

        self.assertEqual(signal.signal_type, "hold")
        self.assertEqual(signal.reason, "no_greenwich_signal")
