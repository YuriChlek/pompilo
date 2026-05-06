from __future__ import annotations

from decimal import Decimal
import unittest

from trading.domain.models import SignalDirection
from trading.domain.execution import (
    best_price_seen,
    break_even_stop_price,
    current_market_price,
    infer_exchange_exit_reason,
    pending_entry_invalidation_reason,
    position_exit_reason,
    should_fill_dry_run,
    stop_improves,
)


class DomainExecutionTests(unittest.TestCase):
    def test_should_fill_dry_run_for_long(self) -> None:
        self.assertTrue(should_fill_dry_run(SignalDirection.LONG, Decimal("100.0"), Decimal("99.9"), Decimal("99.95")))
        self.assertFalse(should_fill_dry_run(SignalDirection.LONG, Decimal("100.0"), Decimal("99.9"), Decimal("100.1")))

    def test_should_fill_dry_run_for_short(self) -> None:
        self.assertTrue(should_fill_dry_run(SignalDirection.SHORT, Decimal("100.0"), Decimal("100.05"), Decimal("100.1")))
        self.assertFalse(should_fill_dry_run(SignalDirection.SHORT, Decimal("100.0"), Decimal("99.95"), Decimal("100.1")))

    def test_pending_invalidation_prefers_signal_reversal(self) -> None:
        reason = pending_entry_invalidation_reason(
            current_signal_reason="defended_ask_wall",
            current_signal_direction=SignalDirection.SHORT,
            pending_signal_direction=SignalDirection.LONG,
            has_pending_wall=True,
            wall_is_active=True,
            has_reference_book=True,
        )
        self.assertEqual(reason, "signal_reversed")

    def test_pending_invalidation_ignores_stale_analysis(self) -> None:
        reason = pending_entry_invalidation_reason(
            current_signal_reason="stale_analysis_book",
            current_signal_direction=SignalDirection.NONE,
            pending_signal_direction=SignalDirection.LONG,
            has_pending_wall=True,
            wall_is_active=False,
            has_reference_book=True,
        )
        self.assertIsNone(reason)

    def test_pending_invalidation_detects_wall_disappeared(self) -> None:
        reason = pending_entry_invalidation_reason(
            current_signal_reason="setup_not_confirmed",
            current_signal_direction=SignalDirection.NONE,
            pending_signal_direction=SignalDirection.LONG,
            has_pending_wall=True,
            wall_is_active=False,
            has_reference_book=True,
        )
        self.assertEqual(reason, "wall_disappeared")

    def test_current_market_price_uses_side_specific_quote(self) -> None:
        self.assertEqual(current_market_price("Buy", Decimal("100.0"), Decimal("100.1"), Decimal("99.0")), Decimal("100.0"))
        self.assertEqual(current_market_price("Sell", Decimal("100.0"), Decimal("100.1"), Decimal("99.0")), Decimal("100.1"))
        self.assertEqual(current_market_price("Buy", None, None, Decimal("99.0")), Decimal("99.0"))

    def test_best_price_seen_tracks_favorable_direction(self) -> None:
        self.assertEqual(best_price_seen("Buy", Decimal("100.0"), Decimal("101.0")), Decimal("101.0"))
        self.assertEqual(best_price_seen("Buy", Decimal("100.0"), Decimal("99.5")), Decimal("100.0"))
        self.assertEqual(best_price_seen("Sell", Decimal("100.0"), Decimal("99.0")), Decimal("99.0"))
        self.assertEqual(best_price_seen("Sell", Decimal("100.0"), Decimal("101.0")), Decimal("100.0"))

    def test_break_even_stop_price_for_long_and_short(self) -> None:
        self.assertEqual(
            break_even_stop_price("Buy", Decimal("100.0"), Decimal("100.4"), Decimal("0.1"), arm_ticks=3, buffer_ticks=1),
            Decimal("100.10000000"),
        )
        self.assertEqual(
            break_even_stop_price("Sell", Decimal("100.0"), Decimal("99.6"), Decimal("0.1"), arm_ticks=3, buffer_ticks=1),
            Decimal("99.90000000"),
        )
        self.assertIsNone(
            break_even_stop_price("Buy", Decimal("100.0"), Decimal("100.2"), Decimal("0.1"), arm_ticks=3, buffer_ticks=1)
        )

    def test_stop_improves_respects_side(self) -> None:
        self.assertTrue(stop_improves("Buy", Decimal("99.9"), Decimal("100.0")))
        self.assertFalse(stop_improves("Buy", Decimal("100.0"), Decimal("99.9")))
        self.assertTrue(stop_improves("Sell", Decimal("100.1"), Decimal("100.0")))
        self.assertFalse(stop_improves("Sell", Decimal("100.0"), Decimal("100.1")))

    def test_position_exit_reason_handles_tp_sl_and_reversal(self) -> None:
        self.assertEqual(
            position_exit_reason("Buy", Decimal("99.0"), Decimal("101.0"), SignalDirection.LONG, Decimal("101.0")),
            "take_profit",
        )
        self.assertEqual(
            position_exit_reason("Sell", Decimal("101.0"), Decimal("99.0"), SignalDirection.SHORT, Decimal("101.0")),
            "stop_loss",
        )
        self.assertEqual(
            position_exit_reason("Buy", Decimal("99.0"), Decimal("101.0"), SignalDirection.SHORT, Decimal("100.0")),
            "signal_reversal",
        )

    def test_infer_exchange_exit_reason_uses_price_vs_targets(self) -> None:
        self.assertEqual(infer_exchange_exit_reason("Buy", Decimal("99.0"), Decimal("101.0"), Decimal("101.5")), "take_profit")
        self.assertEqual(infer_exchange_exit_reason("Sell", Decimal("101.0"), Decimal("99.0"), Decimal("98.5")), "take_profit")
        self.assertEqual(
            infer_exchange_exit_reason("Buy", Decimal("99.0"), Decimal("101.0"), Decimal("100.2")),
            "position_closed_on_exchange",
        )
