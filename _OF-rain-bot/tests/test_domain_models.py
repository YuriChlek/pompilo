from __future__ import annotations

from decimal import Decimal
import unittest

from trading.application.runtime_models import LiquidityWall as LegacyLiquidityWall
from trading.application.runtime_models import ScalpSignal as LegacyScalpSignal
from trading.application.runtime_models import SignalDirection as LegacySignalDirection
from trading.domain.models import (
    LiquidityWall,
    PositionSnapshot,
    ScalpSignal,
    SignalDirection,
    to_domain_liquidity_wall,
    to_domain_scalp_signal,
    to_domain_signal_direction,
)


class DomainModelsTests(unittest.TestCase):
    def test_to_domain_signal_direction_normalizes_legacy_enum(self) -> None:
        self.assertEqual(to_domain_signal_direction(LegacySignalDirection.LONG), SignalDirection.LONG)
        self.assertEqual(to_domain_signal_direction("short"), SignalDirection.SHORT)
        self.assertEqual(to_domain_signal_direction("unexpected"), SignalDirection.NONE)

    def test_to_domain_liquidity_wall_uses_decimal_fields(self) -> None:
        legacy = LegacyLiquidityWall(
            exchange="bybit",
            symbol="BTCUSDT",
            side="bid",
            price=100.1,
            size=5.0,
            notional=500.5,
            distance_ticks=1,
            distance_bps=1.2,
            first_seen_ms=1,
            last_seen_ms=2,
            persistence_ms=1,
            relative_size_ratio=4.5,
            size_stability_score=90.0,
            pull_count=0,
            test_count=2,
            reload_count=1,
            defended_count=2,
            chase_count=0,
            score=80.0,
            spoof_risk_score=10.0,
            metadata={"baseline_size": 1.0},
        )

        wall = to_domain_liquidity_wall(legacy)

        self.assertIsInstance(wall, LiquidityWall)
        assert wall is not None
        self.assertEqual(wall.price, Decimal("100.1"))
        self.assertEqual(wall.score, Decimal("80.0"))

    def test_to_domain_scalp_signal_uses_decimal_fields(self) -> None:
        legacy = LegacyScalpSignal(
            symbol="BTCUSDT",
            direction=LegacySignalDirection.LONG,
            wall=None,
            confidence=87.5,
            reason="defended_bid_wall",
            analysis_entry_price=100.1,
            analysis_stop_price=99.9,
            analysis_take_profit_price=100.5,
            analysis_invalidation_price=99.8,
        )

        signal = to_domain_scalp_signal(legacy)

        self.assertIsInstance(signal, ScalpSignal)
        self.assertEqual(signal.direction, SignalDirection.LONG)
        self.assertEqual(signal.confidence, Decimal("87.5"))
        self.assertEqual(signal.analysis_entry_price, Decimal("100.1"))
