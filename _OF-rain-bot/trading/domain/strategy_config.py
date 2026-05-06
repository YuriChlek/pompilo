from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class LiquidityDetectionConfig:
    max_chase_ticks: int
    max_wall_distance_bps: Decimal
    max_wall_distance_ticks: int
    max_wall_size_drop_pct: Decimal
    min_defended_ratio: Decimal
    min_wall_notional_usdt: Decimal
    min_wall_persist_ms: int
    min_wall_relative_size: Decimal
    test_debounce_ms: int
    test_touch_ticks: int
    min_wall_score: Decimal


@dataclass(frozen=True)
class SpoofFilterConfig:
    max_chase_ticks: int
    max_pull_events: int
    max_spoof_score: Decimal
    min_defended_ratio: Decimal


@dataclass(frozen=True)
class SignalGenerationConfig:
    analysis_reference_exchange: str
    book_stale_ms: int
    cross_confirmation_bps: Decimal
    entry_offset_ticks: int
    stop_loss_size: Decimal
    invalidation_offset_ticks: int
    max_spread_ticks: int
    min_cross_exchange_confirmations: int
    min_defended_ratio: Decimal
    min_rejection_ticks: int
    min_tape_pressure_ratio: Decimal
    min_test_count: int
    min_wall_score: Decimal
    stop_offset_ticks: int
    take_profit_r_multiple: Decimal
    tape_window_ms: int
    test_touch_ticks: int


@dataclass(frozen=True)
class StrategyConfig:
    liquidity_detection: LiquidityDetectionConfig
    spoof_filter: SpoofFilterConfig
    signal_generation: SignalGenerationConfig


DEFAULT_STRATEGY_CONFIG = StrategyConfig(
    liquidity_detection=LiquidityDetectionConfig(
        max_chase_ticks=2,
        max_wall_distance_bps=Decimal("8"),
        max_wall_distance_ticks=8,
        max_wall_size_drop_pct=Decimal("0.45"),
        min_defended_ratio=Decimal("0.55"),
        min_wall_notional_usdt=Decimal("250000"),
        min_wall_persist_ms=800,
        min_wall_relative_size=Decimal("4.0"),
        test_debounce_ms=500,
        test_touch_ticks=2,
        min_wall_score=Decimal("65"),
    ),
    spoof_filter=SpoofFilterConfig(
        max_chase_ticks=2,
        max_pull_events=3,
        max_spoof_score=Decimal("45"),
        min_defended_ratio=Decimal("0.55"),
    ),
    signal_generation=SignalGenerationConfig(
        analysis_reference_exchange="bybit",
        book_stale_ms=2500,
        cross_confirmation_bps=Decimal("6"),
        entry_offset_ticks=1,
        stop_loss_size=Decimal("0.5"),
        invalidation_offset_ticks=1,
        max_spread_ticks=3,
        min_cross_exchange_confirmations=1,
        min_defended_ratio=Decimal("0.55"),
        min_rejection_ticks=1,
        min_tape_pressure_ratio=Decimal("1.15"),
        min_test_count=2,
        min_wall_score=Decimal("65"),
        stop_offset_ticks=1,
        take_profit_r_multiple=Decimal("1.5"),
        tape_window_ms=3000,
        test_touch_ticks=2,
    ),
)
