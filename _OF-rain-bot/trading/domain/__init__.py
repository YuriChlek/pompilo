from __future__ import annotations

__all__ = [
    "BasisDiagnostics",
    "DEFAULT_STRATEGY_CONFIG",
    "LiquidityWall",
    "PositionSnapshot",
    "SetupDiagnostics",
    "ScalpSignal",
    "SignalDiagnostics",
    "StrategyConfig",
    "SignalDirection",
    "SignalGenerationEngine",
    "SignalGenerationInputs",
    "LiquidityDetector",
    "SpoofFilter",
    "WallScanDiagnostics",
    "best_price_seen",
    "break_even_stop_price",
    "current_market_price",
    "infer_exchange_exit_reason",
    "pending_entry_invalidation_reason",
    "position_exit_reason",
    "should_fill_dry_run",
    "stop_improves",
]


def __getattr__(name: str):
    if name in {"BasisDiagnostics", "SetupDiagnostics", "SignalDiagnostics", "WallScanDiagnostics"}:
        from .diagnostics import BasisDiagnostics, SetupDiagnostics, SignalDiagnostics, WallScanDiagnostics

        mapping = {
            "BasisDiagnostics": BasisDiagnostics,
            "SetupDiagnostics": SetupDiagnostics,
            "SignalDiagnostics": SignalDiagnostics,
            "WallScanDiagnostics": WallScanDiagnostics,
        }
        return mapping[name]
    if name in {
        "LiquidityWall",
        "PositionSnapshot",
        "ScalpSignal",
        "SignalDirection",
        "to_domain_liquidity_wall",
        "to_domain_position_snapshot",
        "to_domain_scalp_signal",
        "to_domain_signal_direction",
    }:
        from .models import (
            LiquidityWall,
            PositionSnapshot,
            ScalpSignal,
            SignalDirection,
            to_domain_liquidity_wall,
            to_domain_position_snapshot,
            to_domain_scalp_signal,
            to_domain_signal_direction,
        )

        mapping = {
            "LiquidityWall": LiquidityWall,
            "PositionSnapshot": PositionSnapshot,
            "ScalpSignal": ScalpSignal,
            "SignalDirection": SignalDirection,
            "to_domain_liquidity_wall": to_domain_liquidity_wall,
            "to_domain_position_snapshot": to_domain_position_snapshot,
            "to_domain_scalp_signal": to_domain_scalp_signal,
            "to_domain_signal_direction": to_domain_signal_direction,
        }
        return mapping[name]
    if name in {"DEFAULT_STRATEGY_CONFIG", "StrategyConfig"}:
        from .strategy_config import DEFAULT_STRATEGY_CONFIG, StrategyConfig

        mapping = {
            "DEFAULT_STRATEGY_CONFIG": DEFAULT_STRATEGY_CONFIG,
            "StrategyConfig": StrategyConfig,
        }
        return mapping[name]
    if name in {"LiquidityDetector", "SignalGenerationEngine", "SignalGenerationInputs", "SpoofFilter"}:
        from .signal_generation import LiquidityDetector, SignalGenerationEngine, SignalGenerationInputs, SpoofFilter

        mapping = {
            "LiquidityDetector": LiquidityDetector,
            "SignalGenerationEngine": SignalGenerationEngine,
            "SignalGenerationInputs": SignalGenerationInputs,
            "SpoofFilter": SpoofFilter,
        }
        return mapping[name]
    if name in {
        "best_price_seen",
        "break_even_stop_price",
        "current_market_price",
        "infer_exchange_exit_reason",
        "pending_entry_invalidation_reason",
        "position_exit_reason",
        "should_fill_dry_run",
        "stop_improves",
    }:
        from .execution import (
            best_price_seen,
            break_even_stop_price,
            current_market_price,
            infer_exchange_exit_reason,
            pending_entry_invalidation_reason,
            position_exit_reason,
            should_fill_dry_run,
            stop_improves,
        )

        mapping = {
            "best_price_seen": best_price_seen,
            "break_even_stop_price": break_even_stop_price,
            "current_market_price": current_market_price,
            "infer_exchange_exit_reason": infer_exchange_exit_reason,
            "pending_entry_invalidation_reason": pending_entry_invalidation_reason,
            "position_exit_reason": position_exit_reason,
            "should_fill_dry_run": should_fill_dry_run,
            "stop_improves": stop_improves,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
