__all__ = [
    "DEFAULT_STRATEGY_CONFIG",
    "StopLossUpdate",
    "StrategyConfig",
    "TradeSignal",
    "build_breakeven_update",
    "filter_active_positions",
    "generate_strategy_signal",
    "resolve_order_quantity",
]


def __getattr__(name):
    """Lazily expose domain symbols to keep package import safe during test discovery."""
    if name in {"TradeSignal", "StopLossUpdate"}:
        from .models import StopLossUpdate, TradeSignal

        mapping = {
            "TradeSignal": TradeSignal,
            "StopLossUpdate": StopLossUpdate,
        }
        return mapping[name]

    if name in {"DEFAULT_STRATEGY_CONFIG", "StrategyConfig"}:
        from .strategy_config import DEFAULT_STRATEGY_CONFIG, StrategyConfig

        mapping = {
            "DEFAULT_STRATEGY_CONFIG": DEFAULT_STRATEGY_CONFIG,
            "StrategyConfig": StrategyConfig,
        }
        return mapping[name]

    if name in {"build_breakeven_update", "filter_active_positions", "resolve_order_quantity"}:
        from .execution import build_breakeven_update, filter_active_positions, resolve_order_quantity

        mapping = {
            "build_breakeven_update": build_breakeven_update,
            "filter_active_positions": filter_active_positions,
            "resolve_order_quantity": resolve_order_quantity,
        }
        return mapping[name]

    if name == "generate_strategy_signal":
        from .signals import generate_strategy_signal

        return generate_strategy_signal

    raise AttributeError(name)
