"""Application layer.

Target home for orchestration services, ports, schedulers, and bootstrap
composition.
"""

__all__ = [
    "build_execution_port",
    "build_strategy_config",
    "build_trading_runtime",
    "ExecutionPort",
    "TradingScheduler",
    "TradingService",
]


def __getattr__(name: str):
    if name in {"build_execution_port", "build_strategy_config", "build_trading_runtime"}:
        from .bootstrap import build_execution_port, build_strategy_config, build_trading_runtime

        mapping = {
            "build_execution_port": build_execution_port,
            "build_strategy_config": build_strategy_config,
            "build_trading_runtime": build_trading_runtime,
        }
        return mapping[name]
    if name == "ExecutionPort":
        from .ports import ExecutionPort

        return ExecutionPort
    if name == "TradingScheduler":
        from .scheduler import TradingScheduler

        return TradingScheduler
    if name == "TradingService":
        from .services import TradingService

        return TradingService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
