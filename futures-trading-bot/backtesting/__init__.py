from .models import (
    BacktestConfig,
    BacktestStats,
    BacktestValidationReport,
    PortfolioBacktestResult,
    SymbolBacktestResult,
    Trade,
)

__all__ = [
    "BacktestConfig",
    "BacktestStats",
    "BacktestValidationReport",
    "PortfolioBacktestResult",
    "SymbolBacktestResult",
    "Trade",
    "BacktestRunner",
    "run_v2_entry_validation",
]


def __getattr__(name):
    """Lazily import ``BacktestRunner`` to avoid heavyweight startup imports."""
    if name == "BacktestRunner":
        from .runner import BacktestRunner

        return BacktestRunner
    if name == "run_v2_entry_validation":
        from .validation import run_v2_entry_validation

        return run_v2_entry_validation
    raise AttributeError(name)
