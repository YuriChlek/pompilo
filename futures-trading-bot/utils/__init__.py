from .config import (
    ANALYSIS_CANDLE_LIMIT,
    BUY_DIRECTION,
    BYBIT_API_ENDPOINT,
    BYBIT_API_KEY,
    BYBIT_API_SECRET,
    BYBIT_RECV_WINDOW,
    BYBIT_TAKER_FEE_RATE,
    BREAKEVEN_TRIGGER_R,
    DB_HOST,
    DB_NAME,
    DB_PASS,
    DB_PORT,
    DB_USER,
    ENABLE_BREAKEVEN_PARTIAL_CLOSE,
    ENABLE_BREAKEVEN_STOP_MANAGEMENT,
    LIMIT_ORDER_MAX_AGE_HOURS,
    LIMIT_ORDER_MAX_AGE_MS,
    POSITION_ROUNDING_RULES,
    SELL_DIRECTION,
    SYMBOLS_ROUNDING,
    TEST_TRADING_SYMBOLS,
    TRADING_SYMBOLS,
)

_LAZY_DB_EXPORTS = {"create_tables"}

__all__ = [
    "BUY_DIRECTION",
    "ANALYSIS_CANDLE_LIMIT",
    "BYBIT_API_ENDPOINT",
    "BYBIT_API_KEY",
    "BYBIT_API_SECRET",
    "BYBIT_RECV_WINDOW",
    "BYBIT_TAKER_FEE_RATE",
    "BREAKEVEN_TRIGGER_R",
    "DB_HOST",
    "DB_NAME",
    "DB_PASS",
    "DB_PORT",
    "DB_USER",
    "ENABLE_BREAKEVEN_PARTIAL_CLOSE",
    "ENABLE_BREAKEVEN_STOP_MANAGEMENT",
    "LIMIT_ORDER_MAX_AGE_HOURS",
    "LIMIT_ORDER_MAX_AGE_MS",
    "POSITION_ROUNDING_RULES",
    "SELL_DIRECTION",
    "SYMBOLS_ROUNDING",
    "TEST_TRADING_SYMBOLS",
    "TRADING_SYMBOLS",
    "create_tables",
]


def __getattr__(name: str):
    """Load database helpers lazily so config-only imports do not require DB dependencies."""
    if name in _LAZY_DB_EXPORTS:
        from . import db_actions

        value = getattr(db_actions, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
