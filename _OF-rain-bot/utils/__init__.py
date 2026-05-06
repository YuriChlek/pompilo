from __future__ import annotations

from .config import *

__all__ = [
    "APP_CONFIG",
    "AppConfig",
    "BYBIT_API_KEY",
    "BYBIT_API_SECRET",
    "BYBIT_MARKET_DATA_API_ENDPOINT",
    "BYBIT_MARKET_DATA_WS_ENDPOINT",
    "BYBIT_RECV_WINDOW",
    "BYBIT_TRADING_API_ENDPOINT",
    "BybitConfig",
    "DB_HOST",
    "DB_NAME",
    "DB_PASS",
    "DB_PORT",
    "DB_USER",
    "DatabaseConfig",
    "SYMBOLS_ROUNDING",
    "ORDERFLOW_TICK_SIZES",
    "OrderFlowRuntimeConfig",
    "POSITION_ROUNDING_RULES",
    "ORDERFLOW_ANALYSIS_REFERENCE_EXCHANGE",
    "ORDERFLOW_BREAKEVEN_ARM_TICKS",
    "ORDERFLOW_BREAKEVEN_BUFFER_TICKS",
    "ORDERFLOW_BOOK_HISTORY_SIZE",
    "ORDERFLOW_INVALIDATION_OFFSET_TICKS",
    "ORDERFLOW_OBSERVATION_EXCHANGES",
    "ORDERFLOW_PENDING_WALL_TOLERANCE_TICKS",
    "ORDERFLOW_SYMBOLS",
    "ORDERFLOW_STOP_OFFSET_TICKS",
    "ORDERFLOW_TAKE_PROFIT_R_MULTIPLE",
    "TELEGRAM_TOKEN",
    "TELEGRAM_CHAT_ID",
    "APP_TIMEZONE",
    "TelegramConfig",
    "build_app_config",
    "close_db_pool",
    "get_db_pool",
    "execute_sql_batch",
]


def __getattr__(name: str):
    if name in {"close_db_pool", "get_db_pool", "execute_sql_batch"}:
        from .db import close_db_pool, execute_sql_batch, get_db_pool

        mapping = {
            "close_db_pool": close_db_pool,
            "get_db_pool": get_db_pool,
            "execute_sql_batch": execute_sql_batch,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
