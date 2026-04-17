from __future__ import annotations

from utils.settings import build_db_connection_url as _build_db_connection_url
from utils.settings import get_app_settings, get_database_settings


_APP_SETTINGS = get_app_settings()
_DB_SETTINGS = _APP_SETTINGS.database


DB_HOST = _DB_SETTINGS.host
DB_PORT = _DB_SETTINGS.port
DB_NAME = _DB_SETTINGS.database
DB_USER = _DB_SETTINGS.user
DB_PASS = _DB_SETTINGS.password

BUY_DIRECTION = _APP_SETTINGS.buy_direction
SELL_DIRECTION = _APP_SETTINGS.sell_direction

TRADING_SYMBOLS = list(_APP_SETTINGS.trading_symbols)
AI_TRADING_SYMBOLS = list(_APP_SETTINGS.ai_trading_symbols)


def get_database_config() -> dict[str, str]:
    return get_database_settings().to_connection_kwargs()


def build_db_connection_url() -> str:
    return _build_db_connection_url()


__all__ = [
    "AI_TRADING_SYMBOLS",
    "BUY_DIRECTION",
    "DB_HOST",
    "DB_NAME",
    "DB_PASS",
    "DB_PORT",
    "DB_USER",
    "SELL_DIRECTION",
    "TRADING_SYMBOLS",
    "build_db_connection_url",
    "get_database_config",
]
