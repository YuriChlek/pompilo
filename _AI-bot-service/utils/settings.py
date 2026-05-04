from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from utils.env import load_project_env


load_project_env()


DEFAULT_TRADING_SYMBOLS: tuple[str, ...] = (
    "AAVEUSDT",
    "ADAUSDT",
    "ARBUSDT",
    "APTUSDT",
    "AVAXUSDT",
    "DOGEUSDT",
    "DOTUSDT",
    "ENAUSDT",
    "ETHUSDT",
    "JUPUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "NEARUSDT",
    "PENGUUSDT",
    "SOLUSDT",
    "SUIUSDT",
    "TAOUSDT",
    "TIAUSDT",
    "TONUSDT",
    "UNIUSDT",
    "VIRTUALUSDT",
    "WIFUSDT",
    "WLDUSDT",
    "XRPUSDT",
    "ZECUSDT",
)
"""
DEFAULT_TRADING_SYMBOLS: tuple[str, ...] = (
    "ETHUSDT",
    "XRPUSDT",
    "SOLUSDT",
)
"""

DEFAULT_AI_TRADING_SYMBOLS: tuple[str, ...] = (
    "ETHUSDT",
    "XRPUSDT",
    "SOLUSDT",
)


def _env_str(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _resolved_db_password() -> str | None:
    return _env_str("DB_PASS") or _env_str("DB_PASSWORD")


@dataclass(frozen=True)
class DatabaseSettings:
    host: str | None
    port: str | None
    user: str | None
    password: str | None
    database: str | None

    def missing_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.host:
            missing.append("DB_HOST")
        if not self.port:
            missing.append("DB_PORT")
        if not self.user:
            missing.append("DB_USER")
        if not self.database:
            missing.append("DATABASE")
        if not self.password:
            missing.append("DB_PASS or DB_PASSWORD")
        return missing

    def require_complete(self) -> "DatabaseSettings":
        missing = self.missing_fields()
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"Missing DB config: {missing_str}. Ensure .env file is configured.")
        return self

    def to_connection_kwargs(self) -> dict[str, str]:
        settings = self.require_complete()
        return {
            "host": str(settings.host),
            "port": str(settings.port),
            "user": str(settings.user),
            "password": str(settings.password),
            "database": str(settings.database),
        }


@dataclass(frozen=True)
class AppSettings:
    database: DatabaseSettings
    trading_symbols: tuple[str, ...]
    ai_trading_symbols: tuple[str, ...]
    buy_direction: str = "buy"
    sell_direction: str = "sell"


def load_database_settings() -> DatabaseSettings:
    return DatabaseSettings(
        host=_env_str("DB_HOST"),
        port=_env_str("DB_PORT"),
        user=_env_str("DB_USER"),
        password=_resolved_db_password(),
        database=_env_str("DATABASE"),
    )


@lru_cache(maxsize=1)
def get_database_settings() -> DatabaseSettings:
    return load_database_settings()


@lru_cache(maxsize=1)
def get_app_settings() -> AppSettings:
    return AppSettings(
        database=get_database_settings(),
        trading_symbols=DEFAULT_TRADING_SYMBOLS,
        ai_trading_symbols=DEFAULT_AI_TRADING_SYMBOLS,
    )


def build_db_connection_url(settings: DatabaseSettings | None = None) -> str:
    config = (settings or get_database_settings()).to_connection_kwargs()
    return (
        f"postgresql://{config['user']}:{config['password']}"
        f"@{config['host']}:{config['port']}/{config['database']}"
    )


__all__ = [
    "AppSettings",
    "DatabaseSettings",
    "DEFAULT_AI_TRADING_SYMBOLS",
    "DEFAULT_TRADING_SYMBOLS",
    "build_db_connection_url",
    "get_app_settings",
    "get_database_settings",
    "load_database_settings",
]
