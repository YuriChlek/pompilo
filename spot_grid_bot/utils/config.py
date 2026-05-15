from __future__ import annotations

import os

from utils.env import load_project_env

load_project_env()

"""
DEFAULT_TRADING_SYMBOLS = [
    'BTCUSDT',
    'BNBUSDT',
    "DOGEUSDT",
    "ETHUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "SOLUSDT",
    "SUIUSDT",
    "XRPUSDT",
]
"""

DEFAULT_TRADING_SYMBOLS = [
    "ETHUSDT",
]


def _parse_symbols(raw_symbols: str) -> list[str]:
    symbols = [symbol.strip().upper() for symbol in raw_symbols.split(",") if symbol.strip()]
    return symbols or list(DEFAULT_TRADING_SYMBOLS)


def _get_float_env(name: str, default: str) -> float:
    raw_value = os.getenv(name, default)
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float value for {name}: {raw_value!r}") from exc


def _get_int_env(name: str, default: str) -> int:
    raw_value = os.getenv(name, default)
    try:
        return int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid int value for {name}: {raw_value!r}") from exc

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = _get_int_env("DB_PORT", "5432")
DB_NAME = os.getenv("DATABASE", "pompilo_db")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASS = os.getenv("DB_PASS", os.getenv("DB_PASSWORD", "admin_pass"))
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
BYBIT_API_ENDPOINT = os.getenv("BYBIT_API_ENDPOINT", "https://api-demo.bybit.com")
BYBIT_PUBLIC_WS_URL = os.getenv("BYBIT_PUBLIC_WS_URL", "wss://stream.bybit.com/v5/public/spot").strip()
BYBIT_RECV_WINDOW = _get_int_env("BYBIT_RECV_WINDOW", "10000")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "bybit_spot_demo").strip().lower()
HEALTHCHECK_HOST = os.getenv("HEALTHCHECK_HOST", "0.0.0.0").strip()
HEALTHCHECK_PORT = _get_int_env("HEALTHCHECK_PORT", "8080")
ENABLE_LIVE_PRICE_MONITOR = os.getenv("ENABLE_LIVE_PRICE_MONITOR", "true").strip().lower() in {"1", "true", "yes", "on"}
LIVE_PRICE_DEVIATION_ATR_MULTIPLIER = _get_float_env("LIVE_PRICE_DEVIATION_ATR_MULTIPLIER", "2.0")
LIVE_PRICE_MONITOR_COOLDOWN_SECONDS = _get_float_env("LIVE_PRICE_MONITOR_COOLDOWN_SECONDS", "60")
LIVE_PRICE_MONITOR_OPEN_TIMEOUT_SECONDS = _get_float_env("LIVE_PRICE_MONITOR_OPEN_TIMEOUT_SECONDS", "45")
LIVE_PRICE_MONITOR_RECONNECT_DELAY_SECONDS = _get_float_env("LIVE_PRICE_MONITOR_RECONNECT_DELAY_SECONDS", "5")

DEFAULT_BASE_ASSET = os.getenv("BASE_ASSET", "BTC")
DEFAULT_QUOTE_ASSET = os.getenv("QUOTE_ASSET", "USDT")
STARTING_QUOTE_BALANCE = _get_float_env("STARTING_QUOTE_BALANCE", "10000")
STARTING_BASE_BALANCE = _get_float_env("STARTING_BASE_BALANCE", "0")
CANDLE_LOOKBACK = _get_int_env("CANDLE_LOOKBACK", "2400")
MAX_NEW_ORDERS_PER_CYCLE = _get_int_env("MAX_NEW_ORDERS_PER_CYCLE", "5")
MIN_SYMBOL_ENTRY_NOTIONAL = _get_float_env("MIN_SYMBOL_ENTRY_NOTIONAL", "3")
MAX_SYMBOL_INVENTORY_PCT_OF_EQUITY = _get_float_env("MAX_SYMBOL_INVENTORY_PCT_OF_EQUITY", "0.08")
MAX_SYMBOL_NEW_ENTRY_PCT_OF_FREE_QUOTE = _get_float_env("MAX_SYMBOL_NEW_ENTRY_PCT_OF_FREE_QUOTE", "0.10")
UNDERWATER_AVERAGING_ENABLED = os.getenv("UNDERWATER_AVERAGING_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
UNDERWATER_AVERAGING_TRIGGER_PCT = _get_float_env("UNDERWATER_AVERAGING_TRIGGER_PCT", "0.10")
UNDERWATER_RECOVERY_BUDGET_PCT = _get_float_env("UNDERWATER_RECOVERY_BUDGET_PCT", "0.30")
UNDERWATER_RANGE_BUDGET_MULTIPLIER = _get_float_env("UNDERWATER_RANGE_BUDGET_MULTIPLIER", "0.50")
UNDERWATER_UPTREND_BUDGET_MULTIPLIER = _get_float_env("UNDERWATER_UPTREND_BUDGET_MULTIPLIER", "1.00")
UNDERWATER_MAX_RECOVERY_LEVELS = _get_int_env("UNDERWATER_MAX_RECOVERY_LEVELS", "2")
UNDERWATER_DEEP_STOP_PCT = _get_float_env("UNDERWATER_DEEP_STOP_PCT", "0.25")
RUN_TARGET_MINUTE = _get_int_env("RUN_TARGET_MINUTE", "0") # виправити при переході на мікросервісну архітектуру.
RUN_TARGET_SECOND = _get_int_env("RUN_TARGET_SECOND", "1")
STATE_SCHEMA = os.getenv("STATE_SCHEMA", "_spot_grid_state")
STATE_TABLE = os.getenv("STATE_TABLE", "symbol_runtime_state")

TRADING_SYMBOLS = _parse_symbols(
    os.getenv("SPOT_TRADING_SYMBOLS", os.getenv("TRADING_SYMBOLS", ",".join(DEFAULT_TRADING_SYMBOLS)))
)
