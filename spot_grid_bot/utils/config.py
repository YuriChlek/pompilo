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

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DATABASE", "pompilo_db")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASS = os.getenv("DB_PASS", os.getenv("DB_PASSWORD", "admin_pass"))
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
BYBIT_API_ENDPOINT = os.getenv("BYBIT_API_ENDPOINT", "https://api-demo.bybit.com")
BYBIT_RECV_WINDOW = int(os.getenv("BYBIT_RECV_WINDOW", "10000"))
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "bybit_spot_demo").strip().lower()

DEFAULT_BASE_ASSET = os.getenv("BASE_ASSET", "BTC")
DEFAULT_QUOTE_ASSET = os.getenv("QUOTE_ASSET", "USDT")
STARTING_QUOTE_BALANCE = float(os.getenv("STARTING_QUOTE_BALANCE", "10000"))
STARTING_BASE_BALANCE = float(os.getenv("STARTING_BASE_BALANCE", "0"))
CANDLE_LOOKBACK = int(os.getenv("CANDLE_LOOKBACK", "2400"))
MAX_NEW_ORDERS_PER_CYCLE = int(os.getenv("MAX_NEW_ORDERS_PER_CYCLE", "5"))
MIN_SYMBOL_ENTRY_NOTIONAL = float(os.getenv("MIN_SYMBOL_ENTRY_NOTIONAL", "3"))
MAX_SYMBOL_INVENTORY_PCT_OF_EQUITY = float(os.getenv("MAX_SYMBOL_INVENTORY_PCT_OF_EQUITY", "0.08"))
MAX_SYMBOL_NEW_ENTRY_PCT_OF_FREE_QUOTE = float(os.getenv("MAX_SYMBOL_NEW_ENTRY_PCT_OF_FREE_QUOTE", "0.10"))
UNDERWATER_AVERAGING_ENABLED = os.getenv("UNDERWATER_AVERAGING_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
UNDERWATER_AVERAGING_TRIGGER_PCT = float(os.getenv("UNDERWATER_AVERAGING_TRIGGER_PCT", "0.10"))
UNDERWATER_RECOVERY_BUDGET_PCT = float(os.getenv("UNDERWATER_RECOVERY_BUDGET_PCT", "0.30"))
UNDERWATER_RANGE_BUDGET_MULTIPLIER = float(os.getenv("UNDERWATER_RANGE_BUDGET_MULTIPLIER", "0.50"))
UNDERWATER_UPTREND_BUDGET_MULTIPLIER = float(os.getenv("UNDERWATER_UPTREND_BUDGET_MULTIPLIER", "1.00"))
UNDERWATER_MAX_RECOVERY_LEVELS = int(os.getenv("UNDERWATER_MAX_RECOVERY_LEVELS", "2"))
UNDERWATER_DEEP_STOP_PCT = float(os.getenv("UNDERWATER_DEEP_STOP_PCT", "0.25"))
RUN_TARGET_MINUTE = int(os.getenv("RUN_TARGET_MINUTE", "0")) # виправити при переході на мікросервісну архітектуру.
RUN_TARGET_SECOND = int(os.getenv("RUN_TARGET_SECOND", "1"))
STATE_SCHEMA = os.getenv("STATE_SCHEMA", "_spot_grid_state")
STATE_TABLE = os.getenv("STATE_TABLE", "symbol_runtime_state")

TRADING_SYMBOLS = _parse_symbols(
    os.getenv("SPOT_TRADING_SYMBOLS", os.getenv("TRADING_SYMBOLS", ",".join(DEFAULT_TRADING_SYMBOLS)))
)
