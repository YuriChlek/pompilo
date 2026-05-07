from __future__ import annotations

import os
from decimal import Decimal

from .env import load_project_env

load_project_env()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DATABASE", "pompilo_db")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASS = os.getenv("DB_PASS", os.getenv("DB_PASSWORD", "admin_pass"))

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
BINANCE_REST_ENDPOINT = os.getenv("BINANCE_REST_ENDPOINT", "https://api.binance.com")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
BYBIT_API_ENDPOINT = os.getenv("BYBIT_API_ENDPOINT", "https://api-demo.bybit.com")
BYBIT_RECV_WINDOW = int(os.getenv("BYBIT_RECV_WINDOW", "10000"))

APP_TIMEZONE = os.getenv("APP_TIMEZONE", "Europe/Kyiv")
APP_ENV = os.getenv("APP_ENV", "dev")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", os.getenv("TOKEN", ""))
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", os.getenv("CHAT_ID", ""))

HEALTHCHECK_ENABLED = os.getenv("HEALTHCHECK_ENABLED", "true").lower() == "true"
HEALTHCHECK_HOST = os.getenv("HEALTHCHECK_HOST", "0.0.0.0")
HEALTHCHECK_PORT = int(os.getenv("HEALTHCHECK_PORT", "8080"))

CANDLES_DATA_SCHEMA = "_candles_trading_data"
SPOT_BOT_SCHEMA = "_spot_trading_bot"
D1_TABLE_SUFFIX = "_1d"
H4_TABLE_SUFFIX = "_4h"

SPOT_TRADING_SYMBOLS = [
    "ETHUSDT",
    "SUIUSDT",
    "TAOUSDT",
    "SOLUSDT",
    "BTCUSDT",
    "XRPUSDT",
    "LTCUSDT",
]

BINANCE_D1_INTERVAL = "1d"
BINANCE_H4_INTERVAL = "4h"
DEFAULT_LOOKBACK_DAYS = 1095
THREE_YEARS_DAYS = 1095
H4_ANALYSIS_DAYS = int(os.getenv("H4_ANALYSIS_DAYS", "180"))
H4_INCREMENTAL_SYNC_DAYS = int(os.getenv("H4_INCREMENTAL_SYNC_DAYS", "2"))
DEFAULT_DAILY_TARGET_HOUR = int(os.getenv("DAILY_TARGET_HOUR", "0"))
DEFAULT_DAILY_TARGET_MINUTE = int(os.getenv("DAILY_TARGET_MINUTE", "0"))
DEFAULT_DAILY_TARGET_SECOND = int(os.getenv("DAILY_TARGET_SECOND", "1"))

GREENWICH_LENGTH = int(os.getenv("GREENWICH_LENGTH", "98"))
GREENWICH_BASIS_TYPE = os.getenv("GREENWICH_BASIS_TYPE", "WMA")
GREENWICH_MULTIPLIER_1 = Decimal(os.getenv("GREENWICH_MULTIPLIER_1", "5.5"))
GREENWICH_MULTIPLIER_2 = Decimal(os.getenv("GREENWICH_MULTIPLIER_2", "4.5"))
GREENWICH_MULTIPLIER_3 = Decimal(os.getenv("GREENWICH_MULTIPLIER_3", "3.5"))
ANALYSIS_WINDOW = int(os.getenv("ANALYSIS_WINDOW", str(max(240, GREENWICH_LENGTH * 8))))
H4_ANALYSIS_WINDOW = int(os.getenv("H4_ANALYSIS_WINDOW", str(max(240, GREENWICH_LENGTH * 8))))
H4_SCHEDULER_ENABLED = os.getenv("H4_SCHEDULER_ENABLED", "true").lower() == "true"
D1_REGIME_FILTER_ENABLED = os.getenv("D1_REGIME_FILTER_ENABLED", "true").lower() == "true"

ORDER_DEPOSIT_PERCENT = Decimal(os.getenv("ORDER_DEPOSIT_PERCENT", "5"))
MIN_PROFIT_RATIO = Decimal(os.getenv("MIN_PROFIT_RATIO", "0.01"))
NO_LOSS_GUARD_ENABLED = os.getenv("NO_LOSS_GUARD_ENABLED", "true").lower() == "true"

_MIN_ALLOWED_PROFIT_RATIO = Decimal("0.01")
if MIN_PROFIT_RATIO < _MIN_ALLOWED_PROFIT_RATIO:
    raise ValueError(
        f"MIN_PROFIT_RATIO={MIN_PROFIT_RATIO} lower than minimum allowed "
        f"{_MIN_ALLOWED_PROFIT_RATIO} (minimum profitable SELL threshold is 1%)"
    )

BUY_SIGNAL = "buy"
SELL_SIGNAL = "sell"
HOLD_SIGNAL = "hold"
