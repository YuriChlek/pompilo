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

APP_TIMEZONE = os.getenv("APP_TIMEZONE", "Europe/Kyiv")
APP_ENV = os.getenv("APP_ENV", "dev")

CANDLES_DATA_SCHEMA = "_candles_trading_data"
SPOT_BOT_SCHEMA = "_spot_trading_bot"
D1_TABLE_SUFFIX = "_p_candles_d1"

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
DEFAULT_LOOKBACK_DAYS = 1095
THREE_YEARS_DAYS = 1095
ANALYSIS_WINDOW = 240
DEFAULT_DAILY_TARGET_HOUR = int(os.getenv("DAILY_TARGET_HOUR", "0"))
DEFAULT_DAILY_TARGET_MINUTE = int(os.getenv("DAILY_TARGET_MINUTE", "5"))
DEFAULT_DAILY_TARGET_SECOND = int(os.getenv("DAILY_TARGET_SECOND", "0"))

GREENWICH_LENGTH = int(os.getenv("GREENWICH_LENGTH", "98"))
GREENWICH_BASIS_TYPE = os.getenv("GREENWICH_BASIS_TYPE", "WMA")
GREENWICH_MULTIPLIER_1 = Decimal(os.getenv("GREENWICH_MULTIPLIER_1", "5.5"))
GREENWICH_MULTIPLIER_2 = Decimal(os.getenv("GREENWICH_MULTIPLIER_2", "4.5"))
GREENWICH_MULTIPLIER_3 = Decimal(os.getenv("GREENWICH_MULTIPLIER_3", "3.5"))

ORDER_QUOTE_AMOUNT = Decimal(os.getenv("ORDER_QUOTE_AMOUNT", "50"))
MIN_PROFIT_RATIO = Decimal(os.getenv("MIN_PROFIT_RATIO", "0"))

BUY_SIGNAL = "buy"
SELL_SIGNAL = "sell"
HOLD_SIGNAL = "hold"
