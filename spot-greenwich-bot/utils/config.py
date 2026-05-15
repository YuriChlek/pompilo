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
NOTIFICATION_ONLY_MODE = os.getenv("NOTIFICATION_ONLY_MODE", "false").lower() == "true"
ANTI_CRASH_BUY_BLOCK_ENABLED = os.getenv("ANTI_CRASH_BUY_BLOCK_ENABLED", "true").lower() == "true"
ANTI_CRASH_LOOKBACK_CANDLES = int(os.getenv("ANTI_CRASH_LOOKBACK_CANDLES", "3"))
ANTI_CRASH_MAX_DROP_RATIO = Decimal(os.getenv("ANTI_CRASH_MAX_DROP_RATIO", "0.10"))
CONFIRMATION_CANDLE_ENABLED = os.getenv("CONFIRMATION_CANDLE_ENABLED", "true").lower() == "true"
ATR_POSITION_SIZING_ENABLED = os.getenv("ATR_POSITION_SIZING_ENABLED", "true").lower() == "true"
ATR_POSITION_SIZING_MEDIAN_WINDOW = int(os.getenv("ATR_POSITION_SIZING_MEDIAN_WINDOW", "50"))
ATR_POSITION_SIZING_MIN_MULTIPLIER = Decimal(os.getenv("ATR_POSITION_SIZING_MIN_MULTIPLIER", "0.5"))
ATR_POSITION_SIZING_MAX_MULTIPLIER = Decimal(os.getenv("ATR_POSITION_SIZING_MAX_MULTIPLIER", "1.5"))
BUY_PRICE_GUARD_ENABLED = os.getenv("BUY_PRICE_GUARD_ENABLED", "true").lower() == "true"
BUY_PRICE_GUARD_MAX_DEVIATION_RATIO = Decimal(os.getenv("BUY_PRICE_GUARD_MAX_DEVIATION_RATIO", "0.01"))
PORTFOLIO_CAP_ENABLED = os.getenv("PORTFOLIO_CAP_ENABLED", "true").lower() == "true"
PORTFOLIO_POSITION_LIMIT = int(os.getenv("PORTFOLIO_POSITION_LIMIT", "3"))
PORTFOLIO_PRIORITY_SYMBOLS = tuple(
    symbol.strip().upper()
    for symbol in os.getenv("PORTFOLIO_PRIORITY_SYMBOLS", "BTCUSDT,ETHUSDT").split(",")
    if symbol.strip()
)
DRY_RUN_QUOTE_BALANCE = Decimal(os.getenv("DRY_RUN_QUOTE_BALANCE", "1000"))

ORDER_DEPOSIT_PERCENT = Decimal(os.getenv("ORDER_DEPOSIT_PERCENT", "5"))
AVERAGING_ENTRY_LIMIT = int(os.getenv("AVERAGING_ENTRY_LIMIT", "3"))
AVERAGING_ENTRY_2_SIZE_PERCENT = Decimal(os.getenv("AVERAGING_ENTRY_2_SIZE_PERCENT", "60"))
AVERAGING_ENTRY_3_SIZE_PERCENT = Decimal(os.getenv("AVERAGING_ENTRY_3_SIZE_PERCENT", "30"))
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
