import os
from decimal import Decimal

from utils.env import load_project_env

load_project_env()

# lenovo remote db "172.28.233.170"

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", 5432)
DB_NAME = os.getenv("DATABASE", "pompilo_db")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASS = os.getenv("DB_PASS", os.getenv("DB_PASSWORD", "admin_pass"))
DB_CONNECT_TIMEOUT = float(os.getenv("DB_CONNECT_TIMEOUT", "5"))

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
BYBIT_MARKET_DATA_API_ENDPOINT = os.getenv("BYBIT_MARKET_DATA_API_ENDPOINT", "https://api.bybit.com")
BYBIT_MARKET_DATA_WS_ENDPOINT = os.getenv("BYBIT_MARKET_DATA_WS_ENDPOINT", "wss://stream.bybit.com/v5/public/spot")
BYBIT_TRADING_API_ENDPOINT = os.getenv("BYBIT_TRADING_API_ENDPOINT", "https://api-demo.bybit.com")
BYBIT_RECV_WINDOW = os.getenv("BYBIT_RECV_WINDOW", "10000")
TELEGRAM_TOKEN = os.getenv("TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("CHAT_ID", "")
APP_TIMEZONE = os.getenv("APP_TIMEZONE", "Europe/Kyiv")

# Кількість знаків після коми для кожного символу
SYMBOLS_ROUNDING = {
    'AAVEUSDT': 2,
    'ADAUSDT': 4,
    'APTUSDT': 4,
    'AVAXUSDT': 4,
    'BNBUSDT': 2,
    'BTCUSDT': 2,
    'DOGEUSDT': 5,
    'DOTUSDT': 4,
    'ENAUSDT': 4,
    'ETHUSDT': 2,
    'JUPUSDT': 4,
    'XRPUSDT': 5,
    'SOLUSDT': 3,
    'SUIUSDT': 5,

}

POSITION_ROUNDING_RULES = {
    'BTCUSDT': lambda x: round(Decimal(str(x)), 3),
    'ETHUSDT': lambda x: round(Decimal(str(x)), 2),
    'SOLUSDT': lambda x: round(Decimal(str(x)), 1),
    'DOGEUSDT': lambda x: int(round(Decimal(str(x)), 0)),
    'XRPUSDT': lambda x: int(round(Decimal(str(x)), 0)),
    'SUIUSDT': lambda x: int(round(Decimal(str(x)), -1)),
}

ORDERFLOW_TICK_SIZES = {
    symbol: 10 ** (-precision)
    for symbol, precision in SYMBOLS_ROUNDING.items()
}

ORDERFLOW_DB_SCHEMA = "orderflow_scalp"
ORDERFLOW_ANALYSIS_REFERENCE_EXCHANGE = os.getenv("ORDERFLOW_ANALYSIS_REFERENCE_EXCHANGE", "bybit")
ORDERFLOW_OBSERVATION_EXCHANGES = tuple(
    item.strip()
    for item in os.getenv("ORDERFLOW_OBSERVATION_EXCHANGES", "bybit,binance,okx").split(",")
    if item.strip()
)
ORDERFLOW_SYMBOLS = tuple(
    item.strip().upper()
    for item in os.getenv("ORDERFLOW_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT,XRPUSDT,SUIUSDT").split(",")
    if item.strip()
)
ORDERFLOW_BOOK_DEPTH = int(os.getenv("ORDERFLOW_BOOK_DEPTH", "20"))
ORDERFLOW_BOOK_HISTORY_SIZE = int(os.getenv("ORDERFLOW_BOOK_HISTORY_SIZE", "32"))
ORDERFLOW_TAPE_WINDOW_MS = int(os.getenv("ORDERFLOW_TAPE_WINDOW_MS", "3000"))
ORDERFLOW_WALL_LOOKBACK_MS = int(os.getenv("ORDERFLOW_WALL_LOOKBACK_MS", "1500"))
ORDERFLOW_MAX_WALL_DISTANCE_BPS = float(os.getenv("ORDERFLOW_MAX_WALL_DISTANCE_BPS", "8"))
ORDERFLOW_MAX_WALL_DISTANCE_TICKS = int(os.getenv("ORDERFLOW_MAX_WALL_DISTANCE_TICKS", "8"))
ORDERFLOW_MIN_WALL_NOTIONAL_USDT = float(os.getenv("ORDERFLOW_MIN_WALL_NOTIONAL_USDT", "250000"))
ORDERFLOW_MIN_WALL_RELATIVE_SIZE = float(os.getenv("ORDERFLOW_MIN_WALL_RELATIVE_SIZE", "4.0"))
ORDERFLOW_MIN_WALL_PERSIST_MS = int(os.getenv("ORDERFLOW_MIN_WALL_PERSIST_MS", "800"))
ORDERFLOW_MAX_WALL_SIZE_DROP_PCT = float(os.getenv("ORDERFLOW_MAX_WALL_SIZE_DROP_PCT", "0.45"))
ORDERFLOW_MIN_WALL_SCORE = float(os.getenv("ORDERFLOW_MIN_WALL_SCORE", "65"))
ORDERFLOW_MAX_PULL_EVENTS = int(os.getenv("ORDERFLOW_MAX_PULL_EVENTS", "3"))
ORDERFLOW_MAX_CHASE_TICKS = int(os.getenv("ORDERFLOW_MAX_CHASE_TICKS", "2"))
ORDERFLOW_MAX_SPOOF_SCORE = float(os.getenv("ORDERFLOW_MAX_SPOOF_SCORE", "45"))
ORDERFLOW_MIN_TEST_COUNT = int(os.getenv("ORDERFLOW_MIN_TEST_COUNT", "2"))
ORDERFLOW_TEST_TOUCH_TICKS = int(os.getenv("ORDERFLOW_TEST_TOUCH_TICKS", "2"))
ORDERFLOW_TEST_DEBOUNCE_MS = int(os.getenv("ORDERFLOW_TEST_DEBOUNCE_MS", "500"))
ORDERFLOW_MIN_REJECTION_TICKS = int(os.getenv("ORDERFLOW_MIN_REJECTION_TICKS", "1"))
ORDERFLOW_MIN_DEFENDED_RATIO = float(os.getenv("ORDERFLOW_MIN_DEFENDED_RATIO", "0.55"))
ORDERFLOW_MIN_TAPE_PRESSURE_RATIO = float(os.getenv("ORDERFLOW_MIN_TAPE_PRESSURE_RATIO", "1.15"))
ORDERFLOW_CROSS_CONFIRMATION_BPS = float(os.getenv("ORDERFLOW_CROSS_CONFIRMATION_BPS", "6"))
ORDERFLOW_MIN_CROSS_EXCHANGE_CONFIRMATIONS = int(os.getenv("ORDERFLOW_MIN_CROSS_EXCHANGE_CONFIRMATIONS", "1"))
ORDERFLOW_BOOK_STALE_MS = int(os.getenv("ORDERFLOW_BOOK_STALE_MS", "2500"))
ORDERFLOW_ENTRY_OFFSET_TICKS = int(os.getenv("ORDERFLOW_ENTRY_OFFSET_TICKS", "1"))
ORDERFLOW_INVALIDATION_OFFSET_TICKS = int(os.getenv("ORDERFLOW_INVALIDATION_OFFSET_TICKS", "1"))
ORDERFLOW_STOP_OFFSET_TICKS = int(os.getenv("ORDERFLOW_STOP_OFFSET_TICKS", "1"))
ORDERFLOW_TAKE_PROFIT_R_MULTIPLE = float(os.getenv("ORDERFLOW_TAKE_PROFIT_R_MULTIPLE", "1.5"))
ORDERFLOW_PENDING_WALL_TOLERANCE_TICKS = int(os.getenv("ORDERFLOW_PENDING_WALL_TOLERANCE_TICKS", "2"))
ORDERFLOW_BREAKEVEN_ARM_TICKS = int(os.getenv("ORDERFLOW_BREAKEVEN_ARM_TICKS", "3"))
ORDERFLOW_BREAKEVEN_BUFFER_TICKS = int(os.getenv("ORDERFLOW_BREAKEVEN_BUFFER_TICKS", "1"))
ORDERFLOW_RISK_PER_TRADE_PCT = float(os.getenv("ORDERFLOW_RISK_PER_TRADE_PCT", "0.15"))
ORDERFLOW_MAX_DAILY_LOSS_PCT = float(os.getenv("ORDERFLOW_MAX_DAILY_LOSS_PCT", "1.5"))
ORDERFLOW_MAX_TRADES_PER_DAY = int(os.getenv("ORDERFLOW_MAX_TRADES_PER_DAY", "30"))
ORDERFLOW_MAX_CONSECUTIVE_LOSSES = int(os.getenv("ORDERFLOW_MAX_CONSECUTIVE_LOSSES", "5"))
ORDERFLOW_SYMBOL_COOLDOWN_SECONDS = int(os.getenv("ORDERFLOW_SYMBOL_COOLDOWN_SECONDS", "120"))
