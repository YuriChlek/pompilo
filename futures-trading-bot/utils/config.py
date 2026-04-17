import os
from decimal import Decimal

from .env import load_project_env

# lenovo remote db "172.28.233.170"

load_project_env()


def _env_flag(name: str, default: bool) -> bool:
    """Parse a boolean-like environment flag with a conservative fallback."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _env_decimal(name: str, default: str) -> Decimal:
    """Parse a decimal environment variable with a safe string fallback."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return Decimal(default)
    try:
        return Decimal(raw_value.strip())
    except Exception:
        return Decimal(default)


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    """Parse an integer environment variable with optional lower-bound validation."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value.strip())
    except Exception:
        return default
    if minimum is not None and value < minimum:
        return default
    return value

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", 5432)
DB_NAME = os.getenv("DATABASE", "pompilo_db")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASS = os.getenv("DB_PASS", os.getenv("DB_PASSWORD", "admin_pass"))
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "test-bybit-api-key")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "test-bybit-api-secret")
BYBIT_API_ENDPOINT = os.getenv("BYBIT_API_ENDPOINT", "https://api-demo.bybit.com")
BYBIT_RECV_WINDOW = os.getenv("BYBIT_RECV_WINDOW", "10000")
BYBIT_TAKER_FEE_RATE = Decimal(os.getenv("BYBIT_TAKER_FEE_RATE", "0.00055"))
ENABLE_BREAKEVEN_STOP_MANAGEMENT = _env_flag("ENABLE_BREAKEVEN_STOP_MANAGEMENT", True)
ENABLE_BREAKEVEN_PARTIAL_CLOSE = _env_flag("ENABLE_BREAKEVEN_PARTIAL_CLOSE", True)
BREAKEVEN_TRIGGER_R = _env_decimal("BREAKEVEN_TRIGGER_R", "1.5")
LIMIT_ORDER_MAX_AGE_HOURS = int(os.getenv("LIMIT_ORDER_MAX_AGE", "24"))
LIMIT_ORDER_MAX_AGE_MS = LIMIT_ORDER_MAX_AGE_HOURS * 60 * 60 * 1000
ANALYSIS_CANDLE_LIMIT = _env_int("ANALYSIS_CANDLE_LIMIT", 1500, minimum=100)

BUY_DIRECTION = "buy"
SELL_DIRECTION = "sell"

# Торгові символи
TRADING_SYMBOLS = [
    'AAVEUSDT',
    'ADAUSDT',
    'ARBUSDT',
    'APTUSDT',
    'AVAXUSDT',
    'DOGEUSDT',
    'DOTUSDT',
    'ENAUSDT',
    'ETHUSDT',
    'JUPUSDT',
    'LINKUSDT',
    'LTCUSDT',
    'NEARUSDT',
    'PENGUUSDT',
    'SOLUSDT',
    'SUIUSDT',
    'TAOUSDT',
    'TONUSDT',
    'UNIUSDT',
    'VIRTUALUSDT',
    'WIFUSDT',
    'WLDUSDT',
    'XRPUSDT',
    'ZECUSDT'
]

POSITION_ROUNDING_RULES = {
            # Цілі числа
            'SUIUSDT': lambda x: int(round(x, -1)),
            'ADAUSDT': lambda x: int(round(x, 0)),
            'DOGEUSDT': lambda x: int(round(x, 0)),
            'JUPUSDT': lambda x: int(round(x, 0)),
            'WIFUSDT': lambda x: int(round(x, 0)),
            'XRPUSDT': lambda x: int(round(x, 0)),
            'ENAUSDT': lambda x: round(x, 0),
            'PENGUUSDT': lambda x: round(x, 0),
            'VIRTUALUSDT': lambda x: round(x, 0),

            # 1 знак після коми
            'SOLUSDT': lambda x: round(x, 1),
            'AVAXUSDT': lambda x: round(x, 1),
            'NEARUSDT': lambda x: round(x, 1),
            'WLDUSDT': lambda x: round(x, 1),
            'LINKUSDT': lambda x: round(x, 1),
            'LTCUSDT': lambda x: round(x, 1),
            'UNIUSDT': lambda x: round(x, 1),
            'ARBUSDT': lambda x: round(x, 1),
            'DOTUSDT': lambda x: round(x, 1),
            'TONUSDT': lambda x: round(x, 1),

            # 2 знаки після коми
            'AAVEUSDT': lambda x: round(x, 2),
            'APTUSDT': lambda x: round(x, 2),
            'ETHUSDT': lambda x: round(x, 2),
            'TAOUSDT': lambda x: round(x, 2),
            'ZECUSDT': lambda x: round(x, 2),
        }

TEST_TRADING_SYMBOLS = [
    'SOLUSDT',
]

# Кількість знаків після коми для кожного символу
SYMBOLS_ROUNDING = {
    'AAVEUSDT': 2,
    'ADAUSDT': 4,
    'ARBUSDT': 4,
    'APTUSDT': 4,
    'AVAXUSDT': 4,
    'DOGEUSDT': 5,
    'DOTUSDT': 4,
    'ENAUSDT': 4,
    'ETHUSDT': 2,
    'JUPUSDT': 4,
    'LINKUSDT': 3,
    'LTCUSDT': 4,
    'NEARUSDT': 3,
    'PENGUUSDT': 6,
    'SOLUSDT': 3,
    'SUIUSDT': 5,
    'TAOUSDT': 2,
    'TONUSDT': 4,
    'UNIUSDT': 3,
    'VIRTUALUSDT': 4,
    'WIFUSDT': 4,
    'WLDUSDT': 4,
    'XRPUSDT': 4,
    'ZECUSDT': 2,
}
