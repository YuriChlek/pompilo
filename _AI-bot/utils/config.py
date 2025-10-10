import os

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", 5432)
DB_NAME = os.getenv("DATABASE", "pompilo_db")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASS = os.getenv("DB_PASS", "admin_pass")

BUY_DIRECTION = "buy"
SELL_DIRECTION = "sell"

TRADING_SYMBOLS = [
    'AAVEUSDT',
    'ADAUSDT',
    'APTUSDT',
    'AVAXUSDT',
    'DOTUSDT',
    'DOGEUSDT',
    'JUPUSDT',
    'SOLUSDT',
    'SUIUSDT',
    'TIAUSDT',
    'WIFUSDT',
    'WLDUSDT',
    'XRPUSDT',
]

AI_TRADING_SYMBOLS = [
    'SOLUSDT',
    'XRPUSDT',
]

SCHEMAS = [
    'bybit_trading_history_data',
    'binance_trading_history_data',
    'okx_trading_history_data',
]
