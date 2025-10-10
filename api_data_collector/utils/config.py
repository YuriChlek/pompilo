import os

# lenovo remote db "172.28.233.170"

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", 5432)
DB_NAME = os.getenv("DATABASE", "pompilo_db")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASS = os.getenv("DB_PASS", "admin_pass")

BUY_DIRECTION = "buy"
SELL_DIRECTION = "sell"

SCHEMAS = [
    'bybit_trading_history_data',
    'binance_trading_history_data',
    'okx_trading_history_data',
]
# Торгові символи
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

"""
Розмір трейдів які вважаються великими
"""

MIN_BIG_TRADES_SIZES = {
    'AAVEUSDT': 3000,
    'ADAUSDT': 3500000,
    'APTUSDT': 38000,
    'AVAXUSDT': 80000,
    'DOGEUSDT': 2000000,
    'DOTUSDT': 80000,
    'JUPUSDT': 300000,
    'SOLUSDT': 10000,
    'SUIUSDT': 250000,
    'TIAUSDT': 150000,
    'TAIUSDT': 1000000,
    'WIFUSDT': 150000,
    'WLDUSDT': 80000,
    'XRPUSDT': 300000
}

# Кількість знаків після коми для кожного символу
SYMBOLS_ROUNDING = {
    'AAVEUSDT': 2,
    'ADAUSDT': 4,
    'APTUSDT': 4,
    'AVAXUSDT': 4,
    'BNBUSDT': 2,
    'DOGEUSDT': 5,
    'DOTUSDT': 4,
    'ENAUSDT': 4,
    'JUPUSDT': 4,
    'SOLUSDT': 3,
    'SUIUSDT': 5,
    'TAOUSDT': 2,
    'TIAUSDT': 4,
    'TAIUSDT': 5,
    'TRUMPUSDT': 3,
    'UNIUSDT': 3,
    'WIFUSDT': 4,
    'WLDUSDT': 4,
    'XRPUSDT': 4
}
