import os

# lenovo remote db "172.28.233.170"

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", 5432)
DB_NAME = os.getenv("DATABASE", "pompilo_db")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASS = os.getenv("DB_PASS", "admin_pass")

BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/linear"
BINANCE_WS_URL = ""

IMBALANCE_AND_CVD_PERIOD = 1
EXTREMES_PERIOD = 8

BUY_DIRECTION = "buy"
SELL_DIRECTION = "sell"

SCHEMAS = [
    'bybit_trading_history_data',
    'binance_trading_history_data',
    'okx_trading_history_data',
    'bitget_trading_history_data',
    'gateio_trading_history_data',
]

CANDLES_SHEMA = '_candles_trading_data'

SCHEMAS_CANDLES = [
    'candles_trading_candles_data'
]

# Торгові символи
TRADING_SYMBOLS = [
    'AVAXUSDT',
    'DOTUSDT',
    'SOLUSDT',
    'XRPUSDT',
]

"""
 'AAVEUSDT',
    'ADAUSDT',
    'APTUSDT',
    'AVAXUSDT',
    'BNBUSDT',
    'DOTUSDT',
    'DOGEUSDT',
    'ETHUSDT',
    'JUPUSDT',
    'SUIUSDT',
    'SOLUSDT'
    'TIAUSDT',
    'WIFUSDT',
    'WLDUSDT',
    'XRPUSDT',
"""

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
    'ETHUSDT': 2,
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

SYMBOLS_RANGE = {
    'AAVEUSDT': 2,
    'ADAUSDT': 0.04,
    'APTUSDT': 4,
    'AVAXUSDT': 4,
    'BNBUSDT': 2,
    'DOGEUSDT': 5,
    'DOTUSDT': 4,
    'ENAUSDT': 4,
    'ETHUSDT': 2,
    'JUPUSDT': 4,
    'SOLUSDT': 3,
    'SUIUSDT': 5,
    'TAOUSDT': 2,
    'TIAUSDT': 4,
    'TRUMPUSDT': 3,
    'UNIUSDT': 3,
    'WIFUSDT': 4,
    'WLDUSDT': 4,
    'XRPUSDT': 4
}
