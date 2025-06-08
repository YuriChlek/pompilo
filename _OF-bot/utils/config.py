import os

# lenovo remote db "172.28.233.170"

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", 5432)
DB_NAME = os.getenv("DATABASE", "pompilo_db")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASS = os.getenv("DB_PASS", "admin_pass")

BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/linear"
BINANCE_WS_URL = ""

IMBALANCE_AND_CVD_PERIOD = 8
EXTREMES_PERIOD = 48

BUY_DIRECTION = "buy"
SELL_DIRECTION = "sell"

'''
 BIN_SIZE - 0.005 від ціни як розмір біну необхідна для розрахунку профілю об'єму
 тобто відкриті угоди в межах 0.005 від ціни усереднюються
'''
BIN_SIZE = 0.005

SCHEMAS = [
    'bybit_trading_history_data',
    'binance_trading_history_data',
    'okx_trading_history_data'
]
# Торгові символи
TRADING_SYMBOLS = [
    'ALCHUSDT',
    'AAVEUSDT',
    'ADAUSDT',
    'APTUSDT',
    'BNBUSDT',
    'DOGEUSDT',
    # 'DOTUSDT',
    'ENAUSDT',
    'ETHUSDT',
    'JUPUSDT',
    'SOLUSDT',
    'TAOUSDT',
    'TIAUSDT',
    'UNIUSDT',
    'WIFUSDT',
    'WLDUSDT',
    'XRPUSDT'
]
# Множник для великих трейдів, середній розмір трейду * на множник (можливо вкажу просто розмір трейдів)
BIG_TRADES_MULTIPLIERS = {
    'ALCHUSDT': 0,  # Перевірити чи є пара при підключенні через вебсокет на бінанс
    'AAVEUSDT': 1000,  # Середній розмір угоди 1.75 AAVE
    'ADAUSDT': 940,  # Середній розмір угоди 743 ADA
    'APTUSDT': 1500,  # Середній розмір угоди 33 APT
    'BNBUSDT': 2200,  # Середній розмір угоди 0.81639 BNB
    'DOGEUSDT': 1950,  # Середній розмір угоди 3019.32 DOGE
    # 'DOTUSDT': 2,
    'ENAUSDT': 600,  # Середній розмір угоди 1020.45 ENA
    'ETHUSDT': 2500, # Середній розмір угоди 1.30 ETH
    'JUPUSDT': 1000,  # Середній розмір угоди 496.48 JUP (Поспостерігати на демо)
    'SOLUSDT': 2000,  # Середній розмір угоди 10.4 SOL
    'TAOUSDT': 2,
    'TIAUSDT': 2,
    'UNIUSDT': 2,
    'WIFUSDT': 2,
    'WLDUSDT': 2,
    'XRPUSDT': 1500  # Середній розмір угоди 429 XRP
}

# Кількість знаків після коми для кожного символу
SYMBOLS_ROUNDING = {
    'AAVEUSDT': 2,
    'ADAUSDT': 4,
    'APTUSDT': 4,
    'BNBUSDT': 2,
    'DOGEUSDT': 5,
    # 'DOTUSDT': 4,
    'ENAUSDT': 4,
    'ETHUSDT': 2,
    'JUPUSDT': 4,
    'SOLUSDT': 3,
    'TAOUSDT': 2,
    'TIAUSDT': 4,
    'TRUMPUSDT': 3,
    'UNIUSDT': 3,
    'WIFUSDT': 4,
    'WLDUSDT': 4,
    'XRPUSDT': 4

}
