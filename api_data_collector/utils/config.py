import os

# lenovo remote db "172.28.233.170"

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", 5432)
DB_NAME = os.getenv("DATABASE", "pompilo_db")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASS = os.getenv("DB_PASS", "admin_pass")

BUY_DIRECTION = "buy"
SELL_DIRECTION = "sell"

# Торгові символи
TRADING_SYMBOLS = [
    'AAVEUSDT',
    'ADAUSDT',
    'ARBUSDT',
    'APTUSDT',
    'AVAXUSDT',
    'BNBUSDT',
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
    'TIAUSDT',
    'UNIUSDT',
    'VIRTUALUSDT',
    'WIFUSDT',
    'WLDUSDT',
    'XRPUSDT'
]

POSITION_ROUNDING_RULES = {
            # Цілі числа
            'SUIUSDT': lambda x: int(round(x, -1)),
            'ADAUSDT': lambda x: int(round(x, 0)),
            'DOGEUSDT': lambda x: int(round(x, 0)),
            'JUPUSDT': lambda x: int(round(x, 0)),
            'WIFUSDT': lambda x: int(round(x, 0)),
            'XRPUSDT': lambda x: int(round(x, 0)),
            'TIAUSDT': lambda x: round(x, 0),
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

            # 2 знаки після коми
            'AAVEUSDT': lambda x: round(x, 2),
            'APTUSDT': lambda x: round(x, 2),
            'ETHUSDT': lambda x: round(x, 2),
            'TAOUSDT': lambda x: round(x, 2),
            'BNBUSDT': lambda x: round(x, 2),
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
    'BNBUSDT': 2,
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
    'TAIUSDT': 5,
    'TAOUSDT': 2,
    'UNIUSDT': 3,
    'VIRTUALUSDT': 4,
    'WIFUSDT': 4,
    'WLDUSDT': 4,
    'XRPUSDT': 4
}
