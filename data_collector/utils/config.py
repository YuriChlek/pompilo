import os
from pathlib import Path
# lenovo remote db "172.28.233.170"

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", 5432)
DB_NAME = os.getenv("DATABASE", "pompilo_db")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASS = os.getenv("DB_PASS", "admin_pass")

DATA_FOLDER = Path("../csv_data")

BATCH_SIZE = 2000

BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/linear"

SYMBOL = "SOLUSDT"
