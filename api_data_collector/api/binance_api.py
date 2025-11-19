import asyncio
from decimal import Decimal
from utils import TRADING_SYMBOLS

import asyncpg
import requests
import time
import uuid
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta, UTC


# -------------------- Клас Binance API -------------------- #
class BinanceAPI:
    """Клас для взаємодії з REST API Binance Spot."""

    def __init__(self, rest_endpoint: str = 'https://api.binance.com'):
        self.endpoint = rest_endpoint
        self.count = 1000
        self.session = requests.Session()

        retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    def _make_request(self, url: str, params: dict = None):
        for attempt in range(3):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Помилка запиту ({attempt + 1}): {e}")
                time.sleep(2)
        return []

    def fetch(self, symbol: str, start_time: int, end_time: int, interval: str = '1h'):
        """Отримання свічок з Binance."""
        url = f"{self.endpoint}/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': self.count
        }

        data = self._make_request(url, params)
        if not data:
            return []

        candles = []
        for d in data:
            # Використовуємо UTC час без конвертації в локальний пояс
            open_time = datetime.utcfromtimestamp(d[0] / 1000)
            close_time = datetime.utcfromtimestamp(d[6] / 1000)

            candles.append({
                'id': str(uuid.uuid4()),
                'symbol': symbol,
                'open_time': open_time,
                'close_time': close_time,
                'open': float(d[1]),
                'high': float(d[2]),
                'low': float(d[3]),
                'close': float(d[4]),
                'volume': float(d[5])
            })
        return candles

    def close(self):
        self.session.close()


# -------------------- Основна логіка -------------------- #
async def insert_candles(conn, schema: str, table: str, candles: list):
    """Вставка свічок у базу."""
    if not candles:
        return

    cvd = 0
    records = []
    for c in candles:
        # Просте обчислення CVD: якщо close > open → позитивний дельта
        delta = c['volume'] if c['close'] > c['open'] else -c['volume']
        cvd += delta

        records.append((
            c['open_time'],
            c['close_time'],
            c['symbol'],
            c['open'],
            c['close'],
            c['high'],
            c['low'],
            round(Decimal(cvd)),
            round(Decimal(c['volume']), 1),
            str(c['open_time'])
        ))

    sql = f"""
    INSERT INTO {schema}.{table} (
        open_time, close_time, symbol,
        open, close, high, low,
        cvd, volume, candle_id
    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
    ON CONFLICT (candle_id) DO UPDATE 
    SET open = EXCLUDED.open,
    close = EXCLUDED.close,
    high = EXCLUDED.high,
    low = EXCLUDED.low,
    cvd = EXCLUDED.cvd,
    volume = EXCLUDED.volume;
    """

    try:
        await conn.executemany(sql, records)
        print(f"💾 Записано {len(records)} рядків у {schema}.{table}")
    except Exception as e:
        print(f"❌ Помилка при записі в базу: {e}")
        # Додатковий дебаг
        if records:
            print(f"Перший запис: {records[0]}")
            print(f"Тип open_time: {type(records[0][0])}")


async def fetch_and_store(symbol: str, timeframe: str = '1h', days: int = 700):
    """Завантажує та зберігає свічки за вказаний період."""
    api = BinanceAPI()

    # Асинхронне підключення до бази
    conn = await asyncpg.connect(
        user='admin',
        password='admin_pass',
        database='pompilo_db',
        host='localhost',
        port='5432',
        timeout=7200
    )

    schema = '_candles_trading_data'
    table = f"{symbol.lower()}_p_candles"

    # Використовуємо datetime без timezone для уникнення конфліктів
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    current_start = start_time

    print(f"📊 Починаємо завантаження {symbol} ({timeframe}) за {days} днів...")
    print(f"📅 Період: з {start_time} по {end_time}")

    total_candles = 0

    try:
        while current_start < end_time:
            current_end = min(current_start + timedelta(days=5), end_time)

            print(f"🔍 Запит {symbol}: {current_start} - {current_end}")

            candles = api.fetch(
                symbol=symbol,
                start_time=int(current_start.timestamp() * 1000),
                end_time=int(current_end.timestamp() * 1000),
                interval=timeframe
            )

            if candles:
                await insert_candles(conn, schema, table, candles)
                total_candles += len(candles)
                print(f"📈 {symbol}: отримано {len(candles)} свічок")
            else:
                print(f"⚠️ {symbol}: немає даних для періоду {current_start} - {current_end}")

            current_start = current_end
            time.sleep(0.5)  # Пауза між запитами до API

        print(f"✅ {symbol}: Завантажено {total_candles} свічок.")

    except Exception as e:
        print(f"❌ Помилка для {symbol}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await conn.close()
        api.close()


async def run_api():
    """Основна асинхронна функція."""
    print(f"🔢 Загальна кількість символів: {len(TRADING_SYMBOLS)}")

    # Обробляємо символи послідовно для уникнення перевантаження
    for i, symbol in enumerate(TRADING_SYMBOLS, 1):
        print(f"\n🔄 Обробляємо символ {i}/{len(TRADING_SYMBOLS)}: {symbol}")
        await fetch_and_store(symbol, '1h', 1)

        # Пауза між символами
        if i < len(TRADING_SYMBOLS):
            print("⏳ Пауза 1 секунда між символами...")
            time.sleep(1)

    print(f"\n🎉 Всі {len(TRADING_SYMBOLS)} символи успішно оброблено!")


# -------------------- Запуск -------------------- #
if __name__ == '__main__':
    try:
        asyncio.run(run_api())
    except KeyboardInterrupt:
        print("⏹️ Скрипт зупинено користувачем")
    except Exception as e:
        print(f"💥 Неочікувана помилка: {e}")
        import traceback

        traceback.print_exc()