from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Iterable, Optional, Sequence

import asyncpg
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from infrastructure.db import CANDLES_DATA_SCHEMA, DEFAULT_CANDLE_TABLE_SUFFIX, get_db_pool
from utils.config import TRADING_SYMBOLS

TABLE_SUFFIX = DEFAULT_CANDLE_TABLE_SUFFIX
API_REQUEST_PAUSE_SECONDS = 0.5
SYMBOL_PAUSE_SECONDS = 1.0
MAX_FETCH_RETRIES = 3
DEFAULT_TIMEFRAME = "1h"
DEFAULT_LOOKBACK_DAYS = 1
DEFAULT_BINANCE_ENDPOINT = "https://api.binance.com"
MAX_BINANCE_LIMIT = 1000


@dataclass(frozen=True, slots=True)
class BinanceCandle:
    """Describe one Binance candle normalized for PostgreSQL storage."""

    symbol: str
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def delta(self) -> float:
        """Return signed volume used for CVD approximation."""
        return self.volume if self.close > self.open else -self.volume

    @staticmethod
    def from_kline(symbol: str, payload: Sequence[float]) -> "BinanceCandle":
        """Build a normalized candle from raw Binance kline payload."""
        return BinanceCandle(
            symbol=symbol.upper(),
            open_time=datetime.fromtimestamp(payload[0] / 1000, tz=UTC).replace(tzinfo=None),
            close_time=datetime.fromtimestamp(payload[6] / 1000, tz=UTC).replace(tzinfo=None),
            open=float(payload[1]),
            high=float(payload[2]),
            low=float(payload[3]),
            close=float(payload[4]),
            volume=float(payload[5]),
        )


class BinanceAPI:
    """Wrap Binance Spot REST API calls used to fetch historical candles."""

    def __init__(self, rest_endpoint: str = DEFAULT_BINANCE_ENDPOINT):
        """Create a retry-enabled HTTP session for Binance requests."""
        self.endpoint = rest_endpoint
        self.limit = MAX_BINANCE_LIMIT
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def _make_request(self, url: str, params: Optional[dict] = None) -> list | dict:
        """Perform a GET request with retries and exponential backoff."""
        params = params or {}
        for attempt in range(MAX_FETCH_RETRIES):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as exc:
                print(f"⚠️ Помилка запиту ({attempt + 1}): {exc}")
                time.sleep(2 ** attempt)
        return []

    def fetch(self, symbol: str, start_time: int, end_time: int, interval: str = DEFAULT_TIMEFRAME) -> list[BinanceCandle]:
        """Fetch and normalize candle data for a specific time range."""
        payload = self._make_request(
            f"{self.endpoint}/api/v3/klines",
            {
                "symbol": symbol.upper(),
                "interval": interval,
                "startTime": start_time,
                "endTime": end_time,
                "limit": self.limit,
            },
        )
        return [BinanceCandle.from_kline(symbol, row) for row in payload]

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self.session.close()


async def insert_candles(
    conn: asyncpg.Connection,
    schema: str,
    table: str,
    candles: Sequence[BinanceCandle],
) -> None:
    """Insert a candle batch into PostgreSQL using the shared candle schema."""
    records = _build_candle_records(candles)
    if not records:
        return

    sql = f"""
    INSERT INTO {schema}.{table} (
        open_time, close_time, symbol,
        open, close, high, low,
        cvd, volume,
        candle_id
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
    except Exception as exc:
        print(f"❌ Помилка при записі в базу: {exc}")
        if records:
            print(f"Перший запис: {records[0]}")
            print(f"Тип open_time: {type(records[0][0])}")
        raise


def _build_candle_records(candles: Iterable[BinanceCandle]) -> list[tuple]:
    """Convert normalized candles into ``executemany`` records with cumulative CVD."""
    cvd = Decimal(0)
    records: list[tuple] = []
    for candle in candles:
        cvd += Decimal(candle.delta)
        records.append(
            (
                candle.open_time,
                candle.close_time,
                candle.symbol,
                candle.open,
                candle.close,
                candle.high,
                candle.low,
                round(cvd),
                round(Decimal(candle.volume), 1),
                str(candle.open_time),
            )
        )
    return records


async def fetch_and_store(
    symbol: str,
    timeframe: str = DEFAULT_TIMEFRAME,
    days: int = DEFAULT_LOOKBACK_DAYS,
    *,
    pool: asyncpg.Pool | None = None,
    table_suffix: str = TABLE_SUFFIX,
    api: BinanceAPI | None = None,
) -> None:
    """Fetch historical candles for one symbol and upsert them into PostgreSQL."""
    owns_pool = pool is None
    owns_api = api is None
    pool = pool or await get_db_pool()
    api = api or BinanceAPI()

    table = f"{symbol.lower()}{table_suffix}"
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    current_start = start_time
    total_candles = 0
    print(f"📊 Починаємо завантаження {symbol.upper()} ({timeframe}) за {days} днів...")
    print(f"📅 Період: з {start_time} по {end_time}")

    try:
        async with pool.acquire() as conn:
            while current_start < end_time:
                current_end = min(current_start + timedelta(days=5), end_time)
                print(f"🔍 Запит {symbol.upper()}: {current_start} - {current_end}")
                candles = api.fetch(
                    symbol=symbol,
                    start_time=int(current_start.timestamp() * 1000),
                    end_time=int(current_end.timestamp() * 1000),
                    interval=timeframe,
                )
                if candles:
                    await insert_candles(conn, CANDLES_DATA_SCHEMA, table, candles)
                    total_candles += len(candles)
                    print(f"📈 {symbol.upper()}: отримано {len(candles)} свічок")
                else:
                    print(f"⚠️ {symbol.upper()}: немає даних для періоду {current_start} - {current_end}")
                current_start = current_end
                await asyncio.sleep(API_REQUEST_PAUSE_SECONDS)
        print(f"✅ {symbol.upper()}: Завантажено {total_candles} свічок.")
    except Exception as exc:
        print(f"❌ Помилка для {symbol.upper()}: {exc}")
        raise
    finally:
        if owns_api:
            api.close()
        if owns_pool and pool is not None:
            await pool.close()


async def run_binance_candle_sync(
    symbols: Sequence[str] | None = None,
    timeframe: str = DEFAULT_TIMEFRAME,
    days: int = DEFAULT_LOOKBACK_DAYS,
) -> None:
    """Run sequential Binance candle synchronization for all requested symbols."""
    symbol_list = [str(symbol).upper() for symbol in (symbols or TRADING_SYMBOLS)]
    print(f"🔢 Загальна кількість символів: {len(symbol_list)}")
    pool = await get_db_pool()
    try:
        for index, symbol in enumerate(symbol_list, start=1):
            print(f"\n🔄 Обробляємо символ {index}/{len(symbol_list)}: {symbol}")
            await fetch_and_store(symbol, timeframe, days, pool=pool)
            if index < len(symbol_list):
                print(f"⏳ Пауза {SYMBOL_PAUSE_SECONDS} секунда між символами...")
                await asyncio.sleep(SYMBOL_PAUSE_SECONDS)
        print(f"\n🎉 Всі {len(symbol_list)} символи успішно оброблено!")
    finally:
        await pool.close()
