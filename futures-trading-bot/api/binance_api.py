import asyncio
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Iterable, List, Optional, Sequence

import asyncpg
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.config import TRADING_SYMBOLS
from utils.db_actions import CANDLES_DATA_SCHEMA, get_db_pool

@dataclass(frozen=True)
class Candle:
    """Describe a Binance candle stored in the database.

    Attributes:
        symbol (str): Trading pair ticker in ``BTCUSDT`` format.
        open_time (datetime): Candle open time in naive UTC.
        close_time (datetime): Candle close time in naive UTC.
        open (float): Open price.
        high (float): Highest price inside the interval.
        low (float): Lowest price inside the interval.
        close (float): Close price.
        volume (float): Traded volume during the interval.
    """

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
        """Return signed volume used for CVD approximation.

        Returns:
            float: Positive when the candle closes above open, otherwise negative.

        Edge Cases:
            Zero volume returns ``0.0`` and keeps cumulative calculations stable.
        """
        return self.volume if self.close > self.open else -self.volume

    @staticmethod
    def from_kline(symbol: str, payload: Sequence[float]) -> "Candle":
        """Build a candle object from raw Binance kline payload.

        Args:
            symbol (str): Trading pair requested from the API.
            payload (Sequence[float]): Binance response row with open, high, low, and related fields.

        Returns:
            Candle: Normalized structure ready for storage.

        Edge Cases:
            The API may temporarily return incomplete rows during market updates; those
            records are filtered by the caller before storage.
        """
        return Candle(
            symbol=symbol,
            open_time=datetime.fromtimestamp(payload[0] / 1000, tz=UTC).replace(tzinfo=None),
            close_time=datetime.fromtimestamp(payload[6] / 1000, tz=UTC).replace(tzinfo=None),
            open=float(payload[1]),
            high=float(payload[2]),
            low=float(payload[3]),
            close=float(payload[4]),
            volume=float(payload[5]),
        )


TABLE_SUFFIX = '_1h'
API_REQUEST_PAUSE_SECONDS = 0.5
SYMBOL_PAUSE_SECONDS = 1.0
MAX_FETCH_RETRIES = 3
DEFAULT_TIMEFRAME = '1h'
DEFAULT_LOOKBACK_DAYS = 1


# -------------------- Клас Binance API -------------------- #
class BinanceAPI:
    """Wrap Binance Spot REST API calls used to fetch historical candles."""

    def __init__(self, rest_endpoint: str = 'https://api.binance.com'):
        """Create an HTTP session with retries for Binance requests.

        Args:
            rest_endpoint (str): Base Binance URL; override it for test environments.

        Edge Cases:
            ``self.count`` is capped at 1000, so long date ranges must be split across
            multiple ``fetch`` calls.
        """

        self.endpoint = rest_endpoint
        self.count = 1000
        self.session = requests.Session()

        retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    def _make_request(self, url: str, params: Optional[dict] = None):
        """Perform a GET request with exponential retries.

        Args:
            url (str): Full Binance resource URL.
            params (Optional[dict]): Query parameters passed to ``requests``.

        Returns:
            list | dict: Parsed JSON response, or an empty list after all retries are exhausted.

        Edge Cases:
            On persistent failures the method returns ``[]``, signaling callers to skip insertion.
        """

        params = params or {}
        for attempt in range(MAX_FETCH_RETRIES):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Помилка запиту ({attempt + 1}): {e}")
                time.sleep(2 ** attempt)
        return []

    def fetch(self, symbol: str, start_time: int, end_time: int, interval: str = DEFAULT_TIMEFRAME) -> List[Candle]:
        """Fetch candle data for a specific time range.

        Args:
            symbol (str): Trading pair to request.
            start_time (int): Start timestamp in Unix milliseconds.
            end_time (int): End timestamp in Unix milliseconds.
            interval (str): Candle interval such as ``1h`` or ``4h``.

        Returns:
            List[Candle]: Normalized candles, possibly empty when no data is available.

        Edge Cases:
            Binance limits ``limit`` to 1000, so longer ranges require multiple calls.
        """
        url = f"{self.endpoint}/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': self.count
        }

        data = self._make_request(url, params)
        return [Candle.from_kline(symbol, payload=d) for d in data]

    def close(self):
        """Close the underlying HTTP session and release resources."""

        self.session.close()


# -------------------- Основна логіка -------------------- #
async def insert_candles(
    conn: asyncpg.Connection,
    schema: str,
    table: str,
    candles: Sequence[Candle],
) -> None:
    """Insert a batch of candles into the target PostgreSQL table.

    Args:
        conn (asyncpg.Connection): Active database connection.
        schema (str): Target schema name.
        table (str): Table matching the symbol and timeframe.
        candles (Sequence[Candle]): Normalized candles to insert.

    Returns:
        None: The function completes through side effects only.

    Edge Cases:
        An empty candle sequence is skipped without executing SQL.
    """
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


def _build_candle_records(candles: Iterable[Candle]) -> List[tuple]:
    """Convert candle objects into tuples suitable for ``executemany``.

    Args:
        candles (Iterable[Candle]): Candle source of any iterable type.

    Returns:
        List[tuple]: Prepared records with cumulative CVD included.

    Edge Cases:
        Returns an empty list when the input iterable is empty.
    """
    cvd = Decimal(0)
    records: List[tuple] = []
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
    pool: Optional[asyncpg.Pool] = None,
    table_suffix: str = TABLE_SUFFIX,
) -> None:
    """Fetch historical candles for a symbol and store them in the database.

    Args:
        symbol (str): Trading pair such as ``SOLUSDT``.
        timeframe (str): Candle interval requested from Binance.
        days (int): Number of days of history to load.
        pool (Optional[asyncpg.Pool]): Existing connection pool; created automatically when None.
        table_suffix (str): Table suffix identifying timeframe or mode.

    Returns:
        None: Data is stored through side effects in the database.

    Raises:
        RuntimeError: Internal ``asyncpg`` or network failures may bubble up after logging.

    Edge Cases:
        If ``pool`` is created inside the function, it is closed automatically in ``finally``.
    """
    api = BinanceAPI()
    owns_pool = pool is None
    pool = pool or await get_db_pool()

    schema = CANDLES_DATA_SCHEMA
    table = f"{symbol.lower()}{table_suffix}"

    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    current_start = start_time
    total_candles = 0

    print(f"📊 Починаємо завантаження {symbol} ({timeframe}) за {days} днів...")
    print(f"📅 Період: з {start_time} по {end_time}")

    try:
        async with pool.acquire() as conn:
            while current_start < end_time:
                current_end = min(current_start + timedelta(days=5), end_time)
                print(f"🔍 Запит {symbol}: {current_start} - {current_end}")

                candles = api.fetch(
                    symbol=symbol,
                    start_time=int(current_start.timestamp() * 1000),
                    end_time=int(current_end.timestamp() * 1000),
                    interval=timeframe,
                )

                if candles:
                    await insert_candles(conn, schema, table, candles)
                    total_candles += len(candles)
                    print(f"📈 {symbol}: отримано {len(candles)} свічок")
                else:
                    print(f"⚠️ {symbol}: немає даних для періоду {current_start} - {current_end}")

                current_start = current_end
                await asyncio.sleep(API_REQUEST_PAUSE_SECONDS)

        print(f"✅ {symbol}: Завантажено {total_candles} свічок.")

    except Exception as exc:
        print(f"❌ Помилка для {symbol}: {exc}")
        import traceback

        traceback.print_exc()
    finally:
        api.close()
        if owns_pool and pool:
            await pool.close()


async def run_api(
    symbols: Optional[Sequence[str]] = None,
    timeframe: str = DEFAULT_TIMEFRAME,
    days: int = DEFAULT_LOOKBACK_DAYS,
) -> None:
    """Run sequential candle synchronization for all requested symbols.

    Args:
        symbols (Optional[Sequence[str]]): Custom list of pairs; defaults to ``TRADING_SYMBOLS``.
        timeframe (str): Binance request interval.
        days (int): Number of historical days to load for each symbol.

    Returns:
        None: The function finishes after all symbols are processed.

    Edge Cases:
        The created pool is always closed in ``finally`` even when an exception occurs.
    """
    symbols = list(symbols or TRADING_SYMBOLS)
    total = len(symbols)
    print(f"🔢 Загальна кількість символів: {total}")

    pool = await get_db_pool()
    try:
        for idx, symbol in enumerate(symbols, 1):
            print(f"\n🔄 Обробляємо символ {idx}/{total}: {symbol}")
            await fetch_and_store(symbol, timeframe, days, pool=pool)

            if idx < total:
                print(f"⏳ Пауза {SYMBOL_PAUSE_SECONDS} секунда між символами...")
                await asyncio.sleep(SYMBOL_PAUSE_SECONDS)

        print(f"\n🎉 Всі {total} символи успішно оброблено!")
    finally:
        await pool.close()
