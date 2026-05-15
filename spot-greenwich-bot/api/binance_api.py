from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Iterable, List, Optional, Sequence

import asyncpg
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.config import BINANCE_D1_INTERVAL, BINANCE_REST_ENDPOINT, CANDLES_DATA_SCHEMA, DEFAULT_LOOKBACK_DAYS, D1_TABLE_SUFFIX, SPOT_TRADING_SYMBOLS
from utils.db_actions import d1_table_name, get_db_pool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candle:
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
        return self.volume if self.close > self.open else -self.volume

    @staticmethod
    def from_kline(symbol: str, payload: Sequence[float]) -> "Candle":
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


class BinanceAPI:
    def __init__(self, rest_endpoint: str = BINANCE_REST_ENDPOINT):
        self.endpoint = rest_endpoint
        self.session = requests.Session()
        self.count = 1000
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def fetch(self, symbol: str, start_time: int, end_time: int, interval: str = BINANCE_D1_INTERVAL) -> List[Candle]:
        response = self.session.get(
            f"{self.endpoint}/api/v3/klines",
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": start_time,
                "endTime": end_time,
                "limit": self.count,
            },
            timeout=30,
        )
        response.raise_for_status()
        return [Candle.from_kline(symbol, payload) for payload in response.json()]

    def close(self) -> None:
        self.session.close()


def _build_candle_records(candles: Iterable[Candle]) -> list[tuple]:
    # CVD is retained only for backward-compatible storage; strategy code uses volume.
    cvd = Decimal("0")
    records = []
    for candle in candles:
        cvd += Decimal(str(candle.delta))
        records.append(
            (
                candle.open_time,
                candle.close_time,
                candle.symbol,
                candle.open,
                candle.close,
                candle.high,
                candle.low,
                cvd,
                round(Decimal(str(candle.volume)), 8),
                str(candle.open_time),
            )
        )
    return records


async def insert_candles(conn: asyncpg.Connection, table: str, candles: Sequence[Candle]) -> None:
    records = _build_candle_records(candles)
    if not records:
        return

    sql = f"""
    INSERT INTO {CANDLES_DATA_SCHEMA}.{table} (
        open_time, close_time, symbol, open, close, high, low, cvd, volume, candle_id
    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
    ON CONFLICT (candle_id) DO UPDATE
    SET open = EXCLUDED.open,
        close = EXCLUDED.close,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        cvd = EXCLUDED.cvd,
        volume = EXCLUDED.volume,
        close_time = EXCLUDED.close_time;
    """
    await conn.executemany(sql, records)
    logger.info("candles_saved schema=%s table=%s count=%s", CANDLES_DATA_SCHEMA, table, len(records))


async def fetch_and_store(
    symbol: str,
    timeframe: str = BINANCE_D1_INTERVAL,
    days: int = DEFAULT_LOOKBACK_DAYS,
    *,
    pool: Optional[asyncpg.Pool] = None,
    table_suffix: str = D1_TABLE_SUFFIX,
) -> None:
    api = BinanceAPI()
    owns_pool = pool is None
    pool = pool or await get_db_pool()
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    current_start = start_time
    chunk_days = 365
    table_name = f"{symbol.lower()}{table_suffix}"
    total_candles = 0

    logger.info("candle_sync_started symbol=%s timeframe=%s days=%s start=%s end=%s", symbol, timeframe, days, start_time, end_time)

    try:
        async with pool.acquire() as conn:
            while current_start < end_time:
                current_end = min(current_start + timedelta(days=chunk_days), end_time)
                logger.info("candles_requesting symbol=%s start=%s end=%s", symbol, current_start, current_end)
                candles = api.fetch(
                    symbol=symbol,
                    start_time=int(current_start.timestamp() * 1000),
                    end_time=int(current_end.timestamp() * 1000),
                    interval=timeframe,
                )
                if candles:
                    await insert_candles(conn, table_name, candles)
                    total_candles += len(candles)
                    logger.info("candles_fetched symbol=%s count=%s", symbol, len(candles))
                else:
                    logger.warning("candles_empty symbol=%s start=%s end=%s", symbol, current_start, current_end)
                current_start = current_end
                await asyncio.sleep(0.2)
        logger.info("candle_sync_completed symbol=%s total=%s", symbol, total_candles)
    finally:
        api.close()
        if owns_pool:
            await pool.close()


async def run_api(
    symbols: Optional[Sequence[str]] = None,
    timeframe: str = BINANCE_D1_INTERVAL,
    days: int = DEFAULT_LOOKBACK_DAYS,
    *,
    table_suffix: str = D1_TABLE_SUFFIX,
) -> None:
    selected_symbols = list(symbols or SPOT_TRADING_SYMBOLS)
    total = len(selected_symbols)
    logger.info("sync_batch_started symbols=%s total=%s timeframe=%s days=%s", ",".join(selected_symbols), total, timeframe, days)
    pool = await get_db_pool()
    try:
        for index, symbol in enumerate(selected_symbols, start=1):
            logger.info("sync_symbol_started index=%s total=%s symbol=%s", index, total, symbol)
            await fetch_and_store(
                symbol,
                timeframe=timeframe,
                days=days,
                pool=pool,
                table_suffix=table_suffix,
            )
            if index < total:
                logger.info("sync_symbol_pause seconds=0.5")
                await asyncio.sleep(0.5)
        logger.info("sync_batch_completed total=%s", total)
    finally:
        await pool.close()
