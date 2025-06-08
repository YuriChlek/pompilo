import asyncpg
import asyncio
import zoneinfo
from datetime import datetime, timedelta
from collections import defaultdict
from decimal import Decimal
from utils import SELL_DIRECTION, SCHEMAS, TRADING_SYMBOLS, get_db_pool
from .data_processor import calculate_cvd


def get_hour_window(target_time: datetime):
    kyiv = zoneinfo.ZoneInfo("Europe/Kyiv")

    # Якщо target_time не має tzinfo — вважаємо, що це UTC
    if target_time.tzinfo is None:
        target_time = target_time.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))

    # Переводимо в київський час
    local_time = target_time.astimezone(kyiv)

    # Зменшуємо на 1 годину — бо хочемо завершену попередню годину
    local_time -= timedelta(hours=1)

    # Обрізаємо до початку години
    start_time = local_time.replace(minute=0, second=0, microsecond=0)
    end_time = start_time + timedelta(hours=1)

    start_utc_naive = start_time.replace(tzinfo=None)
    end_utc_naive = end_time.replace(tzinfo=None)

    print("Start time:", start_utc_naive)
    print("End time:", end_utc_naive)

    return start_utc_naive, end_utc_naive


def compute_poc(trading_data: list[dict]) -> float | None:
    """
    Обчислює POC (Point of Control) — ціну з найбільшим наторгованим обсягом.

    :param trading_data: Список тикових трейдів, кожен — dict з ключами 'price' і 'size'
    :return: Ціна з найбільшим volume (POC), або None, якщо список порожній
    """
    if not trading_data:
        return None

    price_volume_map = defaultdict(float)

    for row in trading_data:
        price = round(float(row["price"]), 4)  # округлення для агрегації
        size = float(row["size"])
        price_volume_map[price] += size

    # Знаходимо ціну з найбільшим обсягом
    poc_price = max(price_volume_map.items(), key=lambda item: item[1])[0]
    return poc_price


def compute_candle(trading_data: list[dict]):
    if not trading_data:
        return None

    prices = [float(row["price"]) for row in trading_data]
    sizes = [float(row["size"]) for row in trading_data]
    cvd = calculate_cvd(trading_data)
    poc = compute_poc(trading_data)

    return {
        "open": prices[0],
        "close": prices[-1],
        "high": max(prices),
        "low": min(prices),
        "volume": sum(sizes),
        "cvd": cvd,
        "poc": poc,
    }


async def fetch_hourly_tick_data(pool, target_time: datetime):
    start_time, end_time = get_hour_window(target_time)
    all_results = []

    async with pool.acquire() as conn:
        for symbol in TRADING_SYMBOLS:
            table_name = f"{symbol.lower()}_p_trades"
            query = f"""
                SELECT side, size, price, timestamp 
                FROM (
                    SELECT side, size, price, timestamp 
                    FROM bybit_trading_history_data.{table_name}
                    WHERE timestamp >= $1 AND timestamp < $2

                    UNION ALL

                    SELECT side, size, price, timestamp 
                    FROM binance_trading_history_data.{table_name}
                    WHERE timestamp >= $1 AND timestamp < $2

                    UNION ALL

                    SELECT side, size, price, timestamp 
                    FROM bitget_trading_history_data.{table_name}
                    WHERE timestamp >= $1 AND timestamp < $2

                    UNION ALL

                    SELECT side, size, price, timestamp 
                    FROM okx_trading_history_data.{table_name}
                    WHERE timestamp >= $1 AND timestamp < $2

                    UNION ALL

                    SELECT side, size, price, timestamp 
                    FROM gateio_trading_history_data.{table_name}
                    WHERE timestamp >= $1 AND timestamp < $2
                ) AS combined
                ORDER BY timestamp ASC;
            """
            try:
                rows = await conn.fetch(query, start_time, end_time)
                trading_data = [dict(row) for row in rows]
                candle = compute_candle(trading_data)

                all_results.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "symbol": symbol,
                    "candle": candle,
                    "trading_data": trading_data
                })
            except Exception as e:
                print(f"Error fetching data from {table_name}: {e}")

    return all_results


async def get_hourly_time_ranges_from_db(pool, table_schema: str, table_name: str, timestamp_column: str = "timestamp"):
    kyiv = zoneinfo.ZoneInfo("Europe/Kyiv")

    async with pool.acquire() as conn:
        try:
            query = f"""
                SELECT MIN({timestamp_column}) as earliest_time
                FROM {table_schema}.{table_name}
            """
            result = await conn.fetchval(query)

            if result is None:
                print(f"No data in table {table_schema}.{table_name}")
                return []

            # Встановлюємо часовий пояс (якщо він відсутній — вважаємо UTC)
            if result.tzinfo is None:
                result = result.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))

            # Конвертуємо в Київський час
            result_kyiv = result.astimezone(kyiv)

            # Обрізаємо до початку години
            start = result_kyiv.replace(minute=0, second=0, microsecond=0)

            # Поточний час у Києві, теж обрізаний
            now_kyiv = datetime.now(kyiv).replace(minute=0, second=0, microsecond=0)

            # Перетворюємо на наївні datetime (без tzinfo)
            current = start.replace(tzinfo=None)
            end = now_kyiv.replace(tzinfo=None)

            ranges = []
            while current < end:
                ranges.append(current)
                current += timedelta(hours=1)

            return ranges

        except Exception as e:
            print(f"[ERROR] Failed to get earliest time from {table_schema}.{table_name}: {e}")
            return []


async def run_agregate_candles_data_job():
    pool = await get_db_pool()

    kyiv = zoneinfo.ZoneInfo("Europe/Kyiv")
    now = datetime.now(kyiv)

    print(now, 'now')

    results = await fetch_hourly_tick_data(pool, now)

    for entry in results:
        candle = entry["candle"]
        if candle:
            print(
                f"{entry['symbol']} | {entry['start_time']} - {entry['end_time']} | "
                f"Open: {candle['open']} Close: {candle['close']} "
                f"High: {candle['high']} Low: {candle['low']} "
                f"Volume: {candle['volume']}, CVD: {candle['cvd']} POC: {candle['poc']}"
            )
        else:
            print(f"{entry['symbol']} | {entry['start_time']} - {entry['end_time']} | No data")

    table_schema = SCHEMAS[0]
    table_name = f"{str(TRADING_SYMBOLS[0]).lower()}_p_trades"  # Приклад

    hourly_ranges = await get_hourly_time_ranges_from_db(pool, table_schema, table_name)

    # for hour in hourly_ranges:
    # print(hour)

    await pool.close()


if __name__ == "__main__":
    asyncio.run(run_agregate_candles_data_job())
