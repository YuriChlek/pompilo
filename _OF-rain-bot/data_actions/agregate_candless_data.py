import asyncpg
import asyncio
import zoneinfo
from datetime import datetime, timedelta
from collections import defaultdict
from decimal import Decimal
from utils import (
    SCHEMAS,
    SELL_DIRECTION,
    TRADING_SYMBOLS,
    get_db_pool,
    insert_candles_data
)


def get_hour_window(target_time: datetime, current_time=False):
    """
        Обчислює годинне вікно часу (start і end) для заданого моменту часу.

        Перетворює переданий час у київський часовий пояс (Europe/Kyiv), визначає початок і кінець відповідної години:
        - Якщо `current_time=False`, то обирається попередня завершена година.
        - Якщо `current_time=True`, то обирається поточна або наступна година.

        Повертає час без tzinfo (naive UTC).

        Parameters:
            target_time (datetime): Початковий час, за яким визначається годинне вікно.
            current_time (bool): Якщо True, повертає поточне годинне вікно; інакше — попереднє.

        Returns:
            dict: Словник із двома ключами:
                - "start" (datetime): Початок години (naive UTC).
                - "end" (datetime): Кінець години (naive UTC).
    """
    kyiv = zoneinfo.ZoneInfo("Europe/Kyiv")

    # Якщо target_time не має tzinfo — вважаємо, що це UTC
    if target_time.tzinfo is None and not current_time:
        target_time = target_time.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))

    # Переводимо в київський час
    local_time = target_time.astimezone(kyiv)

    # Зменшуємо на 1 годину — бо хочемо завершену попередню годину
    if not current_time:
        local_time -= timedelta(hours=1)
    else:
        local_time += timedelta(hours=1)

    # Обрізаємо до початку години
    start_time = local_time.replace(minute=0, second=0, microsecond=0)
    end_time = start_time + timedelta(hours=1)

    start_utc_naive = start_time.replace(tzinfo=None)
    end_utc_naive = end_time.replace(tzinfo=None)

    return {
        "start": start_utc_naive,
        "end": end_utc_naive
    }


def compute_vpoc_zone(trading_data: list[dict], high, low) -> str | None:
    """
       Визначає зону найбільшого обсягу (VPOC zone) у межах свічки, поділеної на три частини.

       Поділяє ціновий діапазон між low і high на три рівні зони: lower, middle, upper.
       Підраховує загальний обсяг торгів у кожній зоні та повертає ту, де обсяг найбільший.

       Parameters:
           trading_data (list[dict]): Список тикових трейдів із ключами "price" і "size".
           high (float): Максимальна ціна свічки.
           low (float): Мінімальна ціна свічки.

       Returns:
           str | None: Назва зони з найбільшим обсягом — "lower", "middle" або "upper".
                       Якщо ціна не змінювалась, повертає "middle".
    """
    range_size = high - low

    if range_size == 0:
        return "middle"  # ціна не змінювалася

    # Межі третин
    lower_bound = low + range_size / 3
    upper_bound = low + 2 * range_size / 3

    zone_volumes = {"lower": 0.0, "middle": 0.0, "upper": 0.0}

    for row in trading_data:
        price = float(row["price"])
        size = float(row["size"])

        if price < lower_bound:
            zone_volumes["lower"] += size
        elif price < upper_bound:
            zone_volumes["middle"] += size
        else:
            zone_volumes["upper"] += size

    # Визначаємо зону з найбільшим обсягом
    dominant_zone = max(zone_volumes.items(), key=lambda item: item[1])[0]
    return dominant_zone


def calculate_cvd(rows):
    """
    Обчислює кумулятивну дельту об'єму (CVD) на основі списку тикових трейдів.

    CVD (Cumulative Volume Delta) розраховується як різниця між об'ємом купівельних
    і продажних ордерів у переданих рядках. Кожен рядок повинен містити поля 'side' та 'size'.

    Parameters:
        rows (list[dict]): Список тикових трейдів. Кожен елемент має містити ключі:
                           - 'side' (str): напрям ордера ("buy" або "sell"),
                           - 'size' (str або Decimal): обʼєм ордера.

    Returns:
        Decimal: Округлене значення CVD (buy_volume - sell_volume).
    """
    buy_orders = Decimal(0)
    sell_orders = Decimal(0)

    for row in rows:
        size = Decimal(row['size'])
        if str(row['side']).lower() == SELL_DIRECTION:
            sell_orders += size
        else:
            buy_orders += size

    return round(buy_orders - sell_orders, 0)


def compute_poc(trading_data: list[dict]) -> float | None:
    """
        Обчислює Point of Control (POC) — ціну з найбільшим проторгованим обсягом.

        POC (Point of Control) — це рівень ціни, на якому був зафіксований найбільший
        обсяг торгів за вхідними тиковими даними. Цей метод агрегує обсяги за округленими
        цінами і повертає ціну з максимальним сумарним обсягом.

        Parameters:
            trading_data (list[dict]): Список тикових трейдів, кожен з яких має містити:
                - "price" (str або float): ціна виконання ордера,
                - "size" (str або float): обсяг виконаного ордера.

        Returns:
            float | None: Ціна з найбільшим обсягом (POC), або None, якщо список порожній.
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
    """
        Обчислює OHLC-свічку та супутні метрики з тикових трейдів.

        Метод будує агреговану свічку за списком тикових даних, розраховуючи:
        - Open: перша ціна в серії;
        - Close: остання ціна в серії;
        - High / Low: найвища та найнижча ціна відповідно;
        - Volume: сумарний обсяг;
        - CVD: кумулятивна різниця між обсягами купівлі та продажу;
        - POC: ціна з найбільшим обсягом;
        - VPOC zone: зона найактивнішого обсягу торгів.

        Parameters:
            trading_data (list[dict]): Список тикових даних. Кожен елемент має містити:
                - "price" (str або float): ціна виконання угоди;
                - "size" (str або float): обсяг угоди;
                - "side" (str): напрямок угоди — "buy" або "sell".

        Returns:
            dict | None: Словник з полями "open", "close", "high", "low", "volume",
            "cvd", "poc", "vpoc_zone" або None, якщо список порожній.
    """
    if not trading_data:
        return None

    prices = [float(row["price"]) for row in trading_data]
    sizes = [float(row["size"]) for row in trading_data]

    high = max(prices)
    low = min(prices)

    cvd = calculate_cvd(trading_data)
    poc = compute_poc(trading_data)
    vpoc_zone = compute_vpoc_zone(trading_data, high, low)

    return {
        "open": prices[0],
        "close": prices[-1],
        "high": high,
        "low": low,
        "cvd": cvd,
        "poc": poc,
        "vpoc_zone": vpoc_zone,
        "volume": round(sum(sizes)),
    }


async def fetch_hourly_tick_data(pool, target_time: datetime):
    """
        Завантажує тикові дані для заданої години з кількох бірж та обчислює свічки.

        Ця асинхронна функція виконує агрегацію тикових трейдів для кожного символу
        з визначеного списку TRADING_SYMBOLS, об'єднуючи дані з п'яти бірж:
        Bybit, Binance, Bitget, OKX та Gate.io. Після вибірки даних за годинне вікно,
        будується агрегована свічка (OHLC + обсяг, POC, CVD, VPOC зона) для кожного символу.

        Parameters:
            pool: Асинхронний пул з'єднань з PostgreSQL базою даних (наприклад, від asyncpg).
            target_time (datetime): Момент часу, для якого потрібно отримати годинне вікно.

        Returns:
            list[dict]: Список словників, кожен з яких містить:
                - "start_time": початок годинного вікна (datetime);
                - "end_time": кінець годинного вікна (datetime);
                - "symbol": символ трейдингу (str);
                - "candle": словник зі значеннями open, high, low, close, volume, cvd, poc, vpoc_zone;
                - "trading_data": список тикових трейдів (list[dict]) з полями "side", "size", "price", "timestamp".

        Raises:
            Exception: У разі помилок під час вибірки даних із БД помилки логуються, але не припиняють виконання.
    """
    window = get_hour_window(target_time)
    start_time = window["start"]
    end_time = window["end"]

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
    """
        Отримує список початкових моментів кожної години з бази даних, починаючи від найдавнішого запису.

        Функція асинхронно звертається до вказаної таблиці в базі даних PostgreSQL,
        знаходить найменший час (MIN) у вказаній колонці з часовими мітками,
        перетворює його у часовий пояс Europe/Kyiv, і генерує список datetime об'єктів
        з кроком в 1 годину до поточної години включно.

        Parameters:
            pool: Асинхронний пул з'єднань до бази даних (наприклад, створений за допомогою asyncpg).
            table_schema (str): Назва схеми, в якій знаходиться таблиця.
            table_name (str): Назва таблиці, з якої потрібно витягти дані.
            timestamp_column (str, optional): Назва колонки, що містить часові мітки.
                За замовчуванням — "timestamp".

        Returns:
            list[datetime]: Список об'єктів datetime (без tzinfo), кожен з яких позначає початок повної години
            в часовому поясі Europe/Kyiv, починаючи з першого наявного запису до поточного моменту.

        Raises:
            Exception: У випадку помилки під час запиту до бази даних, повідомлення логуються,
            а функція повертає порожній список.
    """
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

            if result.tzinfo is None:
                result = result.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))

            result_kyiv = result.astimezone(kyiv)
            start = result_kyiv.replace(minute=0, second=0, microsecond=0)
            now_kyiv = datetime.now(kyiv).replace(minute=0, second=0, microsecond=0)

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


async def set_candles_data(hourly_ranges):
    """
        Генерує та зберігає годинні свічки на основі тикових даних для заданих годинних інтервалів.

        Функція виконує такі кроки:
        1. Отримує з'єднання з базою даних.
        2. Для кожного переданого годинного інтервалу викликає функцію збору тикових даних з кількох бірж.
        3. Агрегує ці дані в одну свічку (open, high, low, close, volume, CVD, POC, VPOC-зону).
        4. Виводить інформацію про згенеровану свічку в консоль.
        5. Вставляє згенеровані свічки у відповідну таблицю бази даних за символом.

        Parameters:
            hourly_ranges (list[datetime]): Список об'єктів datetime, що позначають початок кожного годинного інтервалу
            (кожен елемент — початок години, кінець визначається автоматично як +1 година).

        Returns:
            None: Результати записуються в базу даних та виводяться в консоль, значення не повертається.

        Raises:
            Exception: Помилки, які виникають під час доступу до бази або обробки даних, можуть бути виведені у консоль.
    """
    pool = await get_db_pool()

    for hour in hourly_ranges:
        results = await fetch_hourly_tick_data(pool, hour)
        for entry in results:
            candle = entry["candle"]
            if candle:
                print(
                    f"{entry['symbol']} | {entry['start_time']} - {entry['end_time']} | "
                    f"Open: {candle['open']} Close: {candle['close']} "
                    f"High: {candle['high']} Low: {candle['low']} "
                    f"Volume: {candle['volume']}, CVD: {candle['cvd']} POC: {candle['poc']}"
                )
                open_time = entry['start_time']
                close_time = entry['end_time']
                symbol = entry['symbol']
                open_price = candle['open']
                close_price = candle['close']
                high_price = candle['high']
                low_price = candle['low']
                cvd = candle['cvd']
                poc = candle['poc']
                poc_zone = candle['vpoc_zone']
                volume = candle['volume']
                candle_id = f"{open_time}"
                await insert_candles_data(pool, (
                    open_time,
                    close_time,
                    symbol,
                    open_price,
                    close_price,
                    high_price,
                    low_price,
                    cvd,
                    poc,
                    poc_zone,
                    volume,
                    candle_id
                ), symbol)
            else:
                print(f"{entry['symbol']} | {entry['start_time']} - {entry['end_time']} | No data")
        print('')

    await pool.close()


async def run_agregate_all_candles_data_job():
    """
        Запускає повний процес агрегації годинних свічок з тикових даних.

        Функція виконує такі кроки:
        1. Встановлює з'єднання з базою даних.
        2. Визначає часові інтервали для агрегації на основі найранішого запису у вибраній таблиці.
        3. Закриває з'єднання з базою даних.
        4. Викликає агрегацію свічок для кожного годинного інтервалу з отриманого списку.

        Ця функція є точкою входу для побудови історичних або поточних свічок на основі тикових трейдів.

        Parameters:
            None

        Returns:
            None
    """
    pool = await get_db_pool()

    table_schema = SCHEMAS[1]
    table_name = f"{str(TRADING_SYMBOLS[0]).lower()}_p_trades"
    hourly_ranges = await get_hourly_time_ranges_from_db(pool, table_schema, table_name)
    #hourly_ranges = await get_hourly_time_ranges_from_db(pool, 'binance_trading_history_data', "solusdt_p_trades")
    await pool.close()

    await set_candles_data(hourly_ranges)


async def run_agregate_last_1h_candles_data_job():
    """
        Виконує агрегацію свічки за останню годину на основі тикових даних.

        Функція визначає поточну годину за київським часом і передає її
        у вигляді списку до методу `set_candles_data`, який обчислює
        відповідну свічку з торгової історії.

        Args:
            None

        Returns:
            None
    """
    print("Agregate last hour candle.")
    kyiv = zoneinfo.ZoneInfo("Europe/Kyiv")
    now = [datetime.now(kyiv)]

    await set_candles_data(now)
