import pytz
from decimal import Decimal
from datetime import timedelta
from utils import (
    get_db_pool,
    TRADING_SYMBOLS,
    SELL_DIRECTION,
    BUY_DIRECTION,
    CANDLES_SHEMA
)
from .order_actions import calculate_order_data
from .trader import check_balance
from bot import (
    open_order,
    get_open_positions,
)

from telegram_bot import send_pompilo_rain_order_message


async def get_candles(symbol, pool):
    """
        Отримує останні 8 годинних свічок для заданого символу з бази даних.

        Формує назву таблиці на основі символу та виконує SQL-запит для вибірки
        останніх 8 рядків, відсортованих за полем open_time у порядку спадання.

        Parameters:
            symbol (str): Назва торгового символу (наприклад, 'BTCUSDT').
            pool (asyncpg.pool.Pool): Пул підключень до бази даних PostgreSQL.

        Returns:
            List[Record]: Список останніх 8 свічок у форматі записів (Record) з бази даних.

        Raises:
            ValueError: Якщо символ містить недопустимі символи.
    """

    table_name = f"{symbol.lower().strip()}_p_candles"
    if not table_name.replace("_", "").isalnum():
        raise ValueError("Invalid symbol")

    query = f"""
        SELECT *
        FROM {CANDLES_SHEMA}.{str(symbol).lower().strip()}_p_candles
        ORDER BY open_time DESC
        LIMIT 12;
    """

    candles = await pool.fetch(query)
    return candles


def is_full_body_candle(candle):
    """
        Визначає, чи є свічка повнотілою (full body).

        Свічка вважається повнотілою, якщо співвідношення її тіла до загального діапазону
        (від high до low) є більшим або рівним 0.45.

        Parameters:
            candle (dict): Словник з ключами 'open', 'close', 'high', 'low'.
                Значення мають бути числовими (типи int, float або Decimal).

        Returns:
            bool: True, якщо свічка є повнотілою, інакше False.
    """

    candle_range = abs(candle['high'] - candle['low'])
    candle_body = abs(candle['close'] - candle['open'])

    if candle_body / candle_range >= 0.45:
        return True
    return False


def get_candle_type(candle):
    """
        Визначає тип свічки на основі її тіла та тіней.

        Аналізує співвідношення між тілом свічки та її тінями, щоб класифікувати
        її як один з наступних типів:
            - 'gravestone_doji': маленьке тіло з великою верхньою тінню;
            - 'dragonfly_doji': маленьке тіло з великою нижньою тінню;
            - 'doji': маленьке тіло з невеликими тінями;
            - 'simple_candle': звичайна свічка, яка не є до́жі.

        Parameters:
            candle (dict): Словник, що містить ключі 'open', 'close', 'high', 'low'.
                Значення мають бути типу, що сумісний із Decimal (наприклад, float, str або Decimal).

        Returns:
            str or None: Один із рядків 'gravestone_doji', 'dragonfly_doji', 'doji', 'simple_candle',
            або None, якщо свічка має нульовий діапазон (high == low).
    """

    high = candle['high']
    low = candle['low']
    open_ = candle['open']
    close = candle['close']

    body = Decimal(abs(close - open_))
    candle_range = Decimal(high - low)

    if Decimal(candle_range) == Decimal(0):
        return None  # захист від ділення на нуль

    body_ratio = Decimal(body / candle_range)

    upper_shadow = Decimal(high - max(open_, close))
    lower_shadow = Decimal(min(open_, close) - low)

    # Doji з переважно верхньою тінню → Gravestone Doji
    if (
            Decimal(body_ratio) < Decimal(0.25) and
            Decimal(upper_shadow) > Decimal(body) * Decimal(2) and
            Decimal(lower_shadow) * Decimal(2) < Decimal(upper_shadow) and
            (Decimal(lower_shadow) + Decimal(body)) < Decimal(upper_shadow)
    ):
        return 'gravestone_doji'

    # Doji з переважно нижньою тінню → Dragonfly Doji
    if (
            Decimal(body_ratio) < Decimal(0.25) and
            Decimal(lower_shadow) > Decimal(body) * Decimal(2) and
            Decimal(upper_shadow) * Decimal(2) < Decimal(lower_shadow) and
            (Decimal(upper_shadow) + Decimal(body)) < Decimal(lower_shadow)
    ):
        return 'dragonfly_doji'

    # Класичний Doji (маленьке тіло, малі тіні)
    if Decimal(body_ratio) < Decimal(0.25):
        return 'doji'

    print('simple_candle')
    return 'simple_candle'


def is_bullish_candle(candle):
    """
        Визначає, чи є свічка бичачою (тобто закриття вище відкриття).

        Порівнює значення 'close' і 'open' у переданій свічці.
        Якщо ціна закриття більша за ціну відкриття — це вважається бичачою свічкою.

        Parameters:
            candle (dict): Словник, що містить щонайменше ключі 'open' та 'close',
                значення яких можуть бути float, str або Decimal.

        Returns:
            bool: True, якщо свічка бичача (close > open), інакше False.
    """

    if Decimal(candle['close']) > Decimal(candle['open']):
        return True
    return False


async def generate_rain_signal(candles, symbol):
    """
        Генерує торговий сигнал за стратегією RAIN на основі останніх годинних свічок.

        Аналізуються останні 4 свічки з поданого списку `candles`, щоб виявити сигнали:
        - Розвороту (reversal) з урахуванням CVD, POC, VPOC-зони тощо.
        - Абсорбції (absorption) на великих обʼємах у зонах підтримки/опору.
        - Продовження тренду (trend) з підтвердженням повними свічками.
        - Трендові doji-сигнали (dragonfly/gravestone doji).

        Кожен тип сигналу повертається у вигляді текстового опису та відповідної позиції.

        Parameters:
            candles (list[dict]): Список з мінімум 4 свічок, упорядкованих від найсвіжішої до старішої.
                Кожна свічка повинна містити ключі:
                - 'open', 'close', 'high', 'low' (float або Decimal): ціни
                - 'volume' (float або Decimal): обсяг
                - 'cvd' (float або Decimal): кумулятивна дельта обʼєму
                - 'poc' (float або Decimal): point of control
                - 'vpoc_zone' (str): зона проторгованого обʼєму ('upper', 'middle', 'lower')
                - 'open_time' (datetime): час відкриття свічки

            symbol (str): Назва торгового інструменту (наприклад, 'BTCUSDT').

        Returns:
            dict: Словник з результатами аналізу:
                - 'signal' (str): текстовий опис сигналу або 'Order not opened' / 'Insufficient data'
                - 'position' (dict or None): словник з даними про позицію для відкриття або None, якщо сигналу немає.

        Notes:
            - Якщо дані не мають коректної часової послідовності (не годинні свічки) — повертається 'Insufficient data'.
            - Використовує допоміжні функції:
                - `is_full_body_candle()`
                - `get_candle_type()`
                - `is_bullish_candle()`
                - `detect_trend_direction()`
                - `is_strict_hourly_sequence()`
                - `calculate_position()`
    """

    candle_0h = candles[0]
    candle_1h = candles[1]
    candle_2h = candles[2]
    candle_3h = candles[3]

    candle_0h_full_body = is_full_body_candle(candle_0h)
    candle_1h_full_body = is_full_body_candle(candle_1h)
    candle_2h_full_body = is_full_body_candle(candle_2h)

    candle_0h_doji = get_candle_type(candle_0h)

    candle_0h_bullish = is_bullish_candle(candle_0h)
    candle_1h_bullish = is_bullish_candle(candle_1h)
    candle_2h_bullish = is_bullish_candle(candle_2h)

    volumes = [c['volume'] for c in candles]
    position = None
    signal = 'Order not opened'

    trend_direction = detect_trend_direction(candles)
    is_candles_correct = is_strict_hourly_sequence(candles)

    if not is_candles_correct:
        return {
            "signal": "Insufficient data",
            "position": None
        }

    # Reversal
    if (
            candle_1h_full_body and candle_0h_full_body and
            trend_direction == "down_trend" and
            candle_1h['close'] < candle_1h['open'] and
            candle_0h['close'] > candle_0h['open'] and
            candle_0h['close'] > candle_0h['poc'] > candle_0h['open'] and
            str(candle_0h['vpoc_zone']).lower() in ('middle', 'lower') and
            candle_1h['close'] < candle_0h['poc'] < candle_1h['open'] and
            candle_1h['vpoc_zone'] in ('middle', 'lower') and candle_0h['cvd'] > 0 and
            abs(Decimal(candle_0h['cvd'])) > Decimal('0.25') * abs(Decimal(candle_1h['cvd']))
    ):
        signal = '🔼 Buy reversal R'
        position = calculate_position(symbol, BUY_DIRECTION, Decimal(candle_0h['poc']), 'Limit')
    elif (
            candle_1h_full_body and candle_0h_full_body and
            trend_direction == "up_trend" and
            candle_1h['close'] > candle_1h['open'] and
            candle_0h['close'] < candle_0h['open'] and
            candle_0h['close'] < candle_0h['poc'] < candle_0h['open'] and
            str(candle_0h['vpoc_zone']).lower() in ('middle', 'upper') and
            candle_1h['close'] > candle_0h['poc'] > candle_1h['open'] and
            candle_1h['vpoc_zone'] in ('middle', 'upper') and candle_0h['cvd'] < 0 and
            abs(Decimal(candle_0h['cvd'])) > Decimal('0.25') * abs(Decimal(candle_1h['cvd']))
    ):
        signal = '🔽 Sell reversal R'
        position = calculate_position(symbol, SELL_DIRECTION, Decimal(candle_0h['poc']), 'Limit')
    # Absorption
    elif (
            candle_0h_full_body and
            candle_0h['volume'] == max(volumes) and
            trend_direction == "down_trend" and
            candle_0h['close'] < candle_0h['open'] and
            str(candle_0h['vpoc_zone']).lower() == 'lower' and
            candle_0h['close'] > candle_0h['poc']

    ):
        signal = '🔼 Buy absorption A'
        position = calculate_position(symbol, BUY_DIRECTION, Decimal(candle_0h['poc']), 'Limit')
    elif (
            candle_0h_full_body and
            candle_0h['volume'] == max(volumes) and
            trend_direction == "up_trend" and
            candle_0h['close'] > candle_0h['open'] and
            str(candle_0h['vpoc_zone']).lower() == 'upper' and
            candle_0h['close'] < candle_0h['poc']

    ):
        signal = '🔽 Sell absorption A'
        position = calculate_position(symbol, SELL_DIRECTION, Decimal(candle_0h['poc']), 'Limit')
    # Trend
    elif (
            candle_0h_full_body and candle_1h_full_body and candle_2h_full_body and
            candle_0h_bullish and not candle_1h_bullish and candle_2h_bullish and
            trend_direction == "up_trend" and
            candle_0h['open'] < candle_0h['poc'] < candle_0h['close'] and
            candle_1h['close'] < candle_1h['poc'] < candle_1h['open'] and
            str(candle_1h['vpoc_zone']).lower() in ('middle', 'lower') and
            candle_0h['cvd'] > 0 and candle_1h['volume'] != max(volumes)

    ):
        # candle_2h['open'] < candle_2h['poc'] < candle_2h['close'] and можливо додати в умову
        signal = '🔼 Buy trend T'
        position = calculate_position(symbol, BUY_DIRECTION, Decimal(candle_0h['close']), 'Market')
    elif (
            candle_0h_full_body and candle_1h_full_body and candle_2h_full_body and
            not candle_0h_bullish and candle_1h_bullish and not candle_2h_bullish and
            trend_direction == "down_trend" and
            candle_0h['close'] < candle_0h['poc'] < candle_0h['open'] and
            candle_1h['open'] < candle_1h['poc'] < candle_1h['close'] and
            #candle_2h['close'] < candle_2h['poc'] < candle_2h['open'] and
            str(candle_1h['vpoc_zone']).lower() in ('middle', 'upper') and
            candle_0h['cvd'] < 0 and candle_1h['volume'] != max(volumes)
    ):
        # candle_2h['open'] < candle_2h['poc'] < candle_2h['close'] and можливо додати в умову
        signal = '🔽 Sell trend T'
        position = calculate_position(symbol, SELL_DIRECTION, Decimal(candle_0h['close']), 'Market')
    elif (
            candle_1h_full_body and candle_0h_doji == 'dragonfly_doji' and
            candle_1h['open'] < candle_1h['poc'] < candle_1h['close'] and
            trend_direction == "up_trend" and
            candle_1h_bullish and candle_2h_bullish and
            str(candle_0h['vpoc_zone']).lower() in ('middle', 'lower') and
            candle_0h['poc'] < candle_0h['close'] and candle_0h['poc'] < candle_0h['open']
    ):
        signal = '🔼 Buy doji trend T'
        position = calculate_position(symbol, BUY_DIRECTION, Decimal(candle_0h['close']), 'Market')
    elif (
            candle_1h_full_body and candle_0h_doji == 'gravestone_doji' and
            candle_1h['open'] > candle_1h['poc'] > candle_1h['close'] and
            trend_direction == "down_trend" and
            not candle_1h_bullish and not candle_2h_bullish and
            str(candle_0h['vpoc_zone']).lower() in ('middle', 'upper') and
            candle_0h['poc'] > candle_0h['close'] and candle_0h['poc'] > candle_0h['open']
    ):
        signal = '🔽 Sell doji trend T'
        position = calculate_position(symbol, SELL_DIRECTION, Decimal(candle_0h['close']), 'Market')

    print(signal)
    print(f"position {position}, signal {signal}")

    return {
        "signal": signal,
        "position": position
    }


def is_strict_hourly_sequence(candles):
    """
        Перевіряє, чи всі передані свічки мають строго послідовний часовий інтервал у 1 годину між open_time.

        Функція бере список свічок та перевіряє, чи між кожною парою сусідніх значень поля 'open_time'
        інтервал точно становить 1 годину (тобто timedelta == 1 година).

        Умови:
            - Вважається, що список свічок відсортований у порядку зменшення часу (від новішої до старішої).
            - Перевіряється перші 5 свічок (4 інтервали між ними).

        Parameters:
            candles (list[dict]): Список свічок (мінімум 5), де кожна містить поле:
                - 'open_time' (datetime): час відкриття свічки (тип `datetime.datetime`)

        Returns:
            bool: True — якщо кожна наступна свічка віддалена від попередньої рівно на 1 годину,
                  False — якщо хоча б одна пара має інший інтервал.

    """

    open_times = [c['open_time'] for c in candles]
    for i in range(4):
        if open_times[i] - open_times[i + 1] != timedelta(hours=1):
            return False
    return True


def calculate_position(symbol, direction, entry_price, order_type='Limit'):
    """
        Розраховує параметри позиції для відкриття ордера на основі балансу та ціни входу.

        Використовує наявний баланс користувача для обчислення обсягу позиції,
        враховуючи напрямок (лонг або шорт), ціну входу та тип ордера.

        Parameters:
            symbol (str): Торговий символ (наприклад, 'BTCUSDT').
            direction (str): Напрямок угоди — 'buy' або 'sell'.
            entry_price (Decimal or float): Ціна, за якою відкривається позиція.
            order_type (str, optional): Тип ордера ('Limit' або 'Market'). За замовчуванням 'Limit'.

        Returns:
            dict: Словник із параметрами позиції.
    """

    balance = check_balance()
    position = calculate_order_data(symbol, direction, balance, Decimal(entry_price), order_type)

    return position


def detect_trend_direction(candles):
    """
    Визначає напрямок тренду на основі аналізу останніх 8 свічок.

    Алгоритм розглядає співвідношення бичачих і ведмежих свічок,
    загальний обʼєм, суму тіл свічок, а також зміну ціни від першої до останньої свічки.

    Умови:
        - Якщо ціна зросла більш ніж на 0.5% (від 1-ї до 8-ї свічки), і
          сукупні характеристики бичачих свічок (обʼєм, тіла) переважають — тренд "up_trend".
        - Якщо ціна впала більш ніж на 0.5%, і сукупні характеристики ведмежих свічок переважають — тренд "down_trend".
        - В інших випадках — "flat".

    Parameters:
        candles (list[dict]): Список із 8 свічок (словників), кожна з яких повинна містити поля:
            - 'open' (float): ціна відкриття
            - 'close' (float): ціна закриття
            - 'high' (float): максимальна ціна
            - 'low' (float): мінімальна ціна
            - 'cvd' (float): кумулятивна дельта обʼєму (не використовується напряму)
            - 'poc' (float): точка контролю (не використовується напряму)
            - 'vpoc_zone' (str): зона максимального обʼєму (не використовується напряму)
            - 'volume' (float): обʼєм торгів

    Returns:
        str: Один із трьох варіантів напрямку тренду:
            - "up_trend" — висхідний тренд
            - "down_trend" — низхідний тренд
            - "flat" — відсутність чітко вираженого тренду

    Примітка:
        Якщо передано менше ніж 5 свічок — метод повертає порожній рядок і виводить попередження.
    """

    bullish_candles = [c for c in candles if c['close'] > c['open']]
    bearish_candles = [c for c in candles if c['close'] < c['open']]

    bullish_volume = sum(c['volume'] for c in candles if c['close'] > c['open'])
    bearish_volume = sum(c['volume'] for c in candles if c['close'] < c['open'])

    bullish_body_sum = sum(abs(c['close'] - c['open']) for c in bullish_candles)
    bearish_body_sum = sum(abs(c['close'] - c['open']) for c in bearish_candles)

    closes = [c['close'] for c in candles]
    change_pct = abs((closes[-1] - closes[0]) / closes[0] * 100)

    if len(candles) < 8:
        print("Недостатньо даних")
        return ''

    closes = [c['close'] for c in candles]

    if (
            change_pct > 0.5 and
            closes[0] > closes[7] and
            bullish_volume > bearish_volume and
            bullish_body_sum > bearish_body_sum
    ):
        return "up_trend"
    elif (
            change_pct > 0.5 and
            closes[0] < closes[7] and
            bullish_volume < bearish_volume and
            bullish_body_sum < bearish_body_sum
    ):
        return "down_trend"
    else:
        return "flat"


async def open_rain_position(position, signal, symbol):
    """
        Відкриває нову позицію за сигналом RAIN або перевідкриває її, якщо вже існує протилежна позиція.

        Функція перевіряє, чи є вже відкрита позиція для заданого символу:
        - Якщо немає жодної активної позиції — відкриває нову згідно з параметрами `position`.
        - Якщо вже є відкрита позиція в протилежному напрямку — закриває її (через відкриття нової на сумарний обсяг)
          і відкриває нову позицію з поточними параметрами.
        - Якщо вже є позиція в тому ж напрямку — нічого не робить.

        Після відкриття нової позиції надсилає повідомлення про дію через `send_pompilo_rain_order_message`.

        Parameters:
            position (dict): Дані про позицію, що має бути відкрита. Очікуються поля:
                - 'symbol': str — торговий символ (наприклад, 'SOLUSDT')
                - 'direction': str — напрямок позиції ('Buy' або 'Sell')
                - 'size': str або Decimal — розмір позиції
                - 'stop_loss': float — ціна стоп-лосу
                - 'take_profit': float — ціна тейк-профіту
                - 'order_type': str — тип ордера ('Limit' або 'Market')
                - 'price': float — ціна (для лімітного ордера)
            signal (str): Назва або опис сигналу, який ініціював відкриття позиції.
            symbol (str): Символ, по якому аналізуються відкриті позиції (може дублюватися з position['symbol']).

        Returns:
            None
    """

    if not position:
        return

    opened_positions = get_open_positions(symbol)
    active_positions = [
        p for p in opened_positions
        if Decimal(p['size']) > 0 and p['symbol'].upper() == position['symbol'].upper()
    ]

    # Якщо немає активних позицій — відкриваємо нову
    if not active_positions:
        print(f"[OPEN NEW POSITION]: {signal}")
        open_order(
            position['symbol'],
            position['direction'],
            Decimal(position['size']),
            position['stop_loss'],
            position['take_profit'],
            "Limit" if position['order_type'] == 'Limit' else "Market",
            position['price'] if position['order_type'] == 'Limit' else None
        )

        await send_pompilo_rain_order_message(
            position['symbol'],
            position['price'],
            position['take_profit'],
            position['stop_loss'],
            position['direction'],
            signal
        )
        return

        # Є вже відкрита позиція — перевірити напрямок
    existing = active_positions[0]
    if str(existing['direction']).lower() != str(position['direction']).lower():

        print(f"[CLOSE OPPOSITE POSITION]: {existing}")
        order_size = Decimal(position['size']) + Decimal(existing['size'])

        open_order(
            position['symbol'],
            position['direction'],
            order_size,
            position['stop_loss'],
            position['take_profit'],
            "Limit" if position['order_type'] == 'Limit' else "Market",
            position['price'] if position['order_type'] == 'Limit' else None
        )
        print(f"[OPEN NEW POSITION AFTER CLOSE]: {signal}")

        await send_pompilo_rain_order_message(
            position['symbol'],
            position['price'],
            position['take_profit'],
            position['stop_loss'],
            position['direction'],
            signal
        )
    else:
        print(f"[SKIP]: Already have same direction position for {position['symbol']}")


async def start_rain_signal_generator():
    """
        Запускає генератор сигналів за стратегією RAIN для всіх торгових символів.

        Отримує підключення до бази даних, проходить по списку торгових пар (TRADING_SYMBOLS),
        отримує останні свічки для кожної пари, генерує торговий сигнал на основі цих даних
        та, у разі виявлення сигнальної позиції, відкриває її.

        Після завершення роботи закриває пул підключення до бази даних.

        Returns:
            None
    """

    print('Start rain signal generator')
    pool = await get_db_pool()

    for symbol in TRADING_SYMBOLS:
        print(f"Search signal on {symbol}")
        rows = await get_candles(symbol, pool)
        if len(rows) >= 5:
            signal_data = await generate_rain_signal(rows, symbol)
            await open_rain_position(signal_data['position'], signal_data['signal'], symbol)

    await pool.close()
