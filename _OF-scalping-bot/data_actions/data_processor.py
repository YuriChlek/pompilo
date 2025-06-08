from datetime import datetime, timedelta
from decimal import Decimal

from utils import (
    get_db_pool,
    EXTREMES_PERIOD,
    IMBALANCE_AND_CVD_PERIOD,
    SYMBOLS_ROUNDING,
    MIN_BIG_TRADES_SIZES,
    SELL_DIRECTION,
    BUY_DIRECTION
)


async def find_max_size(symbol):
    """
        Метод для пошуку великих ордерів у базі даних,
        які можуть призвести до руху ціни
        (Для тестування)
    """
    conn = await get_db_pool()

    query = f"""
            SELECT timestamp, side, size, price
            FROM (
                SELECT timestamp, side, size, price FROM bybit_trading_history_data.{str(symbol).lower()}_p_trades
                UNION ALL
                SELECT timestamp, side, size, price FROM binance_trading_history_data.{str(symbol).lower()}_p_trades
                UNION ALL
                SELECT timestamp, side, size, price FROM bitget_trading_history_data.{str(symbol).lower()}_p_trades
                UNION ALL
                SELECT timestamp, side, size, price FROM okx_trading_history_data.{str(symbol).lower()}_p_trades
                UNION ALL
                SELECT timestamp, side, size, price FROM gateio_trading_history_data.{str(symbol).lower()}_p_trades
            ) AS all_data
            ORDER BY size DESC
            LIMIT 1;
        """

    result = await conn.fetchrow(query)
    max_size = result['size']
    max_size_timestamp = result['timestamp']
    direction = result['side']
    price = result['price']

    await conn.close()

    return [max_size_timestamp,
            round(float(max_size), SYMBOLS_ROUNDING[symbol]),
            direction,
            round(float(price), SYMBOLS_ROUNDING[symbol])]


def calculate_imbalance(rows):
    """
    Метод який розраховує імбаланс між Sell та Buy

    :param rows:
    :return:
    """
    buy_orders = Decimal(0)
    sell_orders = Decimal(0)

    for row in rows:
        size = abs(Decimal(str(row['size'])))
        side = str(row['side']).lower()

        if side.strip() == SELL_DIRECTION:
            sell_orders += size
        else:
            buy_orders += size

    total = buy_orders + sell_orders

    if total == 0:
        return Decimal("0.0")

    return round((buy_orders - sell_orders) / total, 4)


def calculate_cvd(rows):
    """
    Метод для розрахунку CVD
    :param rows:
    :return:
    """
    buy_orders = Decimal(0)
    sell_orders = Decimal(0)

    for row in rows:
        size = abs(Decimal(str(row['size'])))
        side = str(row['side']).lower()

        if side.strip() == SELL_DIRECTION:
            sell_orders += size
        else:
            buy_orders += size

    return round(buy_orders - sell_orders, 0)


async def get_cvd_and_imbalance(reference_time: datetime, symbol, period=None):
    conn = await get_db_pool()

    cvd_imbalance_period = period if period else IMBALANCE_AND_CVD_PERIOD
    start_time = reference_time - timedelta(hours=cvd_imbalance_period)

    query = f"""
            SELECT side, size, price, timestamp 
            FROM (
                SELECT side, size, price, timestamp 
            FROM bybit_trading_history_data.{str(symbol).lower()}_p_trades
            WHERE timestamp >= $1 AND timestamp <= $2

            UNION ALL

            SELECT side, size, price, timestamp 
            FROM binance_trading_history_data.{str(symbol).lower()}_p_trades
            WHERE timestamp >= $1 AND timestamp <= $2
            
            UNION ALL

            SELECT side, size, price, timestamp 
            FROM bitget_trading_history_data.{str(symbol).lower()}_p_trades
            WHERE timestamp >= $1 AND timestamp <= $2
            
            UNION ALL
            
            SELECT side, size, price, timestamp 
            FROM okx_trading_history_data.{str(symbol).lower()}_p_trades
            WHERE timestamp >= $1 AND timestamp <= $2
            
            UNION ALL
            
            SELECT side, size, price, timestamp 
            FROM gateio_trading_history_data.{str(symbol).lower()}_p_trades
            WHERE timestamp >= $1 AND timestamp <= $2
            ) AS combined
            ORDER BY timestamp ASC;
        """

    rows = await conn.fetch(query, start_time, reference_time)
    imbalance = calculate_imbalance(rows)
    cvd = calculate_cvd(rows)

    await conn.close()

    return {
        "cvd": cvd,
        "imbalance": imbalance,
    }


async def get_support_resistance_cvd(reference_time: datetime, symbol, big_order_price):
    conn = await get_db_pool()
    start_time = reference_time - timedelta(hours=4)
    query = f"""
                SELECT side, size, price, timestamp 
                FROM (
                    SELECT side, size, price, timestamp 
                FROM bybit_trading_history_data.{str(symbol).lower()}_p_trades
                WHERE timestamp >= $1 AND timestamp <= $2

                UNION ALL

                SELECT side, size, price, timestamp 
                FROM binance_trading_history_data.{str(symbol).lower()}_p_trades
                WHERE timestamp >= $1 AND timestamp <= $2

                UNION ALL

                SELECT side, size, price, timestamp 
                FROM bitget_trading_history_data.{str(symbol).lower()}_p_trades
                WHERE timestamp >= $1 AND timestamp <= $2

                UNION ALL

                SELECT side, size, price, timestamp 
                FROM okx_trading_history_data.{str(symbol).lower()}_p_trades
                WHERE timestamp >= $1 AND timestamp <= $2
                
                UNION ALL

                SELECT side, size, price, timestamp 
                FROM gateio_trading_history_data.{str(symbol).lower()}_p_trades
                WHERE timestamp >= $1 AND timestamp <= $2
                ) AS combined
                ORDER BY timestamp ASC;
            """

    rows_support = []
    rows_resistance = []
    rows = await conn.fetch(query, start_time, reference_time)

    for row in rows:
        if row['price'] < big_order_price:
            rows_support.append(row)
        else:
            rows_resistance.append(row)

    cvd_support = calculate_cvd(rows_support)
    cvd_resistance = calculate_cvd(rows_resistance)
    await conn.close()

    return {
        "cvd_support": cvd_support,
        "cvd_resistance": cvd_resistance
    }


async def find_high_low_prices(reference_time: datetime, symbol, period=None):
    """
    Метод для пошуку найвищої та найнижчої ціни за останні 24години.
    Повертає list із значеннями max_price, min_price
    [max_price, min_price]

    :param reference_time:
    :param symbol:
    :param period:
    :return:
    """
    extremes_period = None
    if not period:
        extremes_period = EXTREMES_PERIOD
    if period:
        extremes_period = period

    conn = await get_db_pool()
    start_time = reference_time - timedelta(hours=extremes_period)

    query = f"""
        WITH price_data AS (
            SELECT price, timestamp
            FROM bybit_trading_history_data.{str(symbol).lower()}_p_trades
            WHERE timestamp BETWEEN $1 AND $2
        )
        SELECT
            (SELECT price FROM price_data ORDER BY price DESC LIMIT 1) AS max_price,
            (SELECT timestamp FROM price_data ORDER BY price DESC LIMIT 1) AS max_time,
            (SELECT price FROM price_data ORDER BY price ASC LIMIT 1) AS min_price,
            (SELECT timestamp FROM price_data ORDER BY price ASC LIMIT 1) AS min_time;
    """

    result = await conn.fetchrow(query, start_time, reference_time)
    await conn.close()

    max_price = result['max_price']
    if max_price is not None:
        max_price = round(Decimal(max_price), SYMBOLS_ROUNDING[symbol])
    else:
        max_price = None

    min_price = result['min_price']
    if min_price is not None:
        min_price = round(Decimal(min_price), SYMBOLS_ROUNDING[symbol])
    else:
        min_price = None

    return {
        'max_price': max_price,
        'min_price': min_price,
        'max_price_time': result['max_time'],
        'min_price_time': result['min_time']
    }
