"""
    агрегація + обрахунок imbalance,
    обчислення середнього значення size у двох таблицях,
    CVD,
    Proxy OI ???
"""

from datetime import datetime, timedelta
from decimal import Decimal
from utils import (
    get_db_pool,
    EXTREMES_PERIOD,
    IMBALANCE_AND_CVD_PERIOD,
    SYMBOLS_ROUNDING,
    BIN_SIZE
)
from collections import defaultdict

SYMBOL = "ETHUSDT"

"""
    Метод для розрахунку середнього розміру позиції.
"""


async def calculate_avg_size(symbol):
    conn = await get_db_pool()

    # Запит на обчислення середнього значення size у двох таблицях
    query = f"""
            SELECT COUNT(*) as total_count, AVG(size) as avg_size
            FROM (
                SELECT size FROM bybit_trading_history_data.{str(symbol).lower()}_p_trades
                UNION ALL
                SELECT size FROM binance_trading_history_data.{str(symbol).lower()}_p_trades
            ) AS all_sizes;
        """

    result = await conn.fetchrow(query)
    avg_size = result['avg_size']

    return round(avg_size, 5)


"""
    Метод для пошуку великих ордерів, 
    які можуть призвести до руху ціни
"""


async def find_max_size(symbol):
    conn = await get_db_pool()

    query = f"""
            SELECT timestamp, side, size, price
            FROM (
                SELECT timestamp, side, size, price FROM bybit_trading_history_data.{str(symbol).lower()}_p_trades
                UNION ALL
                SELECT timestamp, side, size, price FROM binance_trading_history_data.{str(symbol).lower()}_p_trades
            ) AS all_data
            ORDER BY size DESC
            LIMIT 1;
        """

    result = await conn.fetchrow(query)
    max_size = result['size']
    max_size_timestamp = result['timestamp']
    direction = result['side']
    price = result['price']

    return [max_size_timestamp,
            round(float(max_size), SYMBOLS_ROUNDING[symbol]),
            direction,
            round(float(price), SYMBOLS_ROUNDING[symbol])]


"""
    Метод який розраховує імбаланс маж Sell та Buy за 8 або 12 годин до вказаного часу.
    повертає значення imbalance
    imbalance
"""


def calculate_imbalance(rows):
    side_map = {row['side'].lower(): float(row['total_size']) for row in rows}
    buy = side_map.get('buy', 0.0)
    sell = side_map.get('sell', 0.0)
    total = buy + sell
    if total == 0:
        return 0.0
    return round((buy - sell) / total, 4)


"""
    Метод для розрахунку Cumulative Volume Delta
    повертає значення Cumulative Volume Delta
    cvd
"""


def calculate_cvd(rows):
    side_map = {row['side'].lower(): float(row['total_size']) for row in rows}
    buy_size = side_map.get('buy', 0.0)
    sell_size = side_map.get('sell', 0.0)
    return round(buy_size - sell_size, 0)


async def get_cvd_and_imbalance(reference_time: datetime, symbol):
    conn = await get_db_pool()
    # 8 or 12
    start_time = reference_time - timedelta(hours=IMBALANCE_AND_CVD_PERIOD)

    query = f"""
            SELECT side, SUM(size) AS total_size
            FROM (
                SELECT side, size, timestamp FROM bybit_trading_history_data.{str(symbol).lower()}_p_trades
                WHERE timestamp >= $1 AND timestamp <= $2
                UNION ALL
                SELECT side, size, timestamp FROM binance_trading_history_data.{str(symbol).lower()}_p_trades
                WHERE timestamp >= $1 AND timestamp <= $2
            ) AS combined
            GROUP BY side;
        """

    rows = await conn.fetch(query, start_time, reference_time)
    imbalance = calculate_imbalance(rows)
    cvd = calculate_cvd(rows)
    await conn.close()

    return {
        "cvd": cvd,
        "imbalance": imbalance,
    }


async def get_volume_profile_data(reference_time: datetime, symbol):
    conn = await get_db_pool()

    volume_start_time = reference_time - timedelta(hours=48)
    query_volume_profile = f"""
            SELECT price, size
            FROM (
                SELECT price, size, timestamp FROM bybit_trading_history_data.{str(symbol).lower()}_p_trades
                WHERE timestamp >= $1 AND timestamp <= $2
                UNION ALL
                SELECT price, size, timestamp FROM binance_trading_history_data.{str(symbol).lower()}_p_trades
                WHERE timestamp >= $1 AND timestamp <= $2
            ) AS combined;
        """

    volume_rows = await conn.fetch(query_volume_profile, volume_start_time, reference_time)
    volume_profile_data = calculate_volume_profile(volume_rows, BIN_SIZE)
    await conn.close()

    return {
        'val': volume_profile_data["VAL"],
        'vah': volume_profile_data["VAH"],
        'vpoc': volume_profile_data["VPOC"],
    }

"""
Метод для розрахунку профілю об'єму VAL, VAH, VPOC
"""

def calculate_volume_profile(rows, bin_size_ratio=0.005):
    if not rows:
        return None

    prices_sizes = []
    for row in rows:
        price = row['price']
        size = row['size']
        if price is not None and size is not None:
            prices_sizes.append((float(price), float(size)))

    if not prices_sizes:
        return None

    prices = [p for p, _ in prices_sizes]
    min_price, max_price = min(prices), max(prices)
    price_range = max_price - min_price
    bin_size = price_range * bin_size_ratio
    num_bins = int(price_range / bin_size) + 1

    volume_bins = defaultdict(float)
    for price, size in prices_sizes:
        bin_index = int((price - min_price) / bin_size)
        volume_bins[bin_index] += size

    sorted_bins = sorted(volume_bins.items(), key=lambda x: x[0])
    total_volume = sum(volume_bins.values())

    vpoc_index = max(volume_bins.items(), key=lambda x: x[1])[0]
    vpoc_price = min_price + vpoc_index * bin_size

    target_volume = total_volume * 0.7
    cumulative_volume = 0
    selected_bins = []

    sorted_indices = sorted(volume_bins.keys())
    center_idx = sorted_indices.index(vpoc_index)

    left = center_idx
    right = center_idx + 1
    selected_bins.append(vpoc_index)
    cumulative_volume += volume_bins[vpoc_index]

    while cumulative_volume < target_volume and (left > 0 or right < len(sorted_indices)):
        left_val = volume_bins.get(sorted_indices[left - 1]) if left > 0 else 0
        right_val = volume_bins.get(sorted_indices[right]) if right < len(sorted_indices) else 0

        if right_val >= left_val and right < len(sorted_indices):
            cumulative_volume += right_val
            selected_bins.append(sorted_indices[right])
            right += 1
        elif left > 0:
            cumulative_volume += left_val
            selected_bins.append(sorted_indices[left - 1])
            left -= 1
        else:
            break

    selected_prices = [min_price + idx * bin_size for idx in selected_bins]
    val, vah = min(selected_prices), max(selected_prices)

    return {
        'VAL': round(val, 4),
        'VAH': round(vah, 4),
        'VPOC': round(vpoc_price, 4),
        'total_volume': round(total_volume, 2),
        'min_price': round(min_price, 4),
        'max_price': round(max_price, 4)
    }

def generate_volume_profile_signal(price, vah, val, vpoc, cvd_delta, imbalance, big_order_side, big_order_price):
    """
    Генерує торговий сигнал на основі профілю об'єму, CVD, імбалансу та великих ордерів.
    """

    signal = None
    reasons = []

    # --- BUY LOGIKA ---
    if price < val:
        if cvd_delta > 0 and imbalance > 0:
            signal = 'BUY'
            reasons.append("Accumulation below VAL with positive CVD and imbalance")
            print(f"Buy")

        # Додатковий тригер: великий ордер на купівлю нижче VAL
        if big_order_side == 'buy' and big_order_price < val:
            signal = 'BUY'
            reasons.append("Big Buy Order placed below VAL")
            print(f"Buy")

    elif price < vpoc:
        if big_order_side == 'buy' and big_order_price <= price:
            signal = 'BUY'
            reasons.append("Big Buy Order near VPOC from below")
            print(f"Buy")

    # --- SELL LOGIKA ---
    if price > vah:
        if cvd_delta < 0 and imbalance < 0:
            signal = 'SELL'
            reasons.append("Distribution above VAH with negative CVD and imbalance")
            print(f"Sell")

        # Додатковий тригер: великий ордер на продаж вище VAH
        if big_order_side == 'sell' and big_order_price > vah:
            signal = 'SELL'
            reasons.append("Big Sell Order placed above VAH")
            print(f"Sell")

    elif price > vpoc:
        if big_order_side == 'sell' and big_order_price >= price:
            signal = 'SELL'
            reasons.append("Big Sell Order near VPOC from above")
            print(f"Sell")

    return signal, reasons


"""
    Метод для пошуку найвищої та найнижчої ціни за останні 48годин.
    Повертає list із значеннями max_price, min_price 
    [max_price, min_price]
"""


async def find_high_low_prices(reference_time: datetime, symbol):
    conn = await get_db_pool()
    start_time = reference_time - timedelta(hours=EXTREMES_PERIOD)

    query = f"""
        SELECT MAX(price) AS max_price, MIN(price) AS min_price
        FROM bybit_trading_history_data.{str(symbol).lower()}_p_trades
        WHERE timestamp BETWEEN $1 AND $2;
    """

    result = await conn.fetchrow(query, start_time, reference_time)

    await conn.close()

    return [
        round(Decimal(result['max_price']), SYMBOLS_ROUNDING[symbol]),
        round(Decimal(result['min_price']), SYMBOLS_ROUNDING[symbol])
    ]


"""
    Метод для тестового запуску 
"""


async def get_aggregator_data():
    avg_size = await calculate_avg_size(SYMBOL)
    print('avg_size', avg_size)
    return {
        "avg_size": avg_size,
    }
