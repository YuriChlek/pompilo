from decimal import Decimal

from data_actions import (
    find_high_low_prices,
    get_cvd_and_imbalance,
    get_volume_profile_data,
    find_max_size,
    calculate_avg_size
)

from utils import (
    get_db_pool,
    BUY_DIRECTION,
    SELL_DIRECTION,
    SYMBOLS_ROUNDING
)
from .order_actions import calculate_order_data


async def get_is_good_price(max_order_value_time, max_order_direction, max_order_price, symbol):
    max_price, min_price = await find_high_low_prices(max_order_value_time, symbol)
    distance_to_max = round(abs(Decimal(max_price) - Decimal(max_order_price)), SYMBOLS_ROUNDING[symbol])
    distance_to_min = round(abs(Decimal(min_price) - Decimal(max_order_price)), SYMBOLS_ROUNDING[symbol])

    if max_order_direction == SELL_DIRECTION and max_order_price >= max_price or distance_to_max < distance_to_min:
        return [SELL_DIRECTION, distance_to_max, distance_to_min]
    elif max_order_direction == BUY_DIRECTION and max_order_price <= min_price or distance_to_max > distance_to_min:
        return [BUY_DIRECTION, distance_to_max, distance_to_min]
    else:
        return ['', distance_to_max, distance_to_min]


async def generate_signal(
        symbol,
        max_order_value_time,
        max_order_direction,
        max_order_price):
    is_good_price_for, distance_to_max, distance_to_min = await get_is_good_price(max_order_value_time,
                                                                                  max_order_direction,
                                                                                  max_order_price, symbol)
    cvd_imbalance_data = await get_cvd_and_imbalance(max_order_value_time, symbol)

    imbalance = cvd_imbalance_data['imbalance']
    cvd = cvd_imbalance_data['cvd']

    print('')

    if imbalance < 0 and cvd < 0 and max_order_direction == SELL_DIRECTION and is_good_price_for == SELL_DIRECTION:
        order = calculate_order_data(
            symbol,
            SELL_DIRECTION,
            max_order_price,
            distance_to_max,
            distance_to_min
        )
        print(f"Bot say: sell, at {max_order_value_time}. Order data {order}")
        print(
            f"Imbalance {imbalance}, cvd: {cvd}, direction: {max_order_direction}, price: {max_order_price}, time: {max_order_value_time}")
    elif imbalance > 0 and cvd > 0 and max_order_direction == BUY_DIRECTION and is_good_price_for == BUY_DIRECTION:
        order = calculate_order_data(
            symbol,
            BUY_DIRECTION,
            max_order_price,
            distance_to_max,
            distance_to_min
        )
        print(f"Bot say: buy, at {max_order_value_time}. Order data {order}")
        print(
            f"Imbalance {imbalance}, cvd: {cvd}, direction: {max_order_direction}, price: {max_order_price}, time: {max_order_value_time}")
    elif float(
            0.005) > imbalance > 0 and cvd > 0 and max_order_direction == SELL_DIRECTION and is_good_price_for == SELL_DIRECTION:
        order = calculate_order_data(
            symbol,
            SELL_DIRECTION,
            max_order_price,
            distance_to_max,
            distance_to_min
        )
        print(f"Bot say: sell, at {max_order_value_time}. Order data {order}")
        print(
            f"Imbalance {imbalance}, cvd: {cvd}, direction: {max_order_direction}, price: {max_order_price}, time: {max_order_value_time}")
    elif float(
            -0.005) < imbalance < 0 and cvd < 0 and max_order_direction == BUY_DIRECTION and is_good_price_for == BUY_DIRECTION:
        order = calculate_order_data(
            symbol,
            BUY_DIRECTION,
            max_order_price,
            distance_to_max,
            distance_to_min
        )
        print(f"Bot say: buy, at {max_order_value_time}. Order data {order}")
        print(
            f"Imbalance {imbalance}, cvd: {cvd}, direction: {max_order_direction}, price: {max_order_price}, time: {max_order_value_time}")
    else:
        print('Order not open')
        print(
            f"Time: {max_order_value_time}, Imbalance {imbalance}, cvd: {cvd}, is_good_price_for: {is_good_price_for} direction: {max_order_direction}, price: {max_order_price} ")


async def generate_signal_vwap(
        symbol,
        max_order_value_time,
        max_order_direction,
        max_order_price):
    print("generate_signal_vwap")


async def generate_signal_volume_profile(
        symbol,
        max_order_value_time,
        big_order_side,
        big_order_price):
    print("")
    cvd_imbalance_data = await get_cvd_and_imbalance(max_order_value_time, symbol)
    imbalance = cvd_imbalance_data['imbalance']
    cvd = cvd_imbalance_data['cvd']

    volume_profile_data = await get_volume_profile_data(max_order_value_time, symbol)
    val = volume_profile_data['val']
    vah = volume_profile_data['vah']
    vpoc = volume_profile_data['vpoc']

    signal = "Нейтрально / недостатньо даних"

    if big_order_side == 'buy':
        if big_order_price < vpoc and imbalance > 0.01 and cvd > 0:
            signal = "🔼 LONG: Акумуляція під VPOC з дисбалансом на покупку"
        elif big_order_price > vah and imbalance > 0.01:
            signal = "⚠️ LONG на хаях: Можливий відкат після агресивного росту"
        elif abs(imbalance) > 0.05 and cvd > 0:
            signal = "🔼 LONG імпульс: Сильний тиск покупців"
        elif big_order_price < vpoc and imbalance > 0.02 and cvd > 0 and big_order_price < val:
            signal = "🔼 LONG: Сильна акумуляція під VPOC при пробитті VAL"
        elif big_order_price < val and imbalance > 0.01 and cvd > 0:
            signal = "🔼 LONG під VAL: Можлива пастка для продавців (spring)"

    elif big_order_side == 'sell':
        if big_order_price > vpoc and imbalance < -0.01 and cvd < 0:
            signal = "🔽 SHORT: Розподіл над VPOC з дисбалансом на продаж"
        elif big_order_price < val and imbalance < -0.01:
            signal = "⚠️ SHORT на лоях: Можливий відкат після сильного падіння"
        elif abs(imbalance) > 0.05 and cvd < 0:
            signal = "🔽 SHORT імпульс: Сильний тиск продавців"
        elif big_order_price < vpoc and imbalance < -0.02 and cvd < 0 and big_order_price < val:
            signal = "🔽 SHORT: Сильний продаж під VPOC при пробитті VAL"
        elif big_order_price > vah and imbalance < -0.01 and cvd < 0:
            signal = "🔽 SHORT над VAH: Можлива пастка для покупців (upthrust)"

    print(f"Signal: {signal}")

    print(
        f"Time: {max_order_value_time}, Imbalance {imbalance}, cvd: {cvd}, direction: {big_order_side}, price: {big_order_price}, vah: {vah}, val: {val}: vpoc: {vpoc}")


"""
    # --- BUY LOGIKA ---
    if float(big_order_price) < float(val):
        if cvd > 0 and imbalance > 0:
            print(
                f"Time: {max_order_value_time}, Imbalance {imbalance}, cvd: {cvd}, direction: {big_order_side}, price: {big_order_price}")
        if big_order_side == 'buy' and big_order_price < val:
            print(
                f"Time: {max_order_value_time}, Imbalance {imbalance}, cvd: {cvd}, direction: {big_order_side}, price: {big_order_price}")
    elif float(big_order_price) < float(vpoc):
        if big_order_side == 'buy':
            print(
                f"Time: {max_order_value_time}, Imbalance {imbalance}, cvd: {cvd}, direction: {big_order_side}, price: {big_order_price}")

    # --- SELL LOGIKA ---
    if float(big_order_price) > float(vah):
        if cvd < 0 and imbalance < 0:
            print(
                f"Time: {max_order_value_time}, Imbalance {imbalance}, cvd: {cvd}, direction: {big_order_side}, price: {big_order_price}")
        if big_order_side == 'sell' and big_order_price > vah:
            print(
                f"Time: {max_order_value_time}, Imbalance {imbalance}, cvd: {cvd}, direction: {big_order_side}, price: {big_order_price}")
    elif float(big_order_price) > float(vpoc):
        if big_order_side == 'sell':
            print(
                f"Time: {max_order_value_time}, Imbalance {imbalance}, cvd: {cvd}, direction: {big_order_side}, price: {big_order_price}")

    return signal, reasons
"""

async def get_large_trades(symbol):
    conn = await get_db_pool()

    query = f"""
        SELECT timestamp, symbol, side, size, price
        FROM binance_trading_history_data.{str(symbol).lower()}_p_trades
        WHERE size > 4000
        UNION
        SELECT timestamp, symbol, side, size, price
        FROM bybit_trading_history_data.{str(symbol).lower()}_p_trades
        WHERE size > 4000
        ORDER BY timestamp;
    """

    rows = await conn.fetch(query)
    await conn.close()

    return rows


async def start_signal_generator():
    # max_order_value_time, max_order_size, max_order_direction, max_order_price = await find_max_size()
    '''max_order_value_time = datetime.fromisoformat('2025-04-09 18:18:29.000')
    max_order_size = 67906
    max_order_direction = 'sell'
    max_order_price = Decimal(118.9)

    if max_order_value_time and max_order_size and max_order_direction:
        await generate_signal(max_order_value_time, max_order_direction, max_order_price)
    '''
    symbol = 'ETHUSDT'

    rows = await get_large_trades(symbol)
    for item in rows:
        symbol = item['symbol']
        max_order_value_time = item['timestamp']
        max_order_size = item['size']
        max_order_direction = item['side']
        max_order_price = item['price']

        if max_order_value_time and max_order_size and max_order_direction:
            await generate_signal_volume_profile(
                symbol,
                max_order_value_time,
                max_order_direction,
                round(Decimal(max_order_price), SYMBOLS_ROUNDING[symbol])
            )
