from decimal import Decimal

from data_actions import (
    find_high_low_prices,
    get_cvd_and_imbalance,
    find_max_size,

)

from utils import (
    BUY_DIRECTION,
    SELL_DIRECTION,
    SYMBOLS_ROUNDING
)
from .order_actions import calculate_order_data


# ********************************************************************************************
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


# ********************************************************************************************
