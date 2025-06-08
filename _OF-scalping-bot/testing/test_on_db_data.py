from decimal import Decimal
from bot import generate_scalping_signal
from utils import (
    get_db_pool,
    SYMBOLS_ROUNDING
)

async def get_large_trades(symbol, size):
    """
    Метод отримує із бази даних найбільші трейди які більші ніж вказана size

    :param symbol:
    :param size:
    :return:
    """

    conn = await get_db_pool()

    query = f"""
        SELECT timestamp, symbol, side, size, price
        FROM binance_trading_history_data.{str(symbol).lower()}_p_trades
        WHERE size > {size}
        UNION
        SELECT timestamp, symbol, side, size, price
        FROM bybit_trading_history_data.{str(symbol).lower()}_p_trades
        WHERE size > {size}
        ORDER BY timestamp;
    """

    rows = await conn.fetch(query)
    await conn.close()
    print(f"Отримано дані із БД, {len(rows)} записів. {symbol}")

    return rows



async def start_signal_generator():
    """
    Метод для запуску тесту на даних із БД

    :return: void
    """

    test_data = [
        {
            'symbol': 'ADAUSDT',
            'size': Decimal(500000)
        },
        {
            'symbol': 'AAVEUSDT',
            'size': Decimal(1200)
        },
        {
            'symbol': 'BNBUSDT',
            'size': Decimal(1200)
        },
        {
            'symbol': 'JUPUSDT',
            'size': Decimal(300000)
        },
        {
            'symbol': 'DOGEUSDT',
            'size': Decimal(4000000)
        },
    ]

    """
                {
                    'symbol': 'AAVEUSDT',
                    'size': Decimal(1750)
                },
                {
                    'symbol': 'ADAUSDT',
                    'size': Decimal(500000)
                },
                {
                    'symbol': 'APTUSDT',
                    'size': Decimal(38000)
                },
                {
                    'symbol': 'BNBUSDT',
                    'size': Decimal(1795)
                },
                {
                    'symbol': 'DOGEUSDT',
                    'size': Decimal(4000000)
                },
                {
                    'symbol': 'JUPUSDT',
                    'size': Decimal(300000)
                },
                {
                    'symbol': 'SOLUSDT',
                    'size': Decimal(20000)
                },
                {
                    'symbol': 'SUIUSDT',
                    'size': Decimal(300000)
                },
                {
                    'symbol': 'XRPUSDT',
                    'size': Decimal(600000)
                },
    """

    for data_tem in test_data:
        rows = await get_large_trades(data_tem['symbol'], data_tem['size'])
        for item in rows:
            #emitter.emit('big_order_open', item)

            symbol = item['symbol']
            max_order_value_time = item['timestamp']
            max_order_size = item['size']
            max_order_direction = item['side']
            max_order_price = item['price']
            
            if max_order_value_time and max_order_size and max_order_direction:
                await generate_scalping_signal(
                    symbol,
                    max_order_value_time,
                    max_order_direction,
                    round(Decimal(max_order_price), SYMBOLS_ROUNDING[symbol]),
                    max_order_size
                )

