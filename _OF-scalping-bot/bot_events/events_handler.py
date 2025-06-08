from decimal import Decimal
from pyee.asyncio import AsyncIOEventEmitter
from bot import generate_scalping_signal, open_order, get_open_positions
from telegram_bot import send_pompilo_order_message, send_big_order_message

emitter = AsyncIOEventEmitter()

stop_trading = True

@emitter.on('big_order_open')
async def handle_order(
        max_order_time,
        symbol,
        max_order_direction,
        max_order_price,
        max_order_size,
        exchange):
    try:
        order_data = await generate_scalping_signal(
            symbol,
            max_order_time,
            str(max_order_direction).lower(),
            max_order_price,
            max_order_size)

        if not order_data:
            print("Failed generate order data")
            return

        signal = order_data['signal']
        position = order_data['position']

        await send_big_order_message(
            symbol,
            max_order_price,
            round(float(max_order_size), 2),
            max_order_time,
            max_order_direction,
            exchange,
            signal
        )

        if not position or stop_trading:
            return

        # Отримати відкриті позиції по цьому символу
        opened_positions = get_open_positions(position['symbol'])

        # Фільтруємо тільки ті, в яких size > 0
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
                position['take_profit']
            )

            await send_pompilo_order_message(
                position['symbol'],
                max_order_price,
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
                position['take_profit']
            )
            print(f"[OPEN NEW POSITION AFTER CLOSE]: {signal}")

            await send_pompilo_order_message(
                position['symbol'],
                max_order_price,
                position['take_profit'],
                position['stop_loss'],
                position['direction'],
                signal
            )
        else:
            print(f"[SKIP]: Already have same direction position for {position['symbol']}")

    except Exception as e:
        print(f"[EVENT HANDLER ERROR]: {e}")
