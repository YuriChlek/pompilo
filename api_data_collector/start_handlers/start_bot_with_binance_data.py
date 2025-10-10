import asyncio
import json
import aiohttp
from datetime import datetime, timezone
from utils import insert_api_data, get_db_pool, TRADING_SYMBOLS, MIN_BIG_TRADES_SIZES
from bot_events import emitter
from decimal import Decimal

exchange = 'binance'
WS_URL = "wss://fstream.binance.com/ws"
NUM_WORKERS = len(TRADING_SYMBOLS) * 3
MAX_QUEUE_SIZE = 5000

queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)


class BinanceWebSocketManager:
    def __init__(self):
        self.connections = {}
        self.reconnect_delays = {}

    async def connect_symbol(self, symbol):
        """Підключення до WebSocket для одного символа"""
        stream_name = f"{symbol.lower()}@trade"
        url = f"{WS_URL}/{stream_name}"

        while True:
            try:
                print(f"🔗 Connecting to {stream_name}")
                session = aiohttp.ClientSession()
                websocket = await session.ws_connect(
                    url,
                    heartbeat=30,
                    timeout=10,
                    receive_timeout=60
                )

                self.connections[symbol] = {
                    'websocket': websocket,
                    'session': session
                }
                self.reconnect_delays[symbol] = 1  # Скидаємо затримку перепідключення

                print(f"✅ Connected to {stream_name}")
                await self.listen_symbol(symbol, websocket, session)

            except Exception as e:
                print(f"❌ Connection error for {symbol}: {e}")
                await self.handle_reconnect(symbol)

    async def listen_symbol(self, symbol, websocket, session):
        """Прослуховування повідомлень для символу"""
        try:
            async for msg in websocket:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await handle_trade_message(data, symbol)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error for {symbol}: {e}")
                    except Exception as e:
                        print(f"Message processing error for {symbol}: {e}")

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"WebSocket error for {symbol}: {msg.data}")
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    print(f"WebSocket closed for {symbol}")
                    break

        except Exception as e:
            print(f"Listen error for {symbol}: {e}")
        finally:
            await self.cleanup_connection(symbol)

    async def handle_reconnect(self, symbol):
        """Обробка перепідключення з експоненційною затримкою"""
        delay = self.reconnect_delays.get(symbol, 1)
        print(f"🔄 Reconnecting {symbol} in {delay}s...")

        await asyncio.sleep(delay)

        # Експоненційна затримка з максимум 30 секунд
        self.reconnect_delays[symbol] = min(delay * 2, 30)

    async def cleanup_connection(self, symbol):
        """Очищення з'єднання"""
        if symbol in self.connections:
            conn = self.connections[symbol]
            try:
                await conn['websocket'].close()
            except:
                pass
            try:
                await conn['session'].close()
            except:
                pass
            del self.connections[symbol]

    async def close_all(self):
        """Закриття всіх з'єднань"""
        for symbol in list(self.connections.keys()):
            await self.cleanup_connection(symbol)


async def handle_trade_message(data, symbol):
    """Обробка торгових повідомлень"""
    try:
        if data.get('e') != 'trade':
            return

        # Швидка обробка даних
        timestamp = datetime.fromtimestamp(data['T'] / 1000, tz=timezone.utc).replace(tzinfo=None)
        side = 'Buy' if not data['m'] else 'Sell'

        trade_data = (
            timestamp,
            symbol.upper(),
            side,
            float(data['p']),
            float(data['q']),
            f"{data['T']}{data['p']}{data['q']}"
        )

        # Ефективне додавання в чергу
        try:
            queue.put_nowait((trade_data, exchange, symbol))
        except asyncio.QueueFull:
            # Відкидаємо повідомлення якщо черга переповнена
            pass

    except Exception as e:
        print(f"Trade message error for {symbol}: {e}")


async def worker(pool):
    """Воркер для обробки даних з черги"""
    while True:
        try:
            data = await queue.get()
            trade_data, exchange_name, symbol = data
            timestamp, symbol, side, price, size, order_id = trade_data

            # ---- Перевірка на великий трейд ----
            threshold = MIN_BIG_TRADES_SIZES.get(symbol.upper())
            if size and threshold and float(size) >= float(threshold):
                emitter.emit('big_order_open', symbol, side, price)

            if size > 0 and price > 0:
                await insert_api_data(pool, *data)

            queue.task_done()

        except Exception as e:
            print(f"[WORKER ERROR] {e}")
            await asyncio.sleep(0.1)


async def start_bot_with_binance_data():
    """Головна функція запуску бота"""
    ws_manager = BinanceWebSocketManager()
    pool = None

    try:
        # Підключаємося до БД
        pool = await get_db_pool()

        # Запускаємо воркери
        worker_tasks = []
        for _ in range(NUM_WORKERS):
            task = asyncio.create_task(worker(pool))
            worker_tasks.append(task)

        # Запускаємо WebSocket для кожного символу
        ws_tasks = [
            asyncio.create_task(ws_manager.connect_symbol(symbol))
            for symbol in TRADING_SYMBOLS
        ]

        # Чекаємо завершення всіх задач
        await asyncio.gather(*ws_tasks, return_exceptions=True)

    except Exception as e:
        print(f"🚨 Main bot error: {e}")
    finally:
        # Коректне закриття
        print("🛑 Shutting down...")
        await ws_manager.close_all()

        # Скасовуємо воркери
        for task in worker_tasks:
            task.cancel()

        # Чекаємо завершення
        if worker_tasks:
            await asyncio.gather(*worker_tasks, return_exceptions=True)

        if pool:
            await pool.close()


async def monitor_system():
    """Моніторинг стану системи"""
    while True:
        await asyncio.sleep(60)
        qsize = queue.qsize()
        status = "✅ OK" if qsize < MAX_QUEUE_SIZE * 0.8 else "⚠️ BUSY"
        print(f"📊 System status: {status} | Queue: {qsize}/{MAX_QUEUE_SIZE}")


if __name__ == "__main__":
    # Запуск моніторингу
    asyncio.create_task(monitor_system())

    # Запуск основного бота
    asyncio.run(start_bot_with_binance_data())