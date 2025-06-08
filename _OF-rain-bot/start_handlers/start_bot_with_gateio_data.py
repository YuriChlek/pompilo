import asyncio
from decimal import Decimal
from datetime import datetime
from gate_api import ApiClient, Configuration, FuturesApi
from gate_api.exceptions import ApiException
from utils import insert_api_data, get_db_pool, TRADING_SYMBOLS

exchange = 'gateio'
queue = asyncio.Queue()


async def worker(pool):
    while True:
        data = await queue.get()
        if data is None:
            break
        try:
            await insert_api_data(pool, *data)
        except Exception as e:
            print(f"[WORKER ERROR] {e}")
        finally:
            queue.task_done()


async def fetch_contract_sizes(api):
    contract_sizes = {}
    while True:
        try:
            contracts = api.list_futures_contracts(settle='usdt')
            for contract in contracts:
                symbol = contract.name.replace('_USDT', 'USDT')
                contract_sizes[symbol] = float(contract.quanto_multiplier)
            break  # успішно завершили — вихід з циклу
        except Exception as e:
            print(f"[ERROR] Fetching contract sizes: {e}. Retrying in 10 seconds...")
            await asyncio.sleep(10)
    return contract_sizes


async def fetch_trades(api, symbol, contract_sizes):
    contract_symbol = symbol.replace('USDT', '_USDT')
    while True:
        try:
            trades = api.list_futures_trades(settle='usdt', contract=contract_symbol)

            for trade in trades:
                timestamp = datetime.fromtimestamp(int(trade.create_time))
                tr_symbol = trade.contract.replace('_USDT', 'USDT')
                side = 'Buy' if trade.size > 0 else 'Sell'
                price = float(trade.price)
                multiplier = contract_sizes.get(tr_symbol, 1)
                size = abs(trade.size) * multiplier
                ord_id = f"{Decimal(trade.create_time)}{price}{size}"

                await queue.put(((timestamp, tr_symbol, side, price, size, ord_id), exchange, tr_symbol))
        except ApiException as e:
            print(f"[GATE.IO API ERROR] {symbol}: {e}. Retrying in 10 seconds...")
        except Exception as e:
            print(f"[ERROR] Fetching trades for {symbol}: {e}. Retrying in 10 seconds...")
        await asyncio.sleep(5)


async def start_bot_with_gateio_data():
    pool = await get_db_pool()
    asyncio.create_task(worker(pool))

    config = Configuration()
    api_client = ApiClient(config)
    futures_api = FuturesApi(api_client)

    contract_sizes = await fetch_contract_sizes(futures_api)

    tasks = [fetch_trades(futures_api, symbol, contract_sizes) for symbol in TRADING_SYMBOLS]
    await asyncio.gather(*tasks)
