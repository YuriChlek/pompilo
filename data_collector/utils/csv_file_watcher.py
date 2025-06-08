import os
import gzip
import zipfile
import csv
from datetime import datetime
from .db import get_db_pool
from .db import insert_csv_data
from constants import EXCHANGE_BYBIT, EXCHANGE_BINANCE


def parse_unix_timestamp(unix_timestamp_str):
    timestamp = float(unix_timestamp_str)
    dt = datetime.utcfromtimestamp(timestamp)

    return dt.replace(microsecond=0)


def parse_trade_row_bybit_perpetual(row):
    try:
        timestamp = parse_unix_timestamp(row[0])
        symbol = row[1]
        side = str(row[2]).lower()
        size = float(row[3])
        price = float(row[4])
        return (
            timestamp, symbol, side, size, price
        )
    except Exception as e:
        print(f"Error parsing row: {row} | {e}")
        return None


def parse_trade_row_bybit_spot(row, trading_symbol):
    try:
        timestamp = parse_unix_timestamp(int(row[1]) / 1000)
        symbol = trading_symbol
        side = str(row[4]).lower()
        size = float(row[3])
        price = float(row[2])
        return (
            timestamp, symbol, side, size, price
        )
    except Exception as e:
        print(f"Error parsing row spot: {row} | {e}")
        return None


def parse_trade_row_binance_perpetual(row, trading_symbol):
    try:
        timestamp = parse_unix_timestamp(int(row[4]) / 1000)
        symbol = trading_symbol
        side = 'sell' if row[-1].lower() == 'true' else 'buy'
        size = float(row[2])
        price = float(row[1])
        return (
            timestamp, symbol, side, size, price
        )
    except Exception as e:
        print(f"Error parsing row: {row} | {e}")
        return None


def parse_trade_row_binance_spot(row):
    print('def parse_trade_row_bybit_spot(row):')
    return None


def get_trading_symbol(filepath):
    filename = os.path.basename(filepath)
    if 'USDT' in filename:
        return filename.split('USDT')[0] + 'USDT'
    raise ValueError("Не вдалося визначити торгову пару з файлу.")


def get_table(contract_type, trading_symbol):
    tables = {
        "s": f"{str(trading_symbol).lower()}_s_trades",
        "p": f"{str(trading_symbol).lower()}_p_trades",
    }
    return tables[contract_type]


async def process_bybit_csv_file(file_path, pool, contract_type):
    parsers = {
        "s": parse_trade_row_bybit_spot,
        "p": parse_trade_row_bybit_perpetual,
    }

    trading_symbol = get_trading_symbol(file_path)
    parser = parsers[contract_type]
    table = get_table(contract_type, trading_symbol)

    print(f"Processing: {file_path}")
    trades = []
    with gzip.open(file_path, 'rt') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if contract_type == 's':
                parsed = parser(row, trading_symbol)
            else:
                parsed = parser(row)
            if parsed:
                trades.append(parsed)

    if trades:
        await insert_csv_data(pool, trades, EXCHANGE_BYBIT, table)
        print(f"Inserted {len(trades)} trades from {file_path}")
        os.remove(file_path)


async def process_binance_csv_file(file_path, pool, contract_type):
    parsers = {
        "s": parse_trade_row_binance_spot,
        "p": parse_trade_row_binance_perpetual,
    }

    trading_symbol = get_trading_symbol(file_path)
    parser = parsers[contract_type]
    table = get_table(contract_type, trading_symbol)

    print(f"Processing: {file_path}")
    trades = []

    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        file_name = zip_ref.namelist()[0]
        with zip_ref.open(file_name) as f:
            reader = csv.reader(line.decode('utf-8') for line in f)
            next(reader, None)

            for row in reader:
                parsed = parser(row, trading_symbol)
                if parsed:
                    trades.append(parsed)

    if trades:
        await insert_csv_data(pool, trades, EXCHANGE_BINANCE, table)
        print(f"Inserted {len(trades)} trades from {file_path}")
        os.remove(file_path)


async def start_csv_watcher(folder_path, contract_type, exchange):
    file_types = {
        "bybit": ".csv.gz",
        "binance": ".zip"
    }

    pool = await get_db_pool()
    files = sorted(f for f in os.listdir(folder_path) if f.endswith(file_types[exchange]))

    if exchange == EXCHANGE_BYBIT:
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            await process_bybit_csv_file(file_path, pool, contract_type)
    elif exchange == EXCHANGE_BINANCE:
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            await process_binance_csv_file(file_path, pool, contract_type)
