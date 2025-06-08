import asyncio
import sys

from api_collector import start_api_collector
from csv_data_colector import start_csv_bybit_collector, start_csv_binance_collector
from data_aggregator import start_data_aggregator


def runner(runner_command):
    parts = runner_command.split(':')
    command_type = parts[0]

    methods = {
        "set:data:bybit:p": start_csv_bybit_collector,
        "set:data:bybit:s": start_csv_bybit_collector,
        "set:data:binance:s": start_csv_binance_collector,
        "set:data:binance:p": start_csv_binance_collector,
        "start:api:collector": start_api_collector,
        "start:data:aggregator": start_data_aggregator
    }

    run_method = methods[runner_command]

    if command_type == 'start':
        asyncio.run(run_method())
        return

    exchange, contract_type = parts[-2], parts[-1]
    if contract_type == 's' or contract_type == 'p':
        asyncio.run(run_method(contract_type))
        return


if __name__ == '__main__':
    if len(sys.argv) > 1:
        runner(sys.argv[1])
    else:
        print("Please enter the command.")
