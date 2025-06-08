from pathlib import Path
from utils import start_csv_watcher, DATA_FOLDER
from constants import EXCHANGE_BYBIT

base_dir = Path(__file__).parent

async def start_csv_bybit_collector(contract_type):
    data_path = (base_dir / DATA_FOLDER).resolve()
    await start_csv_watcher(data_path, contract_type, EXCHANGE_BYBIT)
