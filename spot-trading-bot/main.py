from __future__ import annotations

import argparse
import asyncio

from application.command_dispatcher import dispatch_command
from application.runtime_commands import RuntimeCommandService
from utils.config import DEFAULT_LOOKBACK_DAYS, SPOT_TRADING_SYMBOLS


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spot Greenwich bot entrypoint")
    subparsers = parser.add_subparsers(dest="command")

    sync_parser = subparsers.add_parser("sync", help="Sync D1 candles from Binance")
    sync_parser.add_argument("--period", type=_positive_int, default=DEFAULT_LOOKBACK_DAYS)
    subparsers.add_parser("sync-3y", help="Sync 3 years of D1 candles from Binance")

    analyze_parser = subparsers.add_parser("analyze", help="Run one analysis cycle without scheduling")
    analyze_parser.add_argument("--symbol", choices=SPOT_TRADING_SYMBOLS)
    analyze_parser.add_argument("--dry-run", action="store_true", help="Calculate signals and execution decisions without real Binance orders")

    subparsers.add_parser("init-db", help="Create candle and spot-ledger tables")
    subparsers.add_parser("migrate", help="Run SQL migrations from the migrations directory")
    parser.add_argument("--dry-run", action="store_true", help="Run the scheduled bot without real Binance orders")
    return parser

async def start() -> None:
    args = _build_parser().parse_args()
    await dispatch_command(args, RuntimeCommandService())


if __name__ == "__main__":
    try:
        asyncio.run(start())
    except KeyboardInterrupt:
        print("⏹️ Скрипт зупинено користувачем")
    except Exception as e:
        print(f"💥 Неочікувана помилка: {e}")
        import traceback

        traceback.print_exc()
