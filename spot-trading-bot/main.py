from __future__ import annotations

import argparse
import asyncio

from api import run_api
from trading.application.bootstrap import build_live_trading_cycle, build_live_trading_scheduler
from utils.config import DEFAULT_DAILY_TARGET_HOUR, DEFAULT_DAILY_TARGET_MINUTE, DEFAULT_DAILY_TARGET_SECOND, DEFAULT_LOOKBACK_DAYS, SPOT_TRADING_SYMBOLS, THREE_YEARS_DAYS
from utils.create_tables import main as create_tables_main
from utils.run_migrations import main as run_migrations_main


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


async def _run_once(symbol: str | None = None, *, dry_run: bool = False) -> None:
    trading_cycle = build_live_trading_cycle()
    symbols = [symbol] if symbol else SPOT_TRADING_SYMBOLS
    print(f"🚀 Starting one-off analysis for {len(symbols)} symbol(s), dry_run={dry_run}")
    for current_symbol in symbols:
        print(f"🧠 Running one-off cycle for {current_symbol}")
        await trading_cycle.run(current_symbol, dry_run=dry_run)
    print("✅ One-off analysis completed")


async def start() -> None:
    args = _build_parser().parse_args()
    if args.command == "sync":
        print(f"🚀 Starting D1 sync for {len(SPOT_TRADING_SYMBOLS)} symbol(s)")
        await run_api(days=args.period)
        print("✅ D1 sync completed")
        return
    if args.command == "sync-3y":
        print(f"🚀 Starting 3-year D1 sync for {len(SPOT_TRADING_SYMBOLS)} symbol(s)")
        await run_api(days=THREE_YEARS_DAYS)
        print("✅ 3-year D1 sync completed")
        return
    if args.command == "analyze":
        await _run_once(args.symbol, dry_run=args.dry_run)
        return
    if args.command == "init-db":
        print("🚀 Creating spot bot tables")
        await create_tables_main()
        print("✅ Spot bot tables are ready")
        return
    if args.command == "migrate":
        print("🚀 Running SQL migrations")
        await run_migrations_main()
        print("✅ SQL migrations completed")
        return

    scheduler = build_live_trading_scheduler()
    print("🚀 Starting spot Greenwich bot")
    print(f"📊 Symbols: {', '.join(SPOT_TRADING_SYMBOLS)}")
    await scheduler.run_forever(
        SPOT_TRADING_SYMBOLS,
        target_hour=DEFAULT_DAILY_TARGET_HOUR,
        target_minute=DEFAULT_DAILY_TARGET_MINUTE,
        target_second=DEFAULT_DAILY_TARGET_SECOND,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    try:
        asyncio.run(start())
    except KeyboardInterrupt:
        print("⏹️ Скрипт зупинено користувачем")
    except Exception as e:
        print(f"💥 Неочікувана помилка: {e}")
        import traceback

        traceback.print_exc()
