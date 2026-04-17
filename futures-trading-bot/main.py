import argparse
import asyncio
import logging
from datetime import datetime, timedelta

from api import run_api
from backtesting import BacktestConfig, BacktestRunner
from trading.application.bootstrap import build_live_trading_scheduler
from utils.config import TRADING_SYMBOLS
from utils.logging import setup_logging


logger = logging.getLogger(__name__)


def _positive_period(raw_value: str) -> int:
    """Validate and convert CLI period argument to a positive integer amount of days."""
    try:
        period = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--period має бути цілим числом днів") from exc

    if period <= 0:
        raise argparse.ArgumentTypeError("--period має бути більше 0")
    return period


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser with live, sync, and backtest modes."""
    parser = argparse.ArgumentParser(
        description="Pompilo trading service entrypoint. Without arguments runs the bot in normal mode."
    )
    subparsers = parser.add_subparsers(dest="command")

    backtest_parser = subparsers.add_parser("backtest", help="Run backtesting for one symbol and period in days.")
    backtest_parser.add_argument("--symbol", required=True, help="Trading symbol, for example SOLUSDT.")
    backtest_parser.add_argument("--period", required=True, type=_positive_period, help="Backtest period in days.")

    sync_parser = subparsers.add_parser("sync", help="Sync candles from Binance for all trading symbols.")
    sync_parser.add_argument("--period", type=_positive_period, default=365, help="Sync period in days. Default: 365.")

    return parser


async def _run_backtest(symbol: str, period_days: int) -> None:
    """Run backtesting for a single symbol over the trailing ``period_days`` window."""
    date_to = datetime.now()
    date_from = date_to - timedelta(days=period_days)
    config = BacktestConfig(
        symbols=[symbol.upper()],
        date_from=date_from,
        date_to=date_to,
    )
    runner = BacktestRunner(config)
    result = await runner.run()
    runner.print_report(result)


async def _run_sync(period_days: int) -> None:
    """Synchronize candle history for all configured symbols."""
    await run_api(days=period_days)


async def _run_default_mode() -> None:
    """Start the live trading scheduler for all configured trading symbols."""
    scheduler = build_live_trading_scheduler()
    await scheduler.run_forever(TRADING_SYMBOLS, target_minute=0, target_second=1, is_test=False)


async def start() -> None:
    """Run one of the service modes: live trading, candle sync, or backtest."""
    args = _build_parser().parse_args()

    if args.command == "backtest":
        await _run_backtest(args.symbol, args.period)
        return

    if args.command == "sync":
        await _run_sync(args.period)
        return

    await _run_default_mode()


if __name__ == "__main__":
    setup_logging()
    try:
        asyncio.run(start())
    except KeyboardInterrupt:
        logger.info("Script stopped by user.")
    except Exception:
        logger.exception("unexpected_application_error")
