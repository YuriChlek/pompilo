from __future__ import annotations

import argparse
import asyncio
import logging

from application.bootstrap import build_health_check_server, build_live_trading_cycle, build_live_trading_scheduler
from application.dry_run import format_decision_dry_run
from infrastructure.binance_api import DEFAULT_TIMEFRAME, HIGHER_TIMEFRAME, run_binance_candle_sync
from infrastructure.db import ensure_candle_tables
from utils.config import HEALTHCHECK_PORT, RUN_TARGET_MINUTE, RUN_TARGET_SECOND, TRADING_SYMBOLS
from utils.logging import setup_logging

logger = logging.getLogger(__name__)
SUPPORTED_SYNC_TIMEFRAMES = (DEFAULT_TIMEFRAME, HIGHER_TIMEFRAME)


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
    """Build the command-line parser for one-shot and scheduled bot execution."""
    parser = argparse.ArgumentParser(description="Spot grid bot entrypoint.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("once", help="Run one trading cycle for all configured symbols.")
    subparsers.add_parser("dry-run", help="Preview planner diffs without syncing any orders.")
    subparsers.add_parser("live", help="Run the scheduler with scheduled 1h + 4h candle refresh.")
    sync_parser = subparsers.add_parser("sync", help="Manually sync one candle timeframe from Binance into its dedicated DB tables.")
    sync_parser.add_argument("--period", type=_positive_period, default=365, help="Sync period in days. Default: 365.")
    sync_parser.add_argument(
        "--timeframe",
        default=DEFAULT_TIMEFRAME,
        choices=SUPPORTED_SYNC_TIMEFRAMES,
        help="Manual sync timeframe. Allowed: 1h, 4h. Default: 1h.",
    )
    return parser


async def _run_once() -> None:
    """Run one trading cycle for each configured symbol."""
    logger.info("bot_starting mode=once symbols=%s", ",".join(TRADING_SYMBOLS))
    trading_cycle = build_live_trading_cycle()
    await trading_cycle.initialize(TRADING_SYMBOLS)
    await trading_cycle.run_many(TRADING_SYMBOLS)
    logger.info("bot_finished mode=once symbols=%s", ",".join(TRADING_SYMBOLS))


async def _run_sync(period_days: int, timeframe: str) -> None:
    """Run standalone candle synchronization for all configured symbols."""
    logger.info(
        "bot_starting mode=sync symbols=%s period_days=%s timeframe=%s",
        ",".join(TRADING_SYMBOLS),
        period_days,
        timeframe,
    )
    await ensure_candle_tables(TRADING_SYMBOLS)
    await run_binance_candle_sync(TRADING_SYMBOLS, timeframe=timeframe, days=period_days)
    logger.info(
        "bot_finished mode=sync symbols=%s period_days=%s timeframe=%s",
        ",".join(TRADING_SYMBOLS),
        period_days,
        timeframe,
    )


async def _run_live() -> None:
    """Run the recurring scheduler for all configured symbols."""
    logger.info(
        "bot_starting mode=live symbols=%s target_minute=%s target_second=%s",
        ",".join(TRADING_SYMBOLS),
        RUN_TARGET_MINUTE,
        RUN_TARGET_SECOND,
    )
    scheduler = build_live_trading_scheduler()
    health_server = build_health_check_server(scheduler.health_tracker) if HEALTHCHECK_PORT > 0 else None
    tasks = [
        asyncio.create_task(
            scheduler.run_forever(
                TRADING_SYMBOLS,
                target_minute=RUN_TARGET_MINUTE,
                target_second=RUN_TARGET_SECOND,
            )
        )
    ]
    if health_server is not None:
        tasks.append(asyncio.create_task(health_server.serve_forever()))
    await asyncio.gather(*tasks)


async def _run_dry_run() -> None:
    """Preview one planning cycle and print a structured live-vs-target diff."""
    logger.info("bot_starting mode=dry_run symbols=%s", ",".join(TRADING_SYMBOLS))
    trading_cycle = build_live_trading_cycle()
    await trading_cycle.initialize(TRADING_SYMBOLS)
    decisions = await trading_cycle.preview_many(TRADING_SYMBOLS)
    for symbol in TRADING_SYMBOLS:
        decision = decisions.get(symbol.upper())
        if decision is None:
            print(f"[{symbol.upper()}] Planning failed")
            continue
        print(format_decision_dry_run(decision))
    logger.info("bot_finished mode=dry_run symbols=%s", ",".join(TRADING_SYMBOLS))


async def start() -> None:
    """Dispatch the requested command and start the selected bot execution mode."""
    args = _build_parser().parse_args()
    if args.command == "sync":
        await _run_sync(args.period, args.timeframe)
        return
    if args.command == "live":
        await _run_live()
        return
    if args.command == "dry-run":
        await _run_dry_run()
        return
    await _run_once()


if __name__ == "__main__":
    setup_logging()
    try:
        asyncio.run(start())
    except KeyboardInterrupt:
        logger.info("script_stopped_by_user")
    except Exception:
        logger.exception("unexpected_application_error")
