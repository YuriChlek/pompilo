from __future__ import annotations

import argparse
import asyncio
import logging
import signal

from application.command_dispatcher import dispatch_command
from application.runtime_commands import RuntimeCommandService
from utils.config import DEFAULT_LOOKBACK_DAYS, H4_ANALYSIS_DAYS, LOG_LEVEL, SPOT_TRADING_SYMBOLS

logger = logging.getLogger(__name__)


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
    sync_4h_parser = subparsers.add_parser("sync-4h", help="Sync 4H candles from Binance")
    sync_4h_parser.add_argument("--period", type=_positive_int, default=H4_ANALYSIS_DAYS)
    subparsers.add_parser("sync-full", help="Sync both D1 and 4H candles from Binance")

    analyze_parser = subparsers.add_parser("analyze", help="Run one analysis cycle without scheduling")
    analyze_parser.add_argument("--symbol", choices=SPOT_TRADING_SYMBOLS)
    analyze_parser.add_argument("--timeframe", choices=("4h",), default="4h")
    analyze_parser.add_argument("--dry-run", action="store_true", help="Calculate signals and execution decisions without real Binance orders")

    subparsers.add_parser("init-db", help="Create candle and spot-ledger tables")
    subparsers.add_parser("migrate", help="Run SQL migrations from the migrations directory")
    parser.add_argument("--dry-run", action="store_true", help="Run the scheduled bot without real Binance orders")
    return parser


def _shutdown(sig, frame) -> None:
    logger.info("graceful_shutdown_initiated signal=%s", sig)
    raise SystemExit(0)


def _configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


async def start() -> None:
    args = _build_parser().parse_args()
    await dispatch_command(args, RuntimeCommandService())


if __name__ == "__main__":
    _configure_logging()
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    try:
        asyncio.run(start())
    except KeyboardInterrupt:
        logger.info("script_stopped_by_user")
    except Exception as e:
        logger.exception("unexpected_error error=%s", e)
