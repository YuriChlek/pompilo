from __future__ import annotations

import argparse
import asyncio
from datetime import datetime

from backtesting import BacktestConfig, BacktestRunner, run_v2_entry_validation
from backtesting.validation import render_validation_report, save_validation_report
from utils.logging import setup_logging


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the backtest runner."""
    parser = argparse.ArgumentParser(description="Run database-backed backtest for trading strategy.")
    parser.add_argument("--symbol", dest="symbols", action="append", required=True, help="Trading symbol, repeat for multiple symbols.")
    parser.add_argument("--from", dest="date_from", help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--to", dest="date_to", help="End date in YYYY-MM-DD format.")
    parser.add_argument("--lookback-candles", dest="lookback_candles", type=int, default=1500)
    parser.add_argument("--min-candles", dest="min_candles", type=int, default=100)
    parser.add_argument("--indicator-history-period", dest="indicator_history_period", type=int, default=25)
    parser.add_argument("--no-reversal", dest="allow_reversal", action="store_false")
    parser.add_argument("--intrabar-exit-priority", choices=("stop", "target"), default="stop")
    parser.add_argument("--validation-profile", choices=("v2_entry",))
    parser.add_argument("--output-json", dest="output_json")
    return parser.parse_args()


def _parse_date(raw_value: str | None):
    """Convert a ``YYYY-MM-DD`` string into ``datetime`` or return ``None``."""
    if not raw_value:
        return None
    return datetime.strptime(raw_value, "%Y-%m-%d")


async def _main() -> None:
    """Build the config, run the backtest, and optionally save the result to a file."""
    args = _parse_args()
    config = BacktestConfig(
        symbols=args.symbols,
        date_from=_parse_date(args.date_from),
        date_to=_parse_date(args.date_to),
        lookback_candles=args.lookback_candles,
        min_candles=args.min_candles,
        indicator_history_period=args.indicator_history_period,
        allow_reversal=args.allow_reversal,
        intrabar_exit_priority=args.intrabar_exit_priority,
    )

    if args.validation_profile == "v2_entry":
        report = await run_v2_entry_validation(config)
        print(render_validation_report(report))
        if args.output_json:
            save_validation_report(args.output_json, report)
        return

    runner = BacktestRunner(config)
    result = await runner.run()
    runner.print_report(result)

    if args.output_json:
        runner.save_report(result, args.output_json)


if __name__ == "__main__":
    setup_logging()
    asyncio.run(_main())
