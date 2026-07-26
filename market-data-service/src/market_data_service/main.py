from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from sqlalchemy.engine import make_url

from market_data_service.config.database_config import get_database_url
from market_data_service.config.settings import load_http_settings, load_settings
from market_data_service.domain.scheduler import SUPPORTED_SCHEDULE_TIMEFRAMES
from market_data_service.domain.symbol_normalization import normalize_symbol
from market_data_service.runtime.http_server import run_http_server
from market_data_service.runtime.maintenance import (
    run_collect_once,
    run_outbox_cleanup,
    run_symbols_enable,
    run_symbols_status,
)

EXIT_SUCCESS = 0
EXIT_ERROR = 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "collect":
        return _run_collect_contract(args)
    if args.command == "candles:get":
        return _run_candles_get_contract(args)
    if args.command == "symbols:status":
        return _run_symbols_status_contract(args)
    if args.command == "symbols:enable":
        return _run_symbols_enable_contract(args)

    if args.command == "serve":
        return asyncio.run(_run_serve())
    if args.command == "healthcheck":
        return _run_healthcheck()
    if args.command == "db:check":
        return _run_db_check()
    if args.command == "outbox:cleanup":
        return asyncio.run(_run_outbox_cleanup(args))

    parser.print_help()
    return EXIT_SUCCESS


async def _run_serve() -> int:
    await run_http_server()
    return EXIT_SUCCESS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m market_data_service.main",
        description="Market Data Service runtime and maintenance entrypoint.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    subparsers.add_parser("serve", help="Start the HTTP health/readiness/metrics API.")
    collect_parser = subparsers.add_parser("collect", help="Collect configured closed candles.")
    collect_parser.add_argument("--once", action="store_true", help="Execute a single collection tick and exit.")

    candles_get_parser = subparsers.add_parser("candles:get", help="Fetch candles from a provider for a recent period.")
    candles_get_parser.add_argument(
        "--period",
        required=True,
        help="Lookback period from current UTC time, e.g. 12h, 1d, 100d, 2w, 3mo, 1y.",
    )
    candles_get_parser.add_argument(
        "--symbols",
        help="Comma-separated list of symbols. Defaults to MARKET_DATA_PROVIDER_SYMBOLS.",
    )
    candles_get_parser.add_argument(
        "--timeframes",
        help="Comma-separated list of timeframes. Defaults to MARKET_DATA_TIMEFRAMES.",
    )
    candles_get_parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "binance", "bybit"],
        help="Provider to fetch from.",
    )

    subparsers.add_parser("healthcheck", help="Check HTTP readiness.")
    subparsers.add_parser("db:check", help="Validate database configuration.")

    outbox_cleanup_parser = subparsers.add_parser("outbox:cleanup", help="Clean up old published outbox events.")
    outbox_cleanup_parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for cleanup operations.")

    symbols_status_parser = subparsers.add_parser("symbols:status", help="Show configured symbol readiness.")
    symbols_status_parser.add_argument("--symbols", help="Comma-separated list of symbols. Defaults to MARKET_DATA_PROVIDER_SYMBOLS.")
    symbols_status_parser.add_argument("--timeframes", help="Comma-separated list of timeframes. Defaults to MARKET_DATA_TIMEFRAMES.")
    symbols_status_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")

    symbols_enable_parser = subparsers.add_parser("symbols:enable", help="Sync registry, run bootstrap ticks, and optionally subscribe a bot instance.")
    symbols_enable_parser.add_argument("symbols", help="Comma-separated list of symbols to enable.")
    symbols_enable_parser.add_argument("--timeframes", help="Comma-separated list of timeframes. Defaults to MARKET_DATA_TIMEFRAMES.")
    symbols_enable_parser.add_argument("--bootstrap-ticks", type=int, default=1, help="Collection ticks to run after registry sync.")
    symbols_enable_parser.add_argument("--bot-instance-id", help="Bot instance to subscribe to the symbols/timeframes.")
    symbols_enable_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    return parser


def _run_collect_contract(args: argparse.Namespace) -> int:
    try:
        return asyncio.run(_run_collect(args.once))
    except Exception as exc:
        print(f"collect command failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


async def _run_collect(once: bool) -> int:
    if once:
        result = await run_collect_once()
        print(
            "Collection completed: "
            f"scheduled={result.scheduled_count} processed={result.processed_count} "
            f"failed={result.failed_count} published={result.published_count}"
        )
        return EXIT_SUCCESS

    from market_data_service.runtime.http_server import run_http_server
    await run_http_server()
    return EXIT_SUCCESS


async def _run_outbox_cleanup(args: argparse.Namespace) -> int:
    try:
        deleted_count = await run_outbox_cleanup(batch_size=args.batch_size)
        print(f"Outbox cleanup completed. Deleted {deleted_count} events.")
        return EXIT_SUCCESS
    except Exception as exc:
        print(f"outbox:cleanup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


def _run_candles_get_contract(args: argparse.Namespace) -> int:
    try:
        from_value, to_value, symbols, timeframes, provider = _build_candles_get_contract(args)
    except Exception as exc:
        print(f"candles:get failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    from market_data_service.application.services.candles_get_fetch_service import CandlesGetCommand
    from market_data_service.runtime.maintenance import run_candles_get

    command = CandlesGetCommand(
        from_time=from_value,
        to_time=to_value,
        symbols=symbols,
        timeframes=timeframes,
        provider=provider,
    )

    try:
        result = asyncio.run(run_candles_get(command))
    except Exception as exc:
        print(f"candles:get failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print(json.dumps(_build_candles_get_json(result), sort_keys=True, separators=(",", ":")))
    return EXIT_SUCCESS


def _run_symbols_status_contract(args: argparse.Namespace) -> int:
    try:
        symbols, timeframes = _build_symbols_status_contract(args)
        statuses = asyncio.run(run_symbols_status(symbols=symbols, timeframes=timeframes))
    except Exception as exc:
        print(f"symbols:status failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    _print_symbol_statuses(statuses, json_output=args.json)
    return EXIT_SUCCESS


def _run_symbols_enable_contract(args: argparse.Namespace) -> int:
    try:
        symbols, timeframes, bootstrap_ticks = _build_symbols_enable_contract(args)
        result = asyncio.run(
            run_symbols_enable(
                symbols=symbols,
                timeframes=timeframes,
                bootstrap_ticks=bootstrap_ticks,
                bot_instance_id=args.bot_instance_id,
            )
        )
    except Exception as exc:
        print(f"symbols:enable failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if args.json:
        print(
            json.dumps(
                {
                    "registry": {
                        "market_symbols": result.registry_sync.market_symbols,
                        "provider_symbols": result.registry_sync.provider_symbols,
                    },
                    "bootstrap_ticks": result.bootstrap_ticks,
                    "bot_instance_updated": result.bot_instance_updated,
                    "symbols": [_symbol_status_to_json(status) for status in result.statuses],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    else:
        print(
            "Enabled symbols: "
            f"registry_market_symbols={result.registry_sync.market_symbols} "
            f"registry_provider_symbols={result.registry_sync.provider_symbols} "
            f"bootstrap_ticks={result.bootstrap_ticks} "
            f"bot_instance_updated={_yes_no(result.bot_instance_updated)}"
        )
        _print_symbol_statuses(result.statuses, json_output=False)
    return EXIT_SUCCESS


def _build_candles_get_contract(
    args: argparse.Namespace,
    *,
    now_provider=None,
) -> tuple[datetime, datetime, tuple[str, ...], tuple[str, ...], str]:
    settings = load_settings()
    now = (now_provider or (lambda: datetime.now(UTC)))().astimezone(UTC)
    from_value, to_value = _calculate_period_range(args.period, now=now)
    symbols = _parse_repeated_csv(args.symbols, normalize_symbols=True) or settings.scheduler.provider_symbols
    timeframes = _parse_repeated_csv(args.timeframes, lowercase=True) or settings.scheduler.timeframes
    unsupported = [timeframe for timeframe in timeframes if timeframe not in SUPPORTED_SCHEDULE_TIMEFRAMES]
    if unsupported:
        raise ValueError(f"Unsupported timeframe(s): {', '.join(unsupported)}")
    return (from_value, to_value, symbols, timeframes, args.provider)


def _build_symbols_status_contract(args: argparse.Namespace) -> tuple[tuple[str, ...], tuple[str, ...]]:
    settings = load_settings()
    symbols = _parse_repeated_csv(args.symbols, normalize_symbols=True) or settings.scheduler.provider_symbols
    timeframes = _parse_repeated_csv(args.timeframes, lowercase=True) or settings.scheduler.timeframes
    _validate_timeframes(timeframes)
    return symbols, timeframes


def _build_symbols_enable_contract(args: argparse.Namespace) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    settings = load_settings()
    symbols = _parse_repeated_csv(args.symbols, normalize_symbols=True)
    if not symbols:
        raise ValueError("symbols must not be empty")
    timeframes = _parse_repeated_csv(args.timeframes, lowercase=True) or settings.scheduler.timeframes
    _validate_timeframes(timeframes)
    if args.bootstrap_ticks < 0:
        raise ValueError("--bootstrap-ticks must be non-negative")
    return symbols, timeframes, args.bootstrap_ticks


def _calculate_period_range(period: str, *, now: datetime) -> tuple[datetime, datetime]:
    duration = _parse_period_duration(period)
    range_to = now.astimezone(UTC)
    range_from = range_to - duration
    return range_from, range_to


def _parse_period_duration(period: str) -> timedelta:
    normalized = period.strip().lower()
    if not normalized:
        raise ValueError("period must not be empty")

    index = 0
    while index < len(normalized) and normalized[index].isdigit():
        index += 1
    if index == 0:
        raise ValueError("period must start with a positive integer")

    amount = int(normalized[:index])
    unit = normalized[index:]
    if amount <= 0:
        raise ValueError("period amount must be greater than zero")

    if unit == "h":
        return timedelta(hours=amount)
    if unit == "d":
        return timedelta(days=amount)
    if unit == "w":
        return timedelta(weeks=amount)
    if unit == "mo":
        return timedelta(days=amount * 30)
    if unit == "y":
        return timedelta(days=amount * 365)
    raise ValueError("period unit must be one of: h, d, w, mo, y")


def _build_candles_get_json(result) -> dict[str, object]:
    return {
        "from_time": _datetime_to_json(result.from_time),
        "to_time": _datetime_to_json(result.to_time),
        "symbols": list(result.symbols),
        "timeframes": list(result.timeframes),
        "provider": result.provider,
        "fetched_count": result.total_fetched_count,
        "inserted_count": result.total_inserted_count,
        "skipped_duplicate_count": result.total_skipped_duplicate_count,
        "unresolved_symbols": list(result.unresolved_symbols),
        "items": [_candles_get_item_to_json(item) for item in result.items],
    }


def _candles_get_item_to_json(item) -> dict[str, object]:
    return {
        "source": item.source.value,
        "canonical_symbol": item.canonical_symbol,
        "provider_symbol": item.provider_symbol,
        "timeframe": item.timeframe,
        "fetched_count": item.fetched_count,
        "inserted_count": item.inserted_count,
        "skipped_duplicate_count": item.skipped_duplicate_count,
    }


def _datetime_to_json(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _run_db_check() -> int:
    try:
        url = make_url(get_database_url())
        if not url.host:
            raise ValueError("database host is not configured")
        database_name = url.database or "<default>"
    except Exception as exc:
        print(f"Database check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print(f"Database configuration is valid: host={url.host} port={url.port or 5432} database={database_name}")
    return EXIT_SUCCESS


def _validate_timeframes(timeframes: tuple[str, ...]) -> None:
    unsupported = [timeframe for timeframe in timeframes if timeframe not in SUPPORTED_SCHEDULE_TIMEFRAMES]
    if unsupported:
        raise ValueError(f"Unsupported timeframe(s): {', '.join(unsupported)}")


def _parse_repeated_csv(
    value: str | None,
    *,
    lowercase: bool = False,
    normalize_symbols: bool = False,
) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(
        normalize_symbol(item) if normalize_symbols else item.strip().lower() if lowercase else item.strip().upper()
        for item in value.split(",")
        if item.strip()
    )


def _print_symbol_statuses(statuses, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps([_symbol_status_to_json(status) for status in statuses], sort_keys=True, separators=(",", ":")))
        return

    print(f"{'symbol':<14} {'provider':<10} {'status':<13} {'candles':<7} {'snapshots':<9} {'bot_subscribed':<14}")
    for status in statuses:
        print(
            f"{status.symbol:<14} {status.provider:<10} {status.status:<13} "
            f"{_yes_no(status.candles):<7} {_yes_no(status.snapshots):<9} {_yes_no(status.bot_subscribed):<14}"
        )


def _symbol_status_to_json(status) -> dict[str, object]:
    return {
        "symbol": status.symbol,
        "provider": status.provider,
        "status": status.status,
        "candles": status.candles,
        "snapshots": status.snapshots,
        "bot_subscribed": status.bot_subscribed,
    }


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _run_healthcheck() -> int:
    import httpx

    settings = load_http_settings()
    url = f"http://127.0.0.1:{settings.port}/health/ready"
    try:
        response = httpx.get(url, timeout=5.0)
    except Exception as exc:
        print(f"Healthcheck failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    if not 200 <= response.status_code < 300:
        print(f"Healthcheck failed: HTTP {response.status_code}: {response.text}", file=sys.stderr)
        return EXIT_ERROR
    print("Healthcheck succeeded.")
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
