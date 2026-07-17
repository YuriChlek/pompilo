from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy.engine import make_url

from market_data_service.application.services.backfill_command_service import BackfillCommand
from market_data_service.application.services.gap_scan_service import GapScanCommand, GapScanResult
from market_data_service.application.services.outbox_replay_service import OutboxReplayCommand, OutboxReplayResult
from market_data_service.config.database_config import get_database_url
from market_data_service.config.settings import load_backfill_settings, load_http_settings, load_settings
from market_data_service.domain.enums import MarketDataSource
from market_data_service.runtime.http_server import run_http_server
from market_data_service.runtime.maintenance import (
    run_backfill,
    run_gaps_scan,
    run_outbox_publish_once,
    run_outbox_replay,
    run_scheduler_once,
    run_sync_next_job,
    run_symbols_sync,
)

EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_UNSUPPORTED = 78

_UNIMPLEMENTED_COMMANDS = {
    "scheduler",
    "outbox-publisher",
    "db:revision",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command in _UNIMPLEMENTED_COMMANDS:
        print(f"Command '{args.command}' is not implemented yet.", file=sys.stderr)
        return EXIT_UNSUPPORTED
    if args.command == "serve":
        return asyncio.run(_run_serve())
    if args.command == "healthcheck":
        return _run_healthcheck()
    if args.command == "db:check":
        return _run_db_check()
    if args.command == "symbols:sync":
        return asyncio.run(_run_symbols_sync())
    if args.command == "scheduler:run-once":
        return asyncio.run(_run_scheduler_once())
    if args.command == "sync:run-next":
        return asyncio.run(_run_sync_next_job())
    if args.command == "outbox:publish-once":
        return asyncio.run(_run_outbox_publish_once())
    if args.command == "backfill":
        return asyncio.run(_run_backfill(args))
    if args.command == "gaps:scan":
        return asyncio.run(_run_gaps_scan(args))
    if args.command == "outbox:replay":
        return asyncio.run(_run_outbox_replay(args))

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
    parser.add_argument(
        "command",
        nargs="?",
        choices=[
            "serve",
            "scheduler",
            "outbox-publisher",
            "backfill",
            "healthcheck",
            "db:check",
            "db:revision",
            "scheduler:run-once",
            "sync:run-next",
            "outbox:publish-once",
            "symbols:sync",
            "gaps:scan",
            "outbox:replay",
        ],
        help="Command to execute.",
    )
    parser.add_argument("--symbol", dest="backfill_symbol", help="Provider symbol for the backfill command.")
    parser.add_argument("--timeframe", dest="backfill_timeframe", help="Candle timeframe for the backfill command.")
    parser.add_argument("--from", dest="backfill_from", help="Inclusive backfill range start in ISO-8601 format.")
    parser.add_argument("--to", dest="backfill_to", help="Exclusive backfill range end in ISO-8601 format.")
    parser.add_argument("--batch-size", dest="backfill_batch_size", type=int, help="Maximum candles per backfill chunk.")
    parser.add_argument("--max-concurrency", dest="backfill_max_concurrency", type=int, help="Maximum backfill concurrency.")
    parser.add_argument("--create-backfill", action="store_true", help="Create backfill requests for gaps found by gaps:scan.")
    parser.add_argument("--from-id", dest="outbox_from_id", help="Inclusive outbox event id lower bound for outbox:replay.")
    parser.add_argument("--to-id", dest="outbox_to_id", help="Inclusive outbox event id upper bound for outbox:replay.")
    parser.add_argument("--dry-run", action="store_true", help="Report outbox replay candidates without publishing.")
    return parser


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


async def _run_symbols_sync() -> int:
    try:
        result = await run_symbols_sync()
    except Exception as exc:
        print(f"Symbols sync failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print(
        "Symbols sync completed: "
        f"market_symbols={result.market_symbols} provider_symbols={result.provider_symbols}"
    )
    return EXIT_SUCCESS


async def _run_scheduler_once() -> int:
    try:
        result = await run_scheduler_once()
    except Exception as exc:
        print(f"Scheduler run-once failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print(
        "Scheduler run-once completed: "
        f"created={result.created_count} skipped={result.skipped_count}"
    )
    return EXIT_SUCCESS


async def _run_sync_next_job() -> int:
    try:
        result = await run_sync_next_job()
    except Exception as exc:
        print(f"Sync run-next failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if not result.job_found:
        print("Sync run-next completed: no pending job")
        return EXIT_SUCCESS

    assert result.sync_result is not None
    sync = result.sync_result
    print(
        "Sync run-next completed: "
        f"batch_id={sync.batch_id} fetched={sync.fetched_count} inserted={sync.inserted_count} "
        f"skipped_duplicates={sync.skipped_duplicate_count} snapshot_id={sync.snapshot_id or ''}"
    )
    return EXIT_SUCCESS


async def _run_outbox_publish_once() -> int:
    try:
        result = await run_outbox_publish_once()
    except Exception as exc:
        print(f"Outbox publish-once failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print(
        "Outbox publish-once completed: "
        f"fetched={result.fetched_count} published={result.published_count} "
        f"retry={result.retry_count} failed={result.failed_count}"
    )
    return EXIT_SUCCESS


async def _run_backfill(args: argparse.Namespace) -> int:
    try:
        command = _build_backfill_command(args)
        result = await run_backfill(command)
    except Exception as exc:
        print(f"Backfill failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print(
        "Backfill requested: "
        f"chunks={result.chunk_count} requested={result.requested_count} "
        f"skipped_duplicates={result.skipped_duplicate_count} "
        f"batch_size_candles={result.batch_size_candles} max_concurrency={result.max_concurrency}"
    )
    return EXIT_SUCCESS


def _build_backfill_command(args: argparse.Namespace) -> BackfillCommand:
    settings = load_backfill_settings()
    symbol = args.backfill_symbol
    timeframe = args.backfill_timeframe
    from_value = args.backfill_from
    to_value = args.backfill_to
    if not symbol or not timeframe or not from_value or not to_value:
        raise ValueError("backfill requires --symbol, --timeframe, --from and --to")

    return BackfillCommand(
        source=MarketDataSource.BINANCE_SPOT,
        provider_symbol=symbol,
        timeframe=timeframe,
        from_time=_parse_datetime(from_value),
        to_time=_parse_datetime(to_value),
        batch_size_candles=args.backfill_batch_size or settings.batch_size_candles,
        max_concurrency=args.backfill_max_concurrency or settings.max_concurrency,
    )


def _parse_datetime(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


async def _run_gaps_scan(args: argparse.Namespace) -> int:
    try:
        command = _build_gaps_scan_command(args)
        result = await run_gaps_scan(command)
    except Exception as exc:
        print(f"Gaps scan failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print(_format_gaps_scan_report(result))
    return EXIT_SUCCESS


def _build_gaps_scan_command(args: argparse.Namespace) -> GapScanCommand:
    settings = load_settings()
    from_value = args.backfill_from
    to_value = args.backfill_to
    if not from_value or not to_value:
        raise ValueError("gaps:scan requires --from and --to")

    provider_symbols = _parse_repeated_csv(args.backfill_symbol) or settings.scheduler.provider_symbols
    timeframes = _parse_repeated_csv(args.backfill_timeframe, lowercase=True) or settings.scheduler.timeframes
    return GapScanCommand(
        source=MarketDataSource.BINANCE_SPOT,
        provider_symbols=provider_symbols,
        timeframes=timeframes,
        from_time=_parse_datetime(from_value),
        to_time=_parse_datetime(to_value),
        create_backfill=args.create_backfill,
    )


def _parse_repeated_csv(value: str | None, *, lowercase: bool = False) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(
        item.strip().lower() if lowercase else item.strip().upper()
        for item in value.split(",")
        if item.strip()
    )


def _format_gaps_scan_report(result: GapScanResult) -> str:
    lines = [
        "Gaps scan report:",
        f"create_backfill={str(result.create_backfill).lower()}",
        (
            "totals: "
            f"symbols={len(result.reports)} gaps={result.total_gap_count} "
            f"backfill_requested={result.total_backfill_requested_count} "
            f"backfill_skipped_duplicates={result.total_backfill_skipped_duplicate_count}"
        ),
    ]
    for report in result.reports:
        lines.append(
            " - "
            f"{report.provider_symbol} {report.timeframe} status={report.status.value} "
            f"expected={report.expected_count} actual={report.actual_count} gaps={report.gap_count} "
            f"range={report.from_time.isoformat()}..{report.to_time.isoformat()} "
            f"backfill_requested={report.backfill_requested_count} "
            f"backfill_skipped_duplicates={report.backfill_skipped_duplicate_count}"
        )
        for interval in report.missing_intervals:
            lines.append(f"   missing {interval.open_time.isoformat()}..{interval.close_time.isoformat()}")
    return "\n".join(lines)


async def _run_outbox_replay(args: argparse.Namespace) -> int:
    try:
        result = await run_outbox_replay(
            OutboxReplayCommand(
                from_id=args.outbox_from_id,
                to_id=args.outbox_to_id,
                dry_run=args.dry_run,
            )
        )
    except Exception as exc:
        print(f"Outbox replay failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print(_format_outbox_replay_report(result))
    return EXIT_SUCCESS


def _format_outbox_replay_report(result: OutboxReplayResult) -> str:
    return (
        "Outbox replay report:\n"
        f"dry_run={str(result.dry_run).lower()}\n"
        f"matched={result.matched_count} publishable={result.publishable_count} "
        f"published={result.published_count} skipped_published={result.skipped_published_count}"
    )


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
