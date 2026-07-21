from __future__ import annotations

import argparse
import asyncio
import signal
from collections.abc import Sequence
from urllib.error import URLError
from urllib.request import urlopen

from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from bot_platform_service.application import BotPlatformRunnerService
from bot_platform_service.application.market_data_event_consumer_service import MarketDataEventConsumerService
from bot_platform_service.application.market_data_event_run_dispatcher_service import MarketDataEventRunDispatcherService
from bot_platform_service.config.database_config import get_database_url
from bot_platform_service.config.settings import BotPlatformSettings
from bot_platform_service.infrastructure.market_data import build_redis_market_data_event_consumer
from bot_platform_service.infrastructure.market_data.http_snapshot_client import MarketDataHttpSnapshotClient
from bot_platform_service.persistence.repositories.bot_module_repository import BotModuleRepository
from bot_platform_service.registry import BotModuleRegistry, discover_and_register_trading_bots
from bot_platform_service.runtime import BotPlatformHttpServer, build_runtime_container
from bot_platform_service.observability.structured_logging import JsonStructuredLogger
from bot_platform_service.workers import MarketDataEventConsumerWorker
from bot_platform_service.workers.bot_platform_runner import BotPlatformRunner


async def sync_bot_modules(connection: AsyncConnection) -> tuple[str, ...]:
    """Discover platform-native bot modules and persist their metadata."""
    repository = BotModuleRepository(connection)
    registry = BotModuleRegistry(repository)
    return await discover_and_register_trading_bots(registry)


async def sync_bot_modules_from_database_url(database_url: str) -> tuple[str, ...]:
    """Open a database transaction and sync platform-native bot module metadata."""
    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as connection:
            return await sync_bot_modules(connection)
    finally:
        await engine.dispose()


async def cleanup_event_data() -> tuple[int, int]:
    """Run one cleanup batch for processed market-data event records."""

    container = await build_runtime_container()
    try:
        if container.event_data_cleanup_service is None:
            return (0, 0)
        result = await container.event_data_cleanup_service.run_once()
        if container.connection.in_transaction():
            await container.connection.commit()
        return (result.processed_event_deleted_count, result.event_audit_deleted_count)
    except Exception:
        if container.connection.in_transaction():
            await container.connection.rollback()
        raise
    finally:
        await container.close()


async def serve_http(settings: BotPlatformSettings | None = None) -> None:
    """Start the Bot Platform HTTP API until the process receives a stop signal."""
    container = await build_runtime_container(settings)
    server = BotPlatformHttpServer(container=container)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for stop_signal in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(stop_signal, stop_event.set)

    try:
        await server.start()
        print(f"Bot Platform HTTP API listening on {server.host}:{server.bound_port}")
        await stop_event.wait()
    finally:
        await server.stop()
        await container.close()


async def run_runner(settings: BotPlatformSettings | None = None) -> None:
    """Start the Bot Platform runner skeleton until the process receives a stop signal."""

    resolved_settings = settings or BotPlatformSettings.from_env()
    container = await build_runtime_container(resolved_settings)
    market_data = MarketDataHttpSnapshotClient(base_url=resolved_settings.market_data.base_url)
    service = BotPlatformRunnerService(
        instance_repository=container.repositories.bot_instances,
        market_data=market_data,
        source=resolved_settings.market_data.source,
    )
    runner = BotPlatformRunner(
        service=service,
        poll_interval_seconds=resolved_settings.runtime.runner_poll_interval_seconds,
    )
    event_consumer_worker = _build_market_data_event_consumer_worker(
        resolved_settings,
        idempotency_store=container.repositories.market_data_events,
        instance_repository=container.repositories.bot_instances,
        event_run_service=container.manual_run_service,
        commit=container.connection.commit,
        rollback=container.connection.rollback,
    )
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for stop_signal in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(stop_signal, stop_event.set)

    runner_task = asyncio.create_task(runner.run_forever())
    event_consumer_task = (
        asyncio.create_task(event_consumer_worker.run_forever()) if event_consumer_worker is not None else None
    )
    try:
        print("Bot Platform runner started.")
        await stop_event.wait()
    finally:
        runner.stop()
        if event_consumer_worker is not None:
            event_consumer_worker.stop()
        await runner_task
        if event_consumer_task is not None:
            await event_consumer_task
        await container.close()


def _build_market_data_event_consumer_worker(
    settings: BotPlatformSettings,
    *,
    idempotency_store,
    instance_repository,
    event_run_service,
    commit=None,
    rollback=None,
) -> MarketDataEventConsumerWorker | None:
    if not settings.market_data_events.enabled:
        return None
    stream_consumer = build_redis_market_data_event_consumer(
        redis_url=settings.market_data_events.redis_url,
        stream_name=settings.market_data_events.stream_name,
        consumer_group=settings.market_data_events.consumer_group,
        consumer_name=settings.market_data_events.consumer_name,
        read_count=settings.market_data_events.read_count,
        block_milliseconds=settings.market_data_events.block_milliseconds,
    )
    return MarketDataEventConsumerWorker(
        stream_consumer=stream_consumer,
        service=MarketDataEventConsumerService(
            logger=JsonStructuredLogger(),
            idempotency_store=idempotency_store,
            instance_repository=instance_repository,
            run_dispatcher=MarketDataEventRunDispatcherService(event_run_service=event_run_service),
        ),
        retry_backoff_seconds=settings.market_data_events.retry_backoff_seconds,
        commit=commit,
        rollback=rollback,
    )


def run_healthcheck(url: str, *, timeout_seconds: float) -> int:
    """Check Bot Platform HTTP readiness through the same API Docker uses."""
    try:
        with urlopen(url, timeout=timeout_seconds) as response:
            if response.status == 200:
                return 0
    except URLError:
        return 1
    return 1


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Run the Bot Platform command line interface."""
    parser = argparse.ArgumentParser(prog="bot-platform")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("bot:modules:sync", help="Discover and persist platform-native bot modules")
    subparsers.add_parser("events:cleanup", help="Clean up old processed market-data event records")
    subparsers.add_parser("serve", help="Start the Bot Platform HTTP API")
    subparsers.add_parser("runner", help="Start the Bot Platform runner skeleton")
    healthcheck_parser = subparsers.add_parser("healthcheck", help="Check HTTP readiness")
    healthcheck_parser.add_argument("--url", default=None, help="Readiness URL to check")
    healthcheck_parser.add_argument("--timeout", type=float, default=2.0, help="Request timeout in seconds")

    args = parser.parse_args(list(argv or ()))
    if args.command is None:
        return 0

    if args.command == "bot:modules:sync":
        module_ids = asyncio.run(sync_bot_modules_from_database_url(get_database_url()))
        print(f"Synced bot modules: {', '.join(module_ids) if module_ids else '(none)'}")
        return 0

    if args.command == "events:cleanup":
        processed_count, audit_count = asyncio.run(cleanup_event_data())
        print(
            "Event data cleanup completed: "
            f"processed_events_deleted={processed_count} event_audit_deleted={audit_count}"
        )
        return 0

    if args.command == "serve":
        asyncio.run(serve_http())
        return 0

    if args.command == "runner":
        asyncio.run(run_runner())
        return 0

    if args.command == "healthcheck":
        settings = BotPlatformSettings.from_env()
        url = args.url or f"http://127.0.0.1:{settings.http.port}/health/ready"
        return run_healthcheck(url, timeout_seconds=args.timeout)

    parser.error(f"Unsupported command: {args.command}")
    return 2
