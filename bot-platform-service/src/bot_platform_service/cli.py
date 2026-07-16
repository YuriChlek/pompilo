from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from bot_platform_service.config.database_config import get_database_url
from bot_platform_service.persistence.repositories.bot_module_repository import BotModuleRepository
from bot_platform_service.registry import BotModuleRegistry, discover_and_register_trading_bots


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


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Run the Bot Platform command line interface."""
    parser = argparse.ArgumentParser(prog="bot-platform")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("bot:modules:sync", help="Discover and persist platform-native bot modules")

    args = parser.parse_args(list(argv or ()))
    if args.command is None:
        return 0

    if args.command == "bot:modules:sync":
        module_ids = asyncio.run(sync_bot_modules_from_database_url(get_database_url()))
        print(f"Synced bot modules: {', '.join(module_ids) if module_ids else '(none)'}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
