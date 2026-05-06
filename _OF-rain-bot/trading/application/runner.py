from __future__ import annotations

"""Canonical application runner."""

from trading.application.runtime import OrderFlowScalpBot

__all__ = ["CanonicalTradingRuntime", "run_trading_application"]


class CanonicalTradingRuntime:
    """Stable canonical runtime object returned by bootstrap."""

    def __init__(self, bot: OrderFlowScalpBot, *, dry_run: bool) -> None:
        self.bot = bot
        self.dry_run = dry_run

    async def start(self) -> None:
        await self.bot.start()

    async def close(self) -> None:
        await self.bot.close()


async def run_trading_application(*, dry_run: bool = False) -> None:
    """Canonical async entrypoint target for the migrated runtime."""
    from .bootstrap import build_trading_runtime

    runtime = build_trading_runtime(dry_run=dry_run)
    try:
        await runtime.start()
    finally:
        await runtime.close()
