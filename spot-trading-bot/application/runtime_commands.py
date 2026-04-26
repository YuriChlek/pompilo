from __future__ import annotations

from utils.config import (
    DEFAULT_DAILY_TARGET_HOUR,
    DEFAULT_DAILY_TARGET_MINUTE,
    DEFAULT_DAILY_TARGET_SECOND,
    SPOT_TRADING_SYMBOLS,
    THREE_YEARS_DAYS,
)


def _run_api(*, days: int):
    from api import run_api

    return run_api(days=days)


def _build_initialization_service():
    from application.bootstrap import build_initialization_service

    return build_initialization_service()


def _build_live_trading_cycle():
    from application.bootstrap import build_live_trading_cycle

    return build_live_trading_cycle()


def _build_live_trading_scheduler():
    from application.bootstrap import build_live_trading_scheduler

    return build_live_trading_scheduler()


class RuntimeCommandService:
    """Own the concrete async command handlers used by the CLI entrypoint."""

    async def sync(self, *, days: int) -> None:
        print(f"🚀 Starting D1 sync for {len(SPOT_TRADING_SYMBOLS)} symbol(s)")
        await _run_api(days=days)
        print("✅ D1 sync completed")

    async def sync_3y(self) -> None:
        print(f"🚀 Starting 3-year D1 sync for {len(SPOT_TRADING_SYMBOLS)} symbol(s)")
        await _run_api(days=THREE_YEARS_DAYS)
        print("✅ 3-year D1 sync completed")

    async def analyze(self, *, symbol: str | None = None, dry_run: bool = False) -> None:
        trading_cycle = _build_live_trading_cycle()
        initialization_service = _build_initialization_service()
        symbols = [symbol] if symbol else SPOT_TRADING_SYMBOLS
        await initialization_service.initialize_runtime(symbols)
        print(f"🚀 Starting one-off analysis for {len(symbols)} symbol(s), dry_run={dry_run}")
        for current_symbol in symbols:
            print(f"🧠 Running one-off cycle for {current_symbol}")
            await trading_cycle.run(current_symbol, dry_run=dry_run)
        print("✅ One-off analysis completed")

    async def init_db(self) -> None:
        initialization_service = _build_initialization_service()
        print("🚀 Creating spot bot tables")
        await initialization_service.create_tables()
        print("✅ Spot bot tables are ready")

    async def migrate(self) -> None:
        initialization_service = _build_initialization_service()
        print("🚀 Running SQL migrations")
        await initialization_service.run_migrations()
        print("✅ SQL migrations completed")

    async def live(self, *, dry_run: bool = False) -> None:
        scheduler = _build_live_trading_scheduler()
        print("🚀 Starting spot Greenwich bot")
        print(f"📊 Symbols: {', '.join(SPOT_TRADING_SYMBOLS)}")
        await scheduler.run_forever(
            SPOT_TRADING_SYMBOLS,
            target_hour=DEFAULT_DAILY_TARGET_HOUR,
            target_minute=DEFAULT_DAILY_TARGET_MINUTE,
            target_second=DEFAULT_DAILY_TARGET_SECOND,
            dry_run=dry_run,
        )


__all__ = [
    "RuntimeCommandService",
    "_build_initialization_service",
    "_build_live_trading_cycle",
    "_build_live_trading_scheduler",
    "_run_api",
]
