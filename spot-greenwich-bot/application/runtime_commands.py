from __future__ import annotations

import logging

from utils.config import (
    BINANCE_D1_INTERVAL,
    BINANCE_H4_INTERVAL,
    D1_TABLE_SUFFIX,
    DEFAULT_DAILY_TARGET_HOUR,
    DEFAULT_DAILY_TARGET_MINUTE,
    DEFAULT_DAILY_TARGET_SECOND,
    DEFAULT_LOOKBACK_DAYS,
    H4_ANALYSIS_DAYS,
    H4_TABLE_SUFFIX,
    SPOT_TRADING_SYMBOLS,
    THREE_YEARS_DAYS,
)

logger = logging.getLogger(__name__)


def _run_api(*, days: int, timeframe: str = BINANCE_D1_INTERVAL, table_suffix: str = D1_TABLE_SUFFIX):
    from api import run_api

    return run_api(days=days, timeframe=timeframe, table_suffix=table_suffix)


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
        logger.info("sync_d1_started symbols=%s days=%s", len(SPOT_TRADING_SYMBOLS), days)
        await _run_api(days=days, timeframe=BINANCE_D1_INTERVAL, table_suffix=D1_TABLE_SUFFIX)
        logger.info("sync_d1_completed")

    async def sync_3y(self) -> None:
        logger.info("sync_3y_started symbols=%s days=%s", len(SPOT_TRADING_SYMBOLS), THREE_YEARS_DAYS)
        await _run_api(days=THREE_YEARS_DAYS, timeframe=BINANCE_D1_INTERVAL, table_suffix=D1_TABLE_SUFFIX)
        logger.info("sync_3y_completed")

    async def sync_4h(self, *, days: int = H4_ANALYSIS_DAYS) -> None:
        logger.info("sync_h4_started symbols=%s days=%s", len(SPOT_TRADING_SYMBOLS), days)
        await _run_api(days=days, timeframe=BINANCE_H4_INTERVAL, table_suffix=H4_TABLE_SUFFIX)
        logger.info("sync_h4_completed")

    async def sync_full(self) -> None:
        logger.info("sync_full_started symbols=%s", len(SPOT_TRADING_SYMBOLS))
        await _run_api(days=DEFAULT_LOOKBACK_DAYS, timeframe=BINANCE_D1_INTERVAL, table_suffix=D1_TABLE_SUFFIX)
        await _run_api(days=H4_ANALYSIS_DAYS, timeframe=BINANCE_H4_INTERVAL, table_suffix=H4_TABLE_SUFFIX)
        logger.info("sync_full_completed")

    async def analyze(self, *, symbol: str | None = None, dry_run: bool = False, timeframe: str = "4h") -> None:
        trading_cycle = _build_live_trading_cycle()
        initialization_service = _build_initialization_service()
        symbols = [symbol] if symbol else SPOT_TRADING_SYMBOLS
        await initialization_service.initialize_runtime(symbols)
        logger.info("analysis_started symbols=%s dry_run=%s timeframe=%s", len(symbols), dry_run, timeframe)
        for current_symbol in symbols:
            logger.info("analysis_symbol_started symbol=%s", current_symbol)
            await trading_cycle.run(current_symbol, dry_run=dry_run)
        logger.info("analysis_completed")

    async def init_db(self) -> None:
        initialization_service = _build_initialization_service()
        logger.info("init_db_started")
        await initialization_service.create_tables()
        logger.info("init_db_completed")

    async def migrate(self) -> None:
        initialization_service = _build_initialization_service()
        logger.info("migrations_started")
        await initialization_service.run_migrations()
        logger.info("migrations_completed")

    async def live(self, *, dry_run: bool = False) -> None:
        scheduler = _build_live_trading_scheduler()
        logger.info("live_started symbols=%s", ",".join(SPOT_TRADING_SYMBOLS))
        await scheduler.run_forever(
            SPOT_TRADING_SYMBOLS,
            target_hour=DEFAULT_DAILY_TARGET_HOUR,
            target_minute=DEFAULT_DAILY_TARGET_MINUTE,
            target_second=DEFAULT_DAILY_TARGET_SECOND,
            dry_run=dry_run,
            reconcile_positions=True,
        )


__all__ = [
    "RuntimeCommandService",
    "_build_initialization_service",
    "_build_live_trading_cycle",
    "_build_live_trading_scheduler",
    "_run_api",
]
