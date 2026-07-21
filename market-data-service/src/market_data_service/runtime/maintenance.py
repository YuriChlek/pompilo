from __future__ import annotations

from market_data_service.application.services.candle_collection_service import CollectionResult
from market_data_service.application.services.candles_get_fetch_service import CandlesGetCommand, CandlesGetResult
from market_data_service.application.services.symbol_operations_service import SymbolStatus, SymbolsEnableResult
from market_data_service.runtime.container import build_runtime_container


async def run_collect_once() -> CollectionResult:
    container = await build_runtime_container()
    try:
        result = await container.workers.market_data_scheduler.run_once()
        await _commit_if_open(container.connection)
        publish_result = await container.workers.outbox_publisher.run_once()
        await _commit_if_open(container.connection)
        return CollectionResult(
            scheduled_count=result.scheduled_count,
            processed_count=result.processed_count,
            failed_count=result.failed_count,
            published_count=publish_result.published_count,
        )
    except Exception:
        await _rollback_if_open(container.connection)
        raise
    finally:
        await container.close()


async def _commit_if_open(connection) -> None:
    if connection.in_transaction():
        await connection.commit()


async def _rollback_if_open(connection) -> None:
    if connection.in_transaction():
        await connection.rollback()


async def run_candles_get(command: CandlesGetCommand) -> CandlesGetResult:
    container = await build_runtime_container()
    try:
        result = await container.services.candles_get_fetch.fetch_candles(
            symbols=command.symbols,
            timeframes=command.timeframes,
            provider=command.provider,
            from_time=command.from_time,
            to_time=command.to_time,
        )
        await _commit_if_open(container.connection)
        return result
    except Exception:
        await _rollback_if_open(container.connection)
        raise
    finally:
        await container.close()


async def run_symbols_status(*, symbols: tuple[str, ...], timeframes: tuple[str, ...]) -> tuple[SymbolStatus, ...]:
    container = await build_runtime_container()
    try:
        return await container.services.symbol_operations.list_statuses(symbols=symbols, timeframes=timeframes)
    except Exception:
        await _rollback_if_open(container.connection)
        raise
    finally:
        await container.close()


async def run_symbols_enable(
    *,
    symbols: tuple[str, ...],
    timeframes: tuple[str, ...],
    bootstrap_ticks: int,
    bot_instance_id: str | None,
) -> SymbolsEnableResult:
    container = await build_runtime_container()
    previous_symbols = container.services.candle_collection.provider_symbols
    previous_timeframes = container.services.candle_collection.timeframes
    container.services.candle_collection.provider_symbols = symbols
    container.services.candle_collection.timeframes = timeframes
    try:
        result = await container.services.symbol_operations.enable_symbols(
            symbols=symbols,
            timeframes=timeframes,
            bootstrap_ticks=bootstrap_ticks,
            bot_instance_id=bot_instance_id,
            run_bootstrap_tick=container.workers.market_data_scheduler.run_once,
        )
        await _commit_if_open(container.connection)
        return result
    except Exception:
        await _rollback_if_open(container.connection)
        raise
    finally:
        container.services.candle_collection.provider_symbols = previous_symbols
        container.services.candle_collection.timeframes = previous_timeframes
        await container.close()


async def run_outbox_cleanup(*, batch_size: int = 1000) -> int:
    container = await build_runtime_container()
    try:
        deleted_count = await container.services.outbox_cleanup.cleanup(batch_size=batch_size)
        await _commit_if_open(container.connection)
        return deleted_count
    except Exception:
        await _rollback_if_open(container.connection)
        raise
    finally:
        await container.close()
