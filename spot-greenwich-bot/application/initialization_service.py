from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from typing import Protocol


class PositionStateReader(Protocol):
    """Minimal protocol needed by the initialization service for reconciliation."""

    async def get_position_state(self, symbol: str): ...


class TradingInitializationService:
    """Own startup preparation for runtime tables, migrations, and optional reconciliation."""

    def __init__(
        self,
        *,
        table_initializer: Callable[[], Awaitable[None]],
        migration_runner: Callable[[], Awaitable[None]],
        executor: PositionStateReader | None = None,
    ) -> None:
        self.table_initializer = table_initializer
        self.migration_runner = migration_runner
        self.executor = executor

    async def initialize_runtime(
        self,
        symbols: Iterable[str] = (),
        *,
        reconcile_positions: bool = False,
    ) -> None:
        """Prepare runtime tables and optionally reconcile symbol position state."""

        await self.table_initializer()
        if not reconcile_positions or self.executor is None:
            return
        for symbol in symbols:
            await self.executor.get_position_state(symbol)

    async def create_tables(self) -> None:
        """Create runtime tables without running a trading cycle."""

        await self.table_initializer()

    async def run_migrations(self) -> None:
        """Execute SQL migrations without starting runtime services."""

        await self.migration_runner()
