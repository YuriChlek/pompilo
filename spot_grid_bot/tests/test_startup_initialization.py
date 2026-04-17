import unittest
from unittest.mock import AsyncMock, patch

from application.trading_cycle_service import SpotTradingCycleService
from domain.portfolio_allocator import PortfolioAllocator
from domain.spot_grid_planner import SpotGridPlanner
from domain.strategy_config import DEFAULT_STRATEGY_CONFIG


class _MemoryExecutor:
    def __init__(self, call_order: list[str]) -> None:
        self.exchange = self
        self.call_order = call_order

    async def reconcile_state(self, symbols) -> None:
        self.call_order.append("reconcile_state")

    async def sync_orders(self, symbol: str, target_orders) -> bool:
        return True

    def get_balances(self, symbol: str):
        raise NotImplementedError

    def get_open_orders(self, symbol: str):
        raise NotImplementedError


class _NoOpMarketDataProvider:
    async def get_market_context(self, symbol: str):
        raise NotImplementedError


class _NoOpNotifier:
    async def notify_rebuild(self, decision) -> None:
        return None


class _MemoryStateStore:
    def __init__(self, call_order: list[str]) -> None:
        self.call_order = call_order

    async def initialize(self) -> None:
        self.call_order.append("state_initialize")

    async def load_symbol_state(self, symbol: str):
        self.call_order.append(f"load_{symbol.upper()}")
        return None

    async def save_symbol_state(self, state) -> None:
        return None


class StartupInitializationTests(unittest.IsolatedAsyncioTestCase):
    async def test_initialize_ensures_candle_tables_before_reconcile_and_state_restore(self):
        call_order: list[str] = []
        cycle = SpotTradingCycleService(
            market_data_provider=_NoOpMarketDataProvider(),
            executor=_MemoryExecutor(call_order),
            notifier=_NoOpNotifier(),
            planner=SpotGridPlanner(DEFAULT_STRATEGY_CONFIG),
            state_store=_MemoryStateStore(call_order),
            portfolio_allocator=PortfolioAllocator(DEFAULT_STRATEGY_CONFIG),
        )

        async def _ensure_tables(symbols):
            call_order.append("ensure_candle_tables")

        with patch("application.trading_cycle_service.ensure_candle_tables", AsyncMock(side_effect=_ensure_tables)):
            await cycle.initialize(["SOLUSDT", "ETHUSDT"])

        self.assertEqual(
            call_order,
            [
                "ensure_candle_tables",
                "reconcile_state",
                "state_initialize",
                "load_SOLUSDT",
                "load_ETHUSDT",
            ],
        )
