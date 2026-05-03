import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from application.trading_cycle_service import SpotTradingCycleService
from domain.models import RegimeType, StrategyState, SymbolRuntimeState
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

    def get_balances(self, symbol: str, *, persisted_cost_basis: float | None = None):
        self.call_order.append(f"get_balances_{symbol.upper()}")
        return SimpleNamespace(
            base_balance=0.0,
            quote_balance=1000.0,
            reserved_quote=0.0,
            mark_price=100.0,
            cost_basis_price=persisted_cost_basis,
        )

    def get_open_orders(self, symbol: str):
        self.call_order.append(f"get_open_orders_{symbol.upper()}")
        return []


class _NoOpMarketDataProvider:
    async def get_market_context(self, symbol: str, *, persisted_cost_basis: float | None = None):
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
                "get_balances_SOLUSDT",
                "get_open_orders_SOLUSDT",
                "load_ETHUSDT",
                "get_balances_ETHUSDT",
                "get_open_orders_ETHUSDT",
            ],
        )

    async def test_initialize_clears_persisted_cost_basis_when_live_inventory_is_empty(self):
        call_order: list[str] = []
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)

        class _StateStore(_MemoryStateStore):
            async def load_symbol_state(self, symbol: str):
                self.call_order.append(f"load_{symbol.upper()}")
                return SymbolRuntimeState(
                    symbol=symbol.upper(),
                    strategy_state=StrategyState(regime=RegimeType.RANGE),
                    cost_basis_price=91.25,
                )

        cycle = SpotTradingCycleService(
            market_data_provider=_NoOpMarketDataProvider(),
            executor=_MemoryExecutor(call_order),
            notifier=_NoOpNotifier(),
            planner=planner,
            state_store=_StateStore(call_order),
            portfolio_allocator=PortfolioAllocator(DEFAULT_STRATEGY_CONFIG),
        )

        with patch("application.trading_cycle_service.ensure_candle_tables", AsyncMock()):
            await cycle.initialize(["SOLUSDT"])

        restored = planner.export_symbol_runtime("SOLUSDT")
        self.assertIsNone(restored.cost_basis_price)

    async def test_initialize_refreshes_persisted_cost_basis_from_live_inventory_snapshot(self):
        call_order: list[str] = []
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)

        class _Executor(_MemoryExecutor):
            def get_balances(self, symbol: str, *, persisted_cost_basis: float | None = None):
                self.call_order.append(f"get_balances_{symbol.upper()}")
                return SimpleNamespace(
                    base_balance=1.0,
                    quote_balance=1000.0,
                    reserved_quote=0.0,
                    mark_price=100.0,
                    cost_basis_price=105.5,
                )

        class _StateStore(_MemoryStateStore):
            async def load_symbol_state(self, symbol: str):
                self.call_order.append(f"load_{symbol.upper()}")
                return SymbolRuntimeState(
                    symbol=symbol.upper(),
                    strategy_state=StrategyState(regime=RegimeType.RANGE),
                    cost_basis_price=91.25,
                )

        cycle = SpotTradingCycleService(
            market_data_provider=_NoOpMarketDataProvider(),
            executor=_Executor(call_order),
            notifier=_NoOpNotifier(),
            planner=planner,
            state_store=_StateStore(call_order),
            portfolio_allocator=PortfolioAllocator(DEFAULT_STRATEGY_CONFIG),
        )

        with patch("application.trading_cycle_service.ensure_candle_tables", AsyncMock()):
            await cycle.initialize(["SOLUSDT"])

        restored = planner.export_symbol_runtime("SOLUSDT")
        self.assertEqual(restored.cost_basis_price, 105.5)

    async def test_initialize_logs_state_stale_for_old_runtime_snapshot(self):
        call_order: list[str] = []
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)

        class _StateStore(_MemoryStateStore):
            async def load_symbol_state(self, symbol: str):
                self.call_order.append(f"load_{symbol.upper()}")
                return SymbolRuntimeState(
                    symbol=symbol.upper(),
                    strategy_state=StrategyState(regime=RegimeType.RANGE),
                    last_cycle_completed_at=datetime.now(timezone.utc) - timedelta(hours=72),
                )

        cycle = SpotTradingCycleService(
            market_data_provider=_NoOpMarketDataProvider(),
            executor=_MemoryExecutor(call_order),
            notifier=_NoOpNotifier(),
            planner=planner,
            state_store=_StateStore(call_order),
            portfolio_allocator=PortfolioAllocator(DEFAULT_STRATEGY_CONFIG),
        )

        with patch("application.trading_cycle_service.ensure_candle_tables", AsyncMock()), patch(
            "application.initialization_service.logger.warning"
        ) as warning_log:
            await cycle.initialize(["SOLUSDT"])

        warning_messages = [call.args[0] for call in warning_log.call_args_list]
        self.assertIn("state_stale symbol=%s last_cycle_completed_at=%s threshold_hours=%s", warning_messages)
