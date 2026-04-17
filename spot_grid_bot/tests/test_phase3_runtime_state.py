import unittest
from unittest.mock import AsyncMock, Mock, patch

from application.trading_cycle_service import SpotTradingCycleService
from domain.models import Candle, InventorySnapshot, MarketContext, RegimeSnapshot, RegimeType, StrategyState, SymbolRuntimeState
from domain.spot_grid_planner import SpotGridPlanner
from domain.strategy_config import DEFAULT_STRATEGY_CONFIG


def _build_candles(price: float = 100.0) -> list[Candle]:
    candles: list[Candle] = []
    current = price
    for index in range(260):
        candles.append(Candle(timestamp=index, open=current, high=current + 1.0, low=current - 1.0, close=current, volume=10.0))
    return candles


class _MemoryStateStore:
    def __init__(self) -> None:
        self.initialized = False
        self.data: dict[str, SymbolRuntimeState] = {}

    async def initialize(self) -> None:
        self.initialized = True

    async def load_symbol_state(self, symbol: str) -> SymbolRuntimeState | None:
        return self.data.get(symbol.upper())

    async def save_symbol_state(self, state: SymbolRuntimeState) -> None:
        self.data[state.symbol.upper()] = state


class _MemoryExecutor:
    def __init__(self) -> None:
        self.exchange = self
        self.reconciled_symbols: list[str] = []

    async def reconcile_state(self, symbols) -> None:
        self.reconciled_symbols = list(symbols)

    async def sync_orders(self, symbol: str, target_orders) -> bool:
        return True

    def get_balances(self, symbol: str):
        return InventorySnapshot(base_balance=0.0, quote_balance=1_000.0, reserved_quote=0.0, mark_price=100.0)

    def get_open_orders(self, symbol: str):
        return []


class _MemoryMarketDataProvider:
    async def get_market_context(self, symbol: str) -> MarketContext:
        return MarketContext(
            symbol=symbol.upper(),
            candles=_build_candles(),
            inventory=InventorySnapshot(base_balance=0.0, quote_balance=1_000.0, reserved_quote=0.0, mark_price=100.0),
            live_orders=[],
        )


class _NoOpNotifier:
    async def notify_rebuild(self, decision) -> None:
        return None


class Phase3RuntimeStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_planner_keeps_runtime_state_isolated_per_symbol(self):
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        planner.detector.detect = Mock(return_value=RegimeSnapshot(RegimeType.RANGE, 1.0, ["fixed_range"]))

        context_a = MarketContext("SOLUSDT", _build_candles(100.0), InventorySnapshot(0.0, 1000.0, 0.0, 100.0), [])
        context_b = MarketContext("ETHUSDT", _build_candles(200.0), InventorySnapshot(0.0, 1000.0, 0.0, 200.0), [])

        planner.plan(context_a)
        planner.plan(context_b)

        state_a = planner.export_symbol_runtime("SOLUSDT")
        state_b = planner.export_symbol_runtime("ETHUSDT")

        self.assertEqual(state_a.symbol, "SOLUSDT")
        self.assertEqual(state_b.symbol, "ETHUSDT")
        self.assertNotEqual(id(state_a.strategy_state), id(state_b.strategy_state))

    async def test_trading_cycle_restores_and_saves_symbol_state(self):
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        planner.detector.detect = Mock(return_value=RegimeSnapshot(RegimeType.RANGE, 1.0, ["fixed_range"]))
        state_store = _MemoryStateStore()
        state_store.data["SOLUSDT"] = SymbolRuntimeState(
            symbol="SOLUSDT",
            strategy_state=StrategyState(regime=RegimeType.DOWNTREND, last_rebuild_price=88.5),
        )
        cycle = SpotTradingCycleService(
            market_data_provider=_MemoryMarketDataProvider(),
            executor=_MemoryExecutor(),
            notifier=_NoOpNotifier(),
            planner=planner,
            state_store=state_store,
        )

        with patch("application.trading_cycle_service.ensure_candle_tables", AsyncMock()):
            await cycle.initialize(["SOLUSDT"])
        restored = planner.export_symbol_runtime("SOLUSDT")
        self.assertEqual(restored.strategy_state.regime, RegimeType.DOWNTREND)
        self.assertEqual(restored.strategy_state.last_rebuild_price, 88.5)

        await cycle.run("SOLUSDT")

        self.assertTrue(state_store.initialized)
        self.assertIn("SOLUSDT", state_store.data)
