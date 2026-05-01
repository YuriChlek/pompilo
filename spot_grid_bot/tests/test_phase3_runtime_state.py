import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from application.analysis_batch_service import TradingCycleAnalysisBatchService
from application.trading_cycle_service import SpotTradingCycleService
from domain.state_machine import StrategyStateMachine
from domain.models import Candle, InventorySnapshot, MarketContext, RegimeSnapshot, RegimeType, StrategyState, SymbolRuntimeState
from domain.portfolio_allocator import PortfolioAllocator
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

    def get_balances(self, symbol: str, *, persisted_cost_basis: float | None = None):
        return InventorySnapshot(base_balance=0.0, quote_balance=1_000.0, reserved_quote=0.0, mark_price=100.0)

    def get_open_orders(self, symbol: str):
        return []


class _MemoryMarketDataProvider:
    async def get_market_context(self, symbol: str, *, persisted_cost_basis: float | None = None) -> MarketContext:
        return MarketContext(
            symbol=symbol.upper(),
            candles=_build_candles(),
            inventory=InventorySnapshot(
                base_balance=0.0,
                quote_balance=1_000.0,
                reserved_quote=0.0,
                mark_price=100.0,
                cost_basis_price=persisted_cost_basis,
            ),
            live_orders=[],
        )


class _NoOpNotifier:
    async def notify_rebuild(self, decision) -> None:
        return None


class _FailingMarketDataProvider:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def get_market_context(self, symbol: str, *, persisted_cost_basis: float | None = None) -> MarketContext:
        raise self.exc


class Phase3RuntimeStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_state_machine_returns_new_state_without_mutating_input(self):
        initial_state = StrategyState(
            regime=RegimeType.RANGE,
            bars_in_state=3,
            cooldown_remaining=0,
            volatility_cooldown_remaining=0,
        )
        machine = StrategyStateMachine(DEFAULT_STRATEGY_CONFIG, initial_state)

        updated_state = machine.on_bar(RegimeSnapshot(RegimeType.UPTREND, 1.0, ["fixed_uptrend"]))

        self.assertIsNot(updated_state, initial_state)
        self.assertEqual(initial_state.regime, RegimeType.RANGE)
        self.assertEqual(initial_state.bars_in_state, 3)
        self.assertEqual(initial_state.pending_regime, None)
        self.assertEqual(initial_state.pending_count, 0)
        self.assertEqual(updated_state.pending_regime, RegimeType.UPTREND)
        self.assertEqual(updated_state.pending_count, 1)

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

    async def test_planner_analyze_does_not_commit_runtime_state(self):
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        planner.detector.detect = Mock(return_value=RegimeSnapshot(RegimeType.UPTREND, 1.0, ["fixed_uptrend"]))
        planner.restore_symbol_runtime(
            SymbolRuntimeState(
                symbol="SOLUSDT",
                strategy_state=StrategyState(regime=RegimeType.RANGE, bars_in_state=5),
            )
        )
        context = MarketContext("SOLUSDT", _build_candles(100.0), InventorySnapshot(0.0, 1000.0, 0.0, 100.0), [])

        analysis = planner.analyze(context)
        runtime_after_analyze = planner.export_symbol_runtime("SOLUSDT")

        self.assertEqual(runtime_after_analyze.strategy_state.regime, RegimeType.RANGE)
        self.assertEqual(runtime_after_analyze.strategy_state.bars_in_state, 5)
        self.assertEqual(analysis.strategy_state.pending_regime, RegimeType.UPTREND)
        self.assertEqual(analysis.strategy_state.pending_count, 1)

    async def test_trading_cycle_restores_and_saves_symbol_state(self):
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        planner.detector.detect = Mock(return_value=RegimeSnapshot(RegimeType.RANGE, 1.0, ["fixed_range"]))
        state_store = _MemoryStateStore()
        state_store.data["SOLUSDT"] = SymbolRuntimeState(
            symbol="SOLUSDT",
            strategy_state=StrategyState(regime=RegimeType.DOWNTREND, last_rebuild_price=88.5),
            cost_basis_price=91.25,
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
        self.assertEqual(restored.cost_basis_price, 91.25)

        await cycle.run("SOLUSDT")

        self.assertTrue(state_store.initialized)
        self.assertIn("SOLUSDT", state_store.data)
        self.assertEqual(state_store.data["SOLUSDT"].cost_basis_price, 91.25)

    async def test_analysis_logs_infrastructure_failures_as_critical(self):
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        analysis_service = TradingCycleAnalysisBatchService(
            _FailingMarketDataProvider(OSError("network down")),
            planner,
            PortfolioAllocator(DEFAULT_STRATEGY_CONFIG),
        )

        with patch("application.analysis_batch_service.logger.critical") as critical_log, patch(
            "application.analysis_batch_service.logger.exception"
        ) as exception_log:
            results, context_by_symbol, analysis_by_symbol, allocation_plan = await analysis_service.analyze(
                ["SOLUSDT"],
                set(),
            )

        self.assertEqual(results, {"SOLUSDT": None})
        self.assertEqual(context_by_symbol, {})
        self.assertEqual(analysis_by_symbol, {})
        self.assertIsNone(allocation_plan)
        critical_log.assert_called_once()
        exception_log.assert_not_called()

    async def test_analysis_logs_domain_failures_as_exception(self):
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        analysis_service = TradingCycleAnalysisBatchService(
            _FailingMarketDataProvider(ValueError("bad payload")),
            planner,
            PortfolioAllocator(DEFAULT_STRATEGY_CONFIG),
        )

        with patch("application.analysis_batch_service.logger.critical") as critical_log, patch(
            "application.analysis_batch_service.logger.exception"
        ) as exception_log:
            results, context_by_symbol, analysis_by_symbol, allocation_plan = await analysis_service.analyze(
                ["SOLUSDT"],
                set(),
            )

        self.assertEqual(results, {"SOLUSDT": None})
        self.assertEqual(context_by_symbol, {})
        self.assertEqual(analysis_by_symbol, {})
        self.assertIsNone(allocation_plan)
        critical_log.assert_not_called()
        exception_log.assert_called_once()

    async def test_planning_logs_infrastructure_failures_as_critical(self):
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        cycle = SpotTradingCycleService(
            market_data_provider=_MemoryMarketDataProvider(),
            executor=_MemoryExecutor(),
            notifier=_NoOpNotifier(),
            planner=planner,
            state_store=None,
        )
        fake_context = MarketContext("SOLUSDT", _build_candles(), InventorySnapshot(0.0, 1000.0, 0.0, 100.0), [])
        fake_analysis = SimpleNamespace(symbol="SOLUSDT")
        fake_plan = SimpleNamespace(budget_for=lambda _symbol: None)

        cycle._analysis_service.analyze = AsyncMock(
            return_value=(
                {},
                {"SOLUSDT": fake_context},
                {"SOLUSDT": fake_analysis},
                fake_plan,
            )
        )
        cycle.planner.plan_from_analysis = Mock(side_effect=OSError("db unavailable"))

        with patch("application.trading_cycle_service.logger.critical") as critical_log, patch(
            "application.trading_cycle_service.logger.exception"
        ) as exception_log:
            results = await cycle.run_many(["SOLUSDT"])

        self.assertEqual(results, {"SOLUSDT": None})
        critical_log.assert_called_once()
        exception_log.assert_not_called()

    async def test_planning_logs_domain_failures_as_exception(self):
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        cycle = SpotTradingCycleService(
            market_data_provider=_MemoryMarketDataProvider(),
            executor=_MemoryExecutor(),
            notifier=_NoOpNotifier(),
            planner=planner,
            state_store=None,
        )
        fake_context = MarketContext("SOLUSDT", _build_candles(), InventorySnapshot(0.0, 1000.0, 0.0, 100.0), [])
        fake_analysis = SimpleNamespace(symbol="SOLUSDT")
        fake_plan = SimpleNamespace(budget_for=lambda _symbol: None)

        cycle._analysis_service.analyze = AsyncMock(
            return_value=(
                {},
                {"SOLUSDT": fake_context},
                {"SOLUSDT": fake_analysis},
                fake_plan,
            )
        )
        cycle.planner.plan_from_analysis = Mock(side_effect=ValueError("bad state"))

        with patch("application.trading_cycle_service.logger.critical") as critical_log, patch(
            "application.trading_cycle_service.logger.exception"
        ) as exception_log:
            results = await cycle.run_many(["SOLUSDT"])

        self.assertEqual(results, {"SOLUSDT": None})
        critical_log.assert_not_called()
        exception_log.assert_called_once()
