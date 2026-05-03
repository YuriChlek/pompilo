import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from application.execution_service import TradingCycleExecutionService
from application.trading_cycle_service import SpotTradingCycleService
from domain.models import (
    Candle,
    DeRiskMode,
    IndicatorSnapshot,
    InventorySnapshot,
    MarketContext,
    RegimeSnapshot,
    RegimeType,
    RiskDecision,
    StrategyDecision,
    StrategyState,
    SymbolRuntimeState,
)
from domain.spot_grid_planner import SpotGridPlanner
from domain.strategy_config import DEFAULT_STRATEGY_CONFIG


def _build_candles(price: float = 100.0) -> list[Candle]:
    candles: list[Candle] = []
    current = price
    for index in range(260):
        candles.append(Candle(timestamp=index, open=current, high=current + 1.0, low=current - 1.0, close=current, volume=10.0))
    return candles


def _decision(*, rebuild_required: bool) -> StrategyDecision:
    return StrategyDecision(
        symbol="SOLUSDT",
        regime=RegimeType.RANGE,
        target_orders=[],
        live_orders=[],
        indicators=IndicatorSnapshot(
            ema20=100.0,
            ema50=100.0,
            ema200=100.0,
            atr14=2.0,
            realized_volatility=0.01,
            ema50_slope=0.0,
            range_width=0.02,
            price_vs_ema50=0.0,
            directional_move=0.0,
            directional_sign=0.0,
            abnormal_candle=False,
            atr_spike=False,
        ),
        risk=RiskDecision(
            can_trade=True,
            pause_entries=False,
            force_risk_off=False,
            cancel_entries=False,
            allow_exit_only=False,
            de_risk_mode=DeRiskMode.NONE,
            reasons=[],
        ),
        rebuild_required=rebuild_required,
        target_order_diff_count=0,
        reasons=["test"],
    )


class _MemoryStateStore:
    def __init__(self) -> None:
        self.data: dict[str, SymbolRuntimeState] = {}

    async def initialize(self) -> None:
        return None

    async def load_symbol_state(self, symbol: str) -> SymbolRuntimeState | None:
        return self.data.get(symbol.upper())

    async def save_symbol_state(self, state: SymbolRuntimeState) -> None:
        self.data[state.symbol.upper()] = state


class _RecoveryExecutor:
    def __init__(self, *, base_balance: float = 1.0, cost_basis_price: float | None = None):
        self.exchange = self
        self.base_balance = base_balance
        self.cost_basis_price = cost_basis_price

    async def reconcile_state(self, symbols) -> None:
        return None

    async def sync_orders(self, symbol: str, target_orders) -> bool:
        return True

    def get_balances(self, symbol: str, *, persisted_cost_basis: float | None = None):
        return InventorySnapshot(
            base_balance=self.base_balance,
            quote_balance=1000.0,
            reserved_quote=0.0,
            mark_price=100.0,
            cost_basis_price=self.cost_basis_price if self.cost_basis_price is not None else persisted_cost_basis,
        )

    def get_open_orders(self, symbol: str):
        return []


class _RecoveryMarketDataProvider:
    def __init__(self, *, mark_price: float = 100.0):
        self.mark_price = mark_price

    async def get_market_context(self, symbol: str, *, persisted_cost_basis: float | None = None) -> MarketContext:
        return MarketContext(
            symbol=symbol.upper(),
            candles=_build_candles(self.mark_price),
            inventory=InventorySnapshot(
                base_balance=1.0 if persisted_cost_basis is not None else 0.0,
                quote_balance=1000.0,
                reserved_quote=0.0,
                mark_price=self.mark_price,
                cost_basis_price=persisted_cost_basis,
            ),
            live_orders=[],
        )


class _NoOpNotifier:
    async def notify_rebuild(self, decision) -> None:
        return None


class Phase6RestartRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_restart_restores_saved_runtime_snapshot_with_diagnostic_fields(self):
        state_store = _MemoryStateStore()
        saved_state = SymbolRuntimeState(
            symbol="SOLUSDT",
            strategy_state=StrategyState(regime=RegimeType.DOWNTREND, last_rebuild_price=88.5),
            cost_basis_price=91.25,
            state_version=2,
            last_cycle_started_at=datetime.now(timezone.utc),
            last_cycle_completed_at=datetime.now(timezone.utc),
            last_successful_execution_at=datetime.now(timezone.utc),
            last_execution_status="executed",
            last_known_base_balance=1.0,
            last_known_quote_balance=950.0,
            last_known_reserved_quote=50.0,
            last_known_mark_price=100.0,
        )
        await state_store.save_symbol_state(saved_state)

        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        cycle = SpotTradingCycleService(
            market_data_provider=_RecoveryMarketDataProvider(),
            executor=_RecoveryExecutor(base_balance=1.0),
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
        self.assertEqual(restored.state_version, 2)
        self.assertEqual(restored.last_execution_status, "executed")
        self.assertEqual(restored.last_known_base_balance, 1.0)
        self.assertEqual(restored.last_known_quote_balance, 1000.0)
        self.assertEqual(restored.last_known_reserved_quote, 0.0)
        self.assertEqual(restored.last_known_mark_price, 100.0)

    async def test_execution_not_confirmed_does_not_overwrite_last_good_persisted_state_before_restart(self):
        state_store = _MemoryStateStore()
        previous_state = SymbolRuntimeState(
            symbol="SOLUSDT",
            strategy_state=StrategyState(regime=RegimeType.RANGE, last_rebuild_price=101.0),
            cost_basis_price=95.0,
            last_execution_status="executed",
        )
        await state_store.save_symbol_state(previous_state)

        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        planner.restore_symbol_runtime(
            SymbolRuntimeState(
                symbol="SOLUSDT",
                strategy_state=StrategyState(regime=RegimeType.DOWNTREND, last_rebuild_price=88.0),
                cost_basis_price=120.0,
                last_execution_status="execution_not_confirmed",
            )
        )
        service = TradingCycleExecutionService(
            executor=SimpleNamespace(sync_orders=AsyncMock(return_value=False)),
            notifier=SimpleNamespace(notify_rebuild=AsyncMock()),
            planner=planner,
            state_store=state_store,
        )

        await service.execute("SOLUSDT", _decision(rebuild_required=True))

        restarted_planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        restarted_cycle = SpotTradingCycleService(
            market_data_provider=_RecoveryMarketDataProvider(),
            executor=_RecoveryExecutor(base_balance=1.0),
            notifier=_NoOpNotifier(),
            planner=restarted_planner,
            state_store=state_store,
        )

        with patch("application.trading_cycle_service.ensure_candle_tables", AsyncMock()):
            await restarted_cycle.initialize(["SOLUSDT"])

        restored = restarted_planner.export_symbol_runtime("SOLUSDT")
        self.assertEqual(restored.strategy_state.regime, RegimeType.RANGE)
        self.assertEqual(restored.strategy_state.last_rebuild_price, 101.0)
        self.assertEqual(restored.cost_basis_price, 95.0)
        self.assertEqual(restored.last_execution_status, "executed")

    async def test_restart_continues_from_runtime_state_saved_after_successful_cycle(self):
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        planner.detector.detect = Mock(return_value=RegimeSnapshot(RegimeType.RANGE, 1.0, ["fixed_range"]))
        state_store = _MemoryStateStore()
        first_cycle = SpotTradingCycleService(
            market_data_provider=_RecoveryMarketDataProvider(),
            executor=_RecoveryExecutor(base_balance=1.0),
            notifier=_NoOpNotifier(),
            planner=planner,
            state_store=state_store,
        )

        with patch("application.trading_cycle_service.ensure_candle_tables", AsyncMock()):
            await first_cycle.initialize(["SOLUSDT"])
        await first_cycle.run("SOLUSDT")

        restarted_planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        restarted_planner.detector.detect = Mock(return_value=RegimeSnapshot(RegimeType.RANGE, 1.0, ["fixed_range"]))
        restarted_cycle = SpotTradingCycleService(
            market_data_provider=_RecoveryMarketDataProvider(),
            executor=_RecoveryExecutor(base_balance=1.0),
            notifier=_NoOpNotifier(),
            planner=restarted_planner,
            state_store=state_store,
        )

        with patch("application.trading_cycle_service.ensure_candle_tables", AsyncMock()):
            await restarted_cycle.initialize(["SOLUSDT"])

        restored = restarted_planner.export_symbol_runtime("SOLUSDT")
        self.assertIsNotNone(restored.last_cycle_started_at)
        self.assertIsNotNone(restored.last_cycle_completed_at)
        self.assertEqual(restored.last_execution_status, "executed")
        self.assertIsNotNone(restored.last_successful_execution_at)
        self.assertEqual(restored.last_known_base_balance, 1.0)
        self.assertEqual(restored.last_known_mark_price, 100.0)

