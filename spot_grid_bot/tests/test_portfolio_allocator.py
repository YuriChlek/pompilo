import unittest
from dataclasses import replace

from domain.models import (
    Candle,
    IndicatorSnapshot,
    InventorySnapshot,
    MarketContext,
    PreliminarySymbolAnalysis,
    RegimeSnapshot,
    RegimeType,
    RiskDecision,
    RiskRuntimeState,
    StrategyState,
)
from domain.portfolio_allocator import PortfolioAllocator
from domain.strategy_config import DEFAULT_STRATEGY_CONFIG


def _context(symbol: str, price: float, quote_balance: float) -> MarketContext:
    candles = [
        Candle(timestamp=index, open=price, high=price + 1.0, low=price - 1.0, close=price, volume=10.0)
        for index in range(260)
    ]
    return MarketContext(
        symbol=symbol,
        candles=candles,
        inventory=InventorySnapshot(base_balance=0.0, quote_balance=quote_balance, reserved_quote=0.0, mark_price=price),
        live_orders=[],
    )


def _analysis(
    symbol: str,
    regime: RegimeType,
    outstanding_buy_notional: float = 0.0,
    *,
    bars_in_state: int = 0,
    atr14: float = 1.0,
) -> PreliminarySymbolAnalysis:
    return PreliminarySymbolAnalysis(
        symbol=symbol,
        indicators=IndicatorSnapshot(
            ema20=100.0,
            ema50=100.0,
            ema200=100.0,
            atr14=atr14,
            realized_volatility=0.01,
            ema50_slope=0.0,
            range_width=4.0,
            price_vs_ema50=0.0,
            directional_move=0.3,
            directional_sign=0.0,
            abnormal_candle=False,
            atr_spike=False,
        ),
        regime_snapshot=RegimeSnapshot(regime, 0.8, []),
        risk=RiskDecision(
            can_trade=True,
            pause_entries=False,
            force_risk_off=False,
            cancel_entries=False,
            allow_exit_only=False,
            outstanding_buy_notional=outstanding_buy_notional,
            projected_inventory_notional=outstanding_buy_notional,
            projected_quote_usage=outstanding_buy_notional,
        ),
        strategy_state=StrategyState(regime=regime, bars_in_state=bars_in_state),
        risk_state=RiskRuntimeState(),
    )


class PortfolioAllocatorTests(unittest.TestCase):
    def test_snapshot_aggregates_portfolio_totals(self):
        allocator = PortfolioAllocator(DEFAULT_STRATEGY_CONFIG)
        contexts = [
            _context("SOLUSDT", 100.0, 1_000.0),
            _context("XRPUSDT", 0.5, 2_000.0),
        ]
        analyses = [
            _analysis("SOLUSDT", RegimeType.RANGE, outstanding_buy_notional=120.0),
            _analysis("XRPUSDT", RegimeType.UPTREND, outstanding_buy_notional=80.0),
        ]

        snapshot = allocator.build_snapshot(contexts, analyses)

        self.assertEqual(len(snapshot.symbols), 2)
        self.assertEqual(snapshot.total_quote_balance, 2_000.0)
        self.assertEqual(snapshot.total_outstanding_buy_notional, 200.0)
        self.assertEqual(snapshot.total_equity, 2_000.0)
        self.assertAlmostEqual(snapshot.symbols[0].atr_pct, 0.01)

    def test_allocator_distributes_budget_to_eligible_symbols_only(self):
        allocator = PortfolioAllocator(DEFAULT_STRATEGY_CONFIG)
        snapshot = allocator.build_snapshot(
            [_context("SOLUSDT", 100.0, 1_000.0), _context("ETHUSDT", 100.0, 1_000.0)],
            [
                _analysis("SOLUSDT", RegimeType.RANGE),
                _analysis("ETHUSDT", RegimeType.DOWNTREND),
            ],
        )

        plan = allocator.allocate(snapshot)
        budgets = {budget.symbol: budget for budget in plan.budgets}

        self.assertGreater(plan.total_allocatable_quote, 0.0)
        self.assertGreater(budgets["SOLUSDT"].portfolio_budget or 0.0, 0.0)
        self.assertTrue(budgets["SOLUSDT"].eligible)
        self.assertEqual(budgets["ETHUSDT"].portfolio_budget, 0.0)
        self.assertFalse(budgets["ETHUSDT"].eligible)

    def test_allocator_respects_max_concurrent_entry_symbols(self):
        from dataclasses import replace

        config = replace(
            DEFAULT_STRATEGY_CONFIG,
            risk=replace(
                DEFAULT_STRATEGY_CONFIG.risk,
                max_concurrent_entry_symbols=2,
            ),
        )
        allocator = PortfolioAllocator(config)
        snapshot = allocator.build_snapshot(
            [
                _context("SOLUSDT", 100.0, 1_000.0),
                _context("ETHUSDT", 100.0, 1_000.0),
                _context("XRPUSDT", 1.0, 1_000.0),
            ],
            [
                _analysis("SOLUSDT", RegimeType.UPTREND),
                _analysis("ETHUSDT", RegimeType.RANGE),
                _analysis("XRPUSDT", RegimeType.RANGE),
            ],
        )

        plan = allocator.allocate(snapshot)
        positive_budgets = [budget for budget in plan.budgets if (budget.portfolio_budget or 0.0) > 0.0]

        self.assertEqual(len(positive_budgets), 2)

    def test_allocator_penalizes_underwater_range_inventory(self):
        allocator = PortfolioAllocator(DEFAULT_STRATEGY_CONFIG)
        profitable_context = _context("SOLUSDT", 100.0, 1_000.0)
        underwater_context = _context("ETHUSDT", 100.0, 1_000.0)
        profitable_context.inventory.base_balance = 1.0
        profitable_context.inventory.cost_basis_price = 95.0
        underwater_context.inventory.base_balance = 1.0
        underwater_context.inventory.cost_basis_price = 120.0

        snapshot = allocator.build_snapshot(
            [profitable_context, underwater_context],
            [
                _analysis("SOLUSDT", RegimeType.RANGE),
                _analysis("ETHUSDT", RegimeType.RANGE),
            ],
        )

        plan = allocator.allocate(snapshot)
        budgets = {budget.symbol: budget for budget in plan.budgets}

        self.assertGreater(budgets["SOLUSDT"].portfolio_budget or 0.0, budgets["ETHUSDT"].portfolio_budget or 0.0)

    def test_allocator_splits_global_recovery_pool_from_new_entry_pool(self):
        config = replace(
            DEFAULT_STRATEGY_CONFIG,
            risk=replace(
                DEFAULT_STRATEGY_CONFIG.risk,
                global_recovery_quota_fraction=0.5,
            ),
        )
        allocator = PortfolioAllocator(config)
        recovery_context = _context("ETHUSDT", 100.0, 1_000.0)
        recovery_context.inventory.base_balance = 1.0
        recovery_context.inventory.cost_basis_price = 120.0
        entry_context = _context("SOLUSDT", 100.0, 1_000.0)
        snapshot = allocator.build_snapshot(
            [recovery_context, entry_context],
            [
                _analysis("ETHUSDT", RegimeType.RANGE, bars_in_state=3),
                _analysis("SOLUSDT", RegimeType.RANGE),
            ],
        )

        plan = allocator.allocate(snapshot)
        budgets = {budget.symbol: budget for budget in plan.budgets}

        self.assertEqual(plan.total_allocatable_quote, 300.0)
        self.assertEqual(plan.total_recovery_allocated_quote, 150.0)
        self.assertEqual(plan.total_entry_allocated_quote, 150.0)
        self.assertEqual(budgets["ETHUSDT"].portfolio_budget, 0.0)
        self.assertGreater(budgets["ETHUSDT"].recovery_budget or 0.0, 0.0)
        self.assertEqual(budgets["SOLUSDT"].recovery_budget, 0.0)
        self.assertEqual(budgets["SOLUSDT"].portfolio_budget, 150.0)

    def test_unused_recovery_quota_returns_to_new_entry_pool(self):
        config = replace(
            DEFAULT_STRATEGY_CONFIG,
            risk=replace(
                DEFAULT_STRATEGY_CONFIG.risk,
                global_recovery_quota_fraction=0.5,
            ),
        )
        allocator = PortfolioAllocator(config)
        fresh_entry_context = _context("SOLUSDT", 100.0, 1_000.0)
        blocked_context = _context("BTCUSDT", 100.0, 1_000.0)
        snapshot = allocator.build_snapshot(
            [fresh_entry_context, blocked_context],
            [
                _analysis("SOLUSDT", RegimeType.RANGE),
                _analysis("BTCUSDT", RegimeType.DOWNTREND),
            ],
        )

        plan = allocator.allocate(snapshot)
        budgets = {budget.symbol: budget for budget in plan.budgets}

        self.assertEqual(plan.total_allocatable_quote, 300.0)
        self.assertEqual(plan.total_recovery_allocated_quote, 0.0)
        self.assertEqual(plan.total_entry_allocated_quote, 300.0)
        self.assertEqual(budgets["SOLUSDT"].portfolio_budget, 300.0)

    def test_allocator_penalizes_more_volatile_symbol_by_atr_pct(self):
        allocator = PortfolioAllocator(DEFAULT_STRATEGY_CONFIG)
        low_vol_context = _context("SOLUSDT", 100.0, 1_000.0)
        high_vol_context = _context("ETHUSDT", 100.0, 1_000.0)
        snapshot = allocator.build_snapshot(
            [low_vol_context, high_vol_context],
            [
                _analysis("SOLUSDT", RegimeType.RANGE, atr14=1.0),
                _analysis("ETHUSDT", RegimeType.RANGE, atr14=4.0),
            ],
        )

        plan = allocator.allocate(snapshot)
        budgets = {budget.symbol: budget for budget in plan.budgets}

        self.assertGreater(budgets["SOLUSDT"].portfolio_budget or 0.0, budgets["ETHUSDT"].portfolio_budget or 0.0)
