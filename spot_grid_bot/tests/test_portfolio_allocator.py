import unittest

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


def _analysis(symbol: str, regime: RegimeType, outstanding_buy_notional: float = 0.0) -> PreliminarySymbolAnalysis:
    return PreliminarySymbolAnalysis(
        symbol=symbol,
        indicators=IndicatorSnapshot(
            ema20=100.0,
            ema50=100.0,
            ema200=100.0,
            atr14=1.0,
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
        strategy_state=StrategyState(regime=regime),
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
