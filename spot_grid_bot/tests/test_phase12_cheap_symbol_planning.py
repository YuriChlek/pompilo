import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from domain.indicators import IndicatorSnapshot
from infrastructure.market_data_provider import DatabaseMarketDataProvider
from domain.strategy_config import DEFAULT_STRATEGY_CONFIG
from domain.models import Candle, InventorySnapshot, MarketContext, OrderSide, PreliminarySymbolAnalysis, RegimeSnapshot, RegimeType, RiskDecision, RiskRuntimeState, StrategyState, TargetOrder, VenueConstraints
from domain.grid_builder import GridBuilder
from domain.grid_viability import apply_venue_viability
from domain.inventory_manager import InventoryManager
from domain.range_entry_policy import evaluate_range_entry_profile
from domain.spot_grid_planner import SpotGridPlanner


class _FakeExchange:
    def get_balances(self, symbol: str, *, persisted_cost_basis: float | None = None):
        return InventorySnapshot(base_balance=0.0, quote_balance=1_000.0, reserved_quote=0.0, mark_price=0.08173)

    def get_open_orders(self, symbol: str):
        return []

    def get_instrument_filters(self, symbol: str):
        return SimpleNamespace(
            tick_size=0.00001,
            qty_step=1.0,
            min_order_qty=1.0,
            min_order_amt=5.0,
        )


class Phase12CheapSymbolPlanningTests(unittest.IsolatedAsyncioTestCase):
    async def test_market_context_contains_symbol_specific_venue_constraints(self):
        provider = DatabaseMarketDataProvider(DEFAULT_STRATEGY_CONFIG, _FakeExchange())
        fake_candles = [SimpleNamespace(close=0.08173, timestamp=1, open=0.081, high=0.082, low=0.08, volume=1000.0)] * 260
        fake_higher_timeframe_candles = [
            SimpleNamespace(close=0.08173, timestamp=1, open=0.081, high=0.082, low=0.08, volume=4000.0)
        ] * 200

        with patch.object(provider.repository, "fetch_recent_candles", AsyncMock(return_value=fake_candles)) as fetch_1h_mock, patch.object(
            provider.higher_timeframe_repository,
            "fetch_recent_candles",
            AsyncMock(return_value=fake_higher_timeframe_candles),
        ) as fetch_4h_mock:
            context = await provider.get_market_context("ENAUSDT", persisted_cost_basis=None)

        self.assertIsNotNone(context.venue_constraints)
        self.assertEqual(context.venue_constraints.tick_size, 0.00001)
        self.assertEqual(context.venue_constraints.min_order_amt, 5.0)
        self.assertEqual(context.higher_timeframe_candles, fake_higher_timeframe_candles)
        fetch_1h_mock.assert_awaited_once_with(
            "ENAUSDT",
            DEFAULT_STRATEGY_CONFIG.market_data.lookback_for(DEFAULT_STRATEGY_CONFIG.market_data.candle_interval),
        )
        fetch_4h_mock.assert_awaited_once_with(
            "ENAUSDT",
            DEFAULT_STRATEGY_CONFIG.market_data.lookback_for(DEFAULT_STRATEGY_CONFIG.market_data.higher_timeframe_interval),
        )

    def test_grid_builder_no_longer_forces_global_point_one_tick_on_cheap_symbols(self):
        builder = GridBuilder(DEFAULT_STRATEGY_CONFIG)
        indicators = IndicatorSnapshot(
            ema20=0.081,
            ema50=0.081,
            ema200=0.081,
            atr14=0.004,
            realized_volatility=0.01,
            ema50_slope=0.0,
            range_width=0.02,
            price_vs_ema50=0.0,
            directional_move=0.0,
            directional_sign=0.0,
            abnormal_candle=False,
            atr_spike=False,
        )

        grid = builder.build_range_grid(0.08173, indicators, tick_size=0.00001)
        buy_prices = [level.price for level in grid.levels if level.side == OrderSide.BUY]

        self.assertTrue(buy_prices)
        self.assertTrue(all(price < 0.1 for price in buy_prices))
        self.assertTrue(all(round(price / 0.00001) * 0.00001 == price for price in buy_prices))

    def test_venue_viability_merges_duplicate_levels_after_tick_normalization(self):
        target_orders = [
            TargetOrder(
                client_order_id="buy-0",
                symbol="ENAUSDT",
                side=OrderSide.BUY,
                price=0.08148481,
                size=10.0,
                tag="range_buy",
            ),
            TargetOrder(
                client_order_id="buy-1",
                symbol="ENAUSDT",
                side=OrderSide.BUY,
                price=0.08148482,
                size=12.0,
                tag="trend_pullback_buy",
            ),
            TargetOrder(client_order_id="sell-0", symbol="ENAUSDT", side=OrderSide.SELL, price=0.08411111, size=5.0, reduce_only=True),
        ]

        viable_orders = apply_venue_viability(
            target_orders,
            VenueConstraints(tick_size=0.00001, qty_step=1.0, min_order_qty=1.0, min_order_amt=5.0),
        )

        self.assertEqual(len(viable_orders), 2)
        buy_orders = [order for order in viable_orders if order.side == OrderSide.BUY]
        self.assertEqual(len(buy_orders), 1)
        self.assertEqual(buy_orders[0].price, 0.08148)
        self.assertEqual(buy_orders[0].size, 22.0)
        self.assertEqual(buy_orders[0].tag, "range_buy+trend_pullback_buy")

    def test_sell_levels_are_reduced_when_inventory_cannot_support_exchange_minima(self):
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        planner.detector.detect = lambda *_args, **_kwargs: RegimeSnapshot(RegimeType.RANGE, 1.0, ["fixed_range"])
        candles = [
            Candle(timestamp=index, open=0.872, high=0.878, low=0.868, close=0.873, volume=1000.0)
            for index in range(260)
        ]
        inventory = InventorySnapshot(
            base_balance=3.0,
            quote_balance=1_000.0,
            reserved_quote=0.0,
            mark_price=0.873,
            cost_basis_price=0.80,
        )

        decision = planner.plan(
            MarketContext(
                symbol="SUIUSDT",
                candles=candles,
                inventory=inventory,
                live_orders=[],
                venue_constraints=VenueConstraints(
                    tick_size=0.0001,
                    qty_step=0.1,
                    min_order_qty=1.0,
                    min_order_amt=1.0,
                ),
            )
        )

        sell_orders = [order for order in decision.target_orders if order.side == OrderSide.SELL]
        self.assertLessEqual(len(sell_orders), 2)
        self.assertTrue(all(order.size >= 1.0 for order in sell_orders))

    def test_uptrend_grid_uses_atr_pullback_levels_instead_of_uniform_mini_grid(self):
        builder = GridBuilder(DEFAULT_STRATEGY_CONFIG)
        indicators = IndicatorSnapshot(
            ema20=100.0,
            ema50=99.0,
            ema200=95.0,
            atr14=4.0,
            realized_volatility=0.01,
            ema50_slope=0.01,
            range_width=0.0,
            price_vs_ema50=0.0,
            directional_move=0.0,
            directional_sign=0.0,
            abnormal_candle=False,
            atr_spike=False,
        )

        grid = builder.build_trend_pullback_grid(100.0, indicators, tick_size=0.1)
        buy_prices = [level.price for level in grid.levels if level.side == OrderSide.BUY]
        sell_prices = [level.price for level in grid.levels if level.side == OrderSide.SELL]

        self.assertEqual(buy_prices, [98.0, 96.0, 94.0])
        self.assertEqual(sell_prices, [104.0, 107.0, 110.0])
        self.assertEqual(grid.lower_bound, 94.0)
        self.assertEqual(grid.upper_bound, 110.0)
        self.assertEqual(grid.step, 2.0)

    def test_uptrend_buy_sizing_is_weighted_toward_deeper_pullback_levels(self):
        builder = GridBuilder(DEFAULT_STRATEGY_CONFIG)
        inventory_manager = InventoryManager(DEFAULT_STRATEGY_CONFIG)
        indicators = IndicatorSnapshot(
            ema20=100.0,
            ema50=99.0,
            ema200=95.0,
            atr14=4.0,
            realized_volatility=0.01,
            ema50_slope=0.01,
            range_width=0.0,
            price_vs_ema50=0.0,
            directional_move=0.0,
            directional_sign=0.0,
            abnormal_candle=False,
            atr_spike=False,
        )
        inventory = InventorySnapshot(
            base_balance=0.0,
            quote_balance=10_000.0,
            reserved_quote=0.0,
            mark_price=100.0,
            cost_basis_price=None,
        )
        grid = builder.build_trend_pullback_grid(100.0, indicators, tick_size=0.1)
        sized_grid = inventory_manager.allocate_grid(
            grid,
            inventory,
            symbol_entry_budget=325.0,
            venue_constraints=VenueConstraints(
                tick_size=0.1,
                qty_step=0.0001,
                min_order_qty=0.0001,
                min_order_amt=5.0,
            ),
        )

        buy_levels = [level for level in sized_grid.levels if level.side == OrderSide.BUY]
        buy_levels.sort(key=lambda level: level.price, reverse=True)

        self.assertEqual([level.price for level in buy_levels], [98.0, 96.0, 94.0])
        self.assertLess(buy_levels[0].notional, buy_levels[1].notional)
        self.assertLess(buy_levels[1].notional, buy_levels[2].notional)

    def test_uptrend_sell_sizing_is_weighted_toward_higher_take_profit_levels(self):
        builder = GridBuilder(DEFAULT_STRATEGY_CONFIG)
        inventory_manager = InventoryManager(DEFAULT_STRATEGY_CONFIG)
        indicators = IndicatorSnapshot(
            ema20=100.0,
            ema50=99.0,
            ema200=95.0,
            atr14=4.0,
            realized_volatility=0.01,
            ema50_slope=0.01,
            range_width=0.0,
            price_vs_ema50=0.0,
            directional_move=0.0,
            directional_sign=0.0,
            abnormal_candle=False,
            atr_spike=False,
        )
        inventory = InventorySnapshot(
            base_balance=10.0,
            quote_balance=10_000.0,
            reserved_quote=0.0,
            mark_price=100.0,
            cost_basis_price=95.0,
        )
        grid = builder.build_trend_pullback_grid(100.0, indicators, tick_size=0.1)
        sized_grid = inventory_manager.allocate_grid(
            grid,
            inventory,
            symbol_entry_budget=0.0,
            venue_constraints=VenueConstraints(
                tick_size=0.1,
                qty_step=0.0001,
                min_order_qty=0.0001,
                min_order_amt=1.0,
            ),
        )

        sell_levels = [level for level in sized_grid.levels if level.side == OrderSide.SELL and level.size > 0]
        sell_levels.sort(key=lambda level: level.price)

        self.assertEqual([level.price for level in sell_levels], [104.0, 107.0, 110.0])
        self.assertLess(sell_levels[0].size, sell_levels[1].size)
        self.assertLess(sell_levels[1].size, sell_levels[2].size)
        self.assertLessEqual(sum(level.size for level in sell_levels), inventory.base_balance)

    def test_buy_levels_below_two_usdt_notional_are_not_built(self):
        builder = GridBuilder(DEFAULT_STRATEGY_CONFIG)
        inventory_manager = InventoryManager(DEFAULT_STRATEGY_CONFIG)
        indicators = IndicatorSnapshot(
            ema20=600.0,
            ema50=600.0,
            ema200=600.0,
            atr14=12.0,
            realized_volatility=0.01,
            ema50_slope=0.0,
            range_width=0.0,
            price_vs_ema50=0.0,
            directional_move=0.0,
            directional_sign=0.0,
            abnormal_candle=False,
            atr_spike=False,
        )
        inventory = InventorySnapshot(
            base_balance=0.0,
            quote_balance=100.0,
            reserved_quote=0.0,
            mark_price=600.0,
            cost_basis_price=None,
        )
        grid = builder.build_range_grid(600.0, indicators, tick_size=0.1)
        sized_grid = inventory_manager.allocate_grid(
            grid,
            inventory,
            symbol_entry_budget=3.0,
            venue_constraints=VenueConstraints(
                tick_size=0.1,
                qty_step=0.0001,
                min_order_qty=0.0001,
                min_order_amt=5.0,
            ),
        )

        buy_levels = [level for level in sized_grid.levels if level.side == OrderSide.BUY]
        self.assertTrue(all(level.notional == 0.0 for level in buy_levels))

    def test_expensive_asset_buy_sizing_uses_venue_qty_step_instead_of_global_step(self):
        builder = GridBuilder(DEFAULT_STRATEGY_CONFIG)
        inventory_manager = InventoryManager(DEFAULT_STRATEGY_CONFIG)
        indicators = IndicatorSnapshot(
            ema20=2050.0,
            ema50=2050.0,
            ema200=2050.0,
            atr14=20.0,
            realized_volatility=0.01,
            ema50_slope=0.0,
            range_width=0.0,
            price_vs_ema50=0.0,
            directional_move=0.0,
            directional_sign=0.0,
            abnormal_candle=False,
            atr_spike=False,
        )
        inventory = InventorySnapshot(
            base_balance=0.0,
            quote_balance=1000.0,
            reserved_quote=0.0,
            mark_price=2050.0,
            cost_basis_price=None,
        )
        grid = builder.build_range_grid(2050.0, indicators, tick_size=0.01)
        sized_grid = inventory_manager.allocate_grid(
            grid,
            inventory,
            symbol_entry_budget=15.0,
            venue_constraints=VenueConstraints(
                tick_size=0.01,
                qty_step=0.00001,
                min_order_qty=0.00001,
                min_order_amt=2.0,
            ),
        )

        buy_levels = [level for level in sized_grid.levels if level.side == OrderSide.BUY and level.size > 0]
        self.assertTrue(buy_levels)
        self.assertTrue(any(level.notional >= 2.0 for level in buy_levels))

    def test_buy_level_count_is_reduced_when_budget_cannot_support_full_grid(self):
        builder = GridBuilder(DEFAULT_STRATEGY_CONFIG)
        inventory_manager = InventoryManager(DEFAULT_STRATEGY_CONFIG)
        indicators = IndicatorSnapshot(
            ema20=100.0,
            ema50=100.0,
            ema200=100.0,
            atr14=4.0,
            realized_volatility=0.01,
            ema50_slope=0.0,
            range_width=0.0,
            price_vs_ema50=0.0,
            directional_move=0.0,
            directional_sign=0.0,
            abnormal_candle=False,
            atr_spike=False,
        )
        inventory = InventorySnapshot(
            base_balance=0.0,
            quote_balance=1000.0,
            reserved_quote=0.0,
            mark_price=100.0,
            cost_basis_price=None,
        )
        grid = builder.build_range_grid(100.0, indicators, tick_size=0.1)
        sized_grid = inventory_manager.allocate_grid(
            grid,
            inventory,
            symbol_entry_budget=7.0,
            venue_constraints=VenueConstraints(
                tick_size=0.1,
                qty_step=0.0001,
                min_order_qty=0.0001,
                min_order_amt=2.0,
            ),
        )

        buy_levels = [level for level in sized_grid.levels if level.side == OrderSide.BUY and level.size > 0]
        self.assertEqual(len(buy_levels), 3)


    def test_range_sell_sizing_remains_even(self):
        builder = GridBuilder(DEFAULT_STRATEGY_CONFIG)
        inventory_manager = InventoryManager(DEFAULT_STRATEGY_CONFIG)
        indicators = IndicatorSnapshot(
            ema20=100.0,
            ema50=100.0,
            ema200=100.0,
            atr14=4.0,
            realized_volatility=0.01,
            ema50_slope=0.0,
            range_width=0.0,
            price_vs_ema50=0.0,
            directional_move=0.0,
            directional_sign=0.0,
            abnormal_candle=False,
            atr_spike=False,
        )
        inventory = InventorySnapshot(
            base_balance=10.0,
            quote_balance=10_000.0,
            reserved_quote=0.0,
            mark_price=100.0,
            cost_basis_price=95.0,
        )
        grid = builder.build_range_grid(100.0, indicators, tick_size=0.1)
        sized_grid = inventory_manager.allocate_grid(
            grid,
            inventory,
            symbol_entry_budget=0.0,
            venue_constraints=VenueConstraints(
                tick_size=0.1,
                qty_step=0.0001,
                min_order_qty=0.0001,
                min_order_amt=1.0,
            ),
        )

        sell_levels = [level for level in sized_grid.levels if level.side == OrderSide.SELL and level.size > 0]
        unique_sizes = {round(level.size, 4) for level in sell_levels}

        self.assertEqual(len(unique_sizes), 1)

    def test_overextended_uptrend_blocks_new_buy_entries_but_keeps_sell_side(self):
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        inventory = InventorySnapshot(
            base_balance=2.0,
            quote_balance=10_000.0,
            reserved_quote=0.0,
            mark_price=100.0,
            cost_basis_price=95.0,
        )
        indicators = IndicatorSnapshot(
            ema20=98.0,
            ema50=95.0,
            ema200=90.0,
            atr14=4.0,
            realized_volatility=0.01,
            ema50_slope=0.01,
            range_width=0.0,
            price_vs_ema50=0.0,
            directional_move=0.0,
            directional_sign=0.0,
            abnormal_candle=False,
            atr_spike=False,
        )
        analysis = PreliminarySymbolAnalysis(
            symbol="SOLUSDT",
            indicators=indicators,
            regime_snapshot=RegimeSnapshot(RegimeType.UPTREND, 1.0, ["fixed_uptrend"]),
            risk=RiskDecision(
                can_trade=True,
                pause_entries=False,
                force_risk_off=False,
                cancel_entries=False,
                allow_exit_only=False,
            ),
            strategy_state=StrategyState(regime=RegimeType.UPTREND, bars_in_state=3),
            risk_state=RiskRuntimeState(),
        )
        candles = [
            Candle(timestamp=index, open=100.0, high=101.0, low=99.0, close=100.0, volume=1000.0)
            for index in range(260)
        ]

        decision = planner.plan_from_analysis(
            MarketContext(
                symbol="SOLUSDT",
                candles=candles,
                inventory=inventory,
                live_orders=[],
                venue_constraints=VenueConstraints(
                    tick_size=0.1,
                    qty_step=0.0001,
                    min_order_qty=0.0001,
                    min_order_amt=5.0,
                ),
            ),
            analysis,
        )

        buy_orders = [order for order in decision.target_orders if order.side == OrderSide.BUY]
        sell_orders = [order for order in decision.target_orders if order.side == OrderSide.SELL]

        self.assertFalse(buy_orders)
        self.assertTrue(sell_orders)

    def test_underwater_uptrend_averaging_uses_recovery_budget_and_levels(self):
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        inventory = InventorySnapshot(
            base_balance=2.0,
            quote_balance=10_000.0,
            reserved_quote=0.0,
            mark_price=100.0,
            cost_basis_price=120.0,
        )
        indicators = IndicatorSnapshot(
            ema20=99.0,
            ema50=96.0,
            ema200=90.0,
            atr14=4.0,
            realized_volatility=0.01,
            ema50_slope=0.01,
            range_width=0.0,
            price_vs_ema50=0.0,
            directional_move=0.0,
            directional_sign=0.0,
            abnormal_candle=False,
            atr_spike=False,
        )
        analysis = PreliminarySymbolAnalysis(
            symbol="SOLUSDT",
            indicators=indicators,
            regime_snapshot=RegimeSnapshot(RegimeType.UPTREND, 1.0, ["fixed_uptrend"]),
            risk=RiskDecision(
                can_trade=True,
                pause_entries=False,
                force_risk_off=False,
                cancel_entries=False,
                allow_exit_only=False,
            ),
            strategy_state=StrategyState(regime=RegimeType.UPTREND, bars_in_state=3),
            risk_state=RiskRuntimeState(),
        )
        candles = [
            Candle(timestamp=index, open=100.0, high=101.0, low=99.0, close=100.0, volume=1000.0)
            for index in range(260)
        ]

        decision = planner.plan_from_analysis(
            MarketContext(
                symbol="SOLUSDT",
                candles=candles,
                inventory=inventory,
                live_orders=[],
                venue_constraints=VenueConstraints(
                    tick_size=0.1,
                    qty_step=0.0001,
                    min_order_qty=0.0001,
                    min_order_amt=5.0,
                ),
            ),
            analysis,
            portfolio_budget=300.0,
        )

        buy_orders = [order for order in decision.target_orders if order.side == OrderSide.BUY]

        self.assertLessEqual(len(buy_orders), DEFAULT_STRATEGY_CONFIG.grid.underwater_max_recovery_levels)
        self.assertLess(sum(order.price * order.size for order in buy_orders), 300.0)

    def test_weak_range_entry_quality_profile_reduces_budget_and_levels(self):
        builder = GridBuilder(DEFAULT_STRATEGY_CONFIG)
        indicators = IndicatorSnapshot(
            ema20=105.0,
            ema50=100.0,
            ema200=99.5,
            atr14=4.0,
            realized_volatility=0.01,
            ema50_slope=0.0080,
            range_width=40.0,
            price_vs_ema50=0.06,
            directional_move=4.2,
            directional_sign=1.0,
            abnormal_candle=False,
            atr_spike=False,
        )
        grid = builder.build_range_grid(100.0, indicators, tick_size=0.1)
        profile = evaluate_range_entry_profile(
            price=100.0,
            indicators=indicators,
            grid=grid,
            config=DEFAULT_STRATEGY_CONFIG,
        )
        self.assertLessEqual(profile.quality_score, DEFAULT_STRATEGY_CONFIG.grid.range_entry_quality_soft_threshold)
        self.assertLess(profile.budget_penalty, 1.0)
        self.assertIsNotNone(profile.max_buy_levels)

    def test_range_breakdown_risk_blocks_new_buy_entries(self):
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        inventory = InventorySnapshot(
            base_balance=1.0,
            quote_balance=10_000.0,
            reserved_quote=0.0,
            mark_price=98.0,
            cost_basis_price=105.0,
        )
        indicators = IndicatorSnapshot(
            ema20=99.0,
            ema50=100.0,
            ema200=101.0,
            atr14=4.0,
            realized_volatility=0.01,
            ema50_slope=-0.0012,
            range_width=12.0,
            price_vs_ema50=0.0,
            directional_move=1.5,
            directional_sign=-1.0,
            abnormal_candle=False,
            atr_spike=False,
        )
        analysis = PreliminarySymbolAnalysis(
            symbol="ETHUSDT",
            indicators=indicators,
            regime_snapshot=RegimeSnapshot(RegimeType.RANGE, 1.0, ["fixed_range"]),
            risk=RiskDecision(
                can_trade=True,
                pause_entries=False,
                force_risk_off=False,
                cancel_entries=False,
                allow_exit_only=False,
            ),
            strategy_state=StrategyState(regime=RegimeType.RANGE, bars_in_state=3),
            risk_state=RiskRuntimeState(),
        )
        candles = [
            Candle(timestamp=index, open=98.0, high=100.0, low=97.0, close=98.0, volume=1000.0)
            for index in range(260)
        ]
        decision = planner.plan_from_analysis(
            MarketContext(
                symbol="ETHUSDT",
                candles=candles,
                inventory=inventory,
                live_orders=[],
                venue_constraints=VenueConstraints(
                    tick_size=0.1,
                    qty_step=0.0001,
                    min_order_qty=0.0001,
                    min_order_amt=5.0,
                ),
            ),
            analysis,
            portfolio_budget=500.0,
        )

        self.assertFalse([order for order in decision.target_orders if order.side == OrderSide.BUY])

    def test_strong_uptrend_raises_take_profit_targets(self):
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        inventory = InventorySnapshot(
            base_balance=10.0,
            quote_balance=10_000.0,
            reserved_quote=0.0,
            mark_price=100.0,
            cost_basis_price=95.0,
        )
        indicators = IndicatorSnapshot(
            ema20=102.0,
            ema50=99.0,
            ema200=94.0,
            atr14=4.0,
            realized_volatility=0.01,
            ema50_slope=0.01,
            range_width=0.0,
            price_vs_ema50=0.0,
            directional_move=0.0,
            directional_sign=0.0,
            abnormal_candle=False,
            atr_spike=False,
        )
        analysis = PreliminarySymbolAnalysis(
            symbol="SOLUSDT",
            indicators=indicators,
            regime_snapshot=RegimeSnapshot(RegimeType.UPTREND, 1.0, ["fixed_uptrend"]),
            risk=RiskDecision(
                can_trade=True,
                pause_entries=False,
                force_risk_off=False,
                cancel_entries=False,
                allow_exit_only=False,
            ),
            strategy_state=StrategyState(regime=RegimeType.UPTREND, bars_in_state=3),
            risk_state=RiskRuntimeState(),
        )
        candles = [
            Candle(timestamp=index, open=100.0, high=103.0, low=99.0, close=100.0, volume=1000.0)
            for index in range(260)
        ]

        decision = planner.plan_from_analysis(
            MarketContext(
                symbol="SOLUSDT",
                candles=candles,
                inventory=inventory,
                live_orders=[],
                venue_constraints=VenueConstraints(
                    tick_size=0.1,
                    qty_step=0.0001,
                    min_order_qty=0.0001,
                    min_order_amt=1.0,
                ),
            ),
            analysis,
            portfolio_budget=300.0,
        )

        sell_orders = sorted(
            [order for order in decision.target_orders if order.side == OrderSide.SELL],
            key=lambda order: order.price,
        )
        self.assertEqual([order.price for order in sell_orders], [105.4, 109.1, 112.8])

    def test_deep_underwater_range_blocks_new_buys(self):
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        inventory = InventorySnapshot(
            base_balance=2.0,
            quote_balance=10_000.0,
            reserved_quote=0.0,
            mark_price=70.0,
            cost_basis_price=100.0,
        )
        indicators = IndicatorSnapshot(
            ema20=80.5,
            ema50=80.2,
            ema200=79.8,
            atr14=3.0,
            realized_volatility=0.01,
            ema50_slope=0.0,
            range_width=9.0,
            price_vs_ema50=0.0,
            directional_move=0.8,
            directional_sign=-1.0,
            abnormal_candle=False,
            atr_spike=False,
        )
        analysis = PreliminarySymbolAnalysis(
            symbol="ETHUSDT",
            indicators=indicators,
            regime_snapshot=RegimeSnapshot(RegimeType.RANGE, 1.0, ["fixed_range"]),
            risk=RiskDecision(
                can_trade=True,
                pause_entries=False,
                force_risk_off=False,
                cancel_entries=False,
                allow_exit_only=False,
            ),
            strategy_state=StrategyState(regime=RegimeType.RANGE, bars_in_state=3),
            risk_state=RiskRuntimeState(),
        )
        candles = [
            Candle(timestamp=index, open=80.0, high=81.0, low=79.0, close=80.0, volume=1000.0)
            for index in range(260)
        ]

        decision = planner.plan_from_analysis(
            MarketContext(
                symbol="ETHUSDT",
                candles=candles,
                inventory=inventory,
                live_orders=[],
                venue_constraints=VenueConstraints(
                    tick_size=0.1,
                    qty_step=0.0001,
                    min_order_qty=0.0001,
                    min_order_amt=5.0,
                ),
            ),
            analysis,
            portfolio_budget=300.0,
        )

        self.assertFalse([order for order in decision.target_orders if order.side == OrderSide.BUY])

    def test_underwater_range_averaging_activates_after_trigger(self):
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        inventory = InventorySnapshot(
            base_balance=2.0,
            quote_balance=10_000.0,
            reserved_quote=0.0,
            mark_price=88.0,
            cost_basis_price=100.0,
        )
        indicators = IndicatorSnapshot(
            ema20=88.5,
            ema50=88.0,
            ema200=86.0,
            atr14=3.0,
            realized_volatility=0.01,
            ema50_slope=0.0,
            range_width=9.0,
            price_vs_ema50=0.0,
            directional_move=0.8,
            directional_sign=-1.0,
            abnormal_candle=False,
            atr_spike=False,
        )
        analysis = PreliminarySymbolAnalysis(
            symbol="ETHUSDT",
            indicators=indicators,
            regime_snapshot=RegimeSnapshot(RegimeType.RANGE, 1.0, ["fixed_range"]),
            risk=RiskDecision(
                can_trade=True,
                pause_entries=False,
                force_risk_off=False,
                cancel_entries=False,
                allow_exit_only=False,
            ),
            strategy_state=StrategyState(regime=RegimeType.RANGE, bars_in_state=3),
            risk_state=RiskRuntimeState(),
        )
        candles = [
            Candle(timestamp=index, open=88.0, high=89.0, low=87.0, close=88.0, volume=1000.0)
            for index in range(260)
        ]

        decision = planner.plan_from_analysis(
            MarketContext(
                symbol="ETHUSDT",
                candles=candles,
                inventory=inventory,
                live_orders=[],
                venue_constraints=VenueConstraints(
                    tick_size=0.1,
                    qty_step=0.0001,
                    min_order_qty=0.0001,
                    min_order_amt=5.0,
                ),
            ),
            analysis,
            portfolio_budget=300.0,
        )

        buy_orders = [order for order in decision.target_orders if order.side == OrderSide.BUY]
        self.assertTrue(buy_orders)
        self.assertLessEqual(len(buy_orders), DEFAULT_STRATEGY_CONFIG.grid.underwater_max_recovery_levels)

    def test_underwater_downtrend_does_not_average(self):
        planner = SpotGridPlanner(DEFAULT_STRATEGY_CONFIG)
        inventory = InventorySnapshot(
            base_balance=2.0,
            quote_balance=10_000.0,
            reserved_quote=0.0,
            mark_price=88.0,
            cost_basis_price=100.0,
        )
        indicators = IndicatorSnapshot(
            ema20=87.0,
            ema50=90.0,
            ema200=95.0,
            atr14=3.0,
            realized_volatility=0.01,
            ema50_slope=-0.01,
            range_width=9.0,
            price_vs_ema50=0.0,
            directional_move=1.2,
            directional_sign=-1.0,
            abnormal_candle=False,
            atr_spike=False,
        )
        analysis = PreliminarySymbolAnalysis(
            symbol="ETHUSDT",
            indicators=indicators,
            regime_snapshot=RegimeSnapshot(RegimeType.DOWNTREND, 1.0, ["fixed_downtrend"]),
            risk=RiskDecision(
                can_trade=True,
                pause_entries=False,
                force_risk_off=False,
                cancel_entries=False,
                allow_exit_only=False,
            ),
            strategy_state=StrategyState(regime=RegimeType.DOWNTREND, bars_in_state=3),
            risk_state=RiskRuntimeState(),
        )
        candles = [
            Candle(timestamp=index, open=88.0, high=89.0, low=87.0, close=88.0, volume=1000.0)
            for index in range(260)
        ]

        decision = planner.plan_from_analysis(
            MarketContext(
                symbol="ETHUSDT",
                candles=candles,
                inventory=inventory,
                live_orders=[],
                venue_constraints=VenueConstraints(
                    tick_size=0.1,
                    qty_step=0.0001,
                    min_order_qty=0.0001,
                    min_order_amt=5.0,
                ),
            ),
            analysis,
            portfolio_budget=300.0,
        )

        self.assertFalse([order for order in decision.target_orders if order.side == OrderSide.BUY])
