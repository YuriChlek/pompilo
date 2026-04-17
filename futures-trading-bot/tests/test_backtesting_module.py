import unittest
import asyncio
from decimal import Decimal
import types
from unittest.mock import AsyncMock, patch
from tests.support import install_common_test_stubs


install_common_test_stubs()

from backtesting.adapters import StrategyBacktestAdapter
from backtesting.models import BacktestConfig, BacktestStats, PortfolioBacktestResult, SymbolBacktestResult, Trade
from backtesting.validation import build_v2_entry_validation_variants, run_v2_entry_validation
from backtesting.execution import ExecutionSimulator
from backtesting.portfolio import combine_symbol_results
from backtesting.reporting import build_cluster_drawdown, build_regime_stats, build_setup_expectancy, build_stats
from trading.domain.strategy_config import DEFAULT_STRATEGY_CONFIG


class BacktestingExecutionTests(unittest.TestCase):
    def test_market_order_opens_next_bar_and_closes_on_take_profit(self):
        simulator = ExecutionSimulator()
        simulator.queue_signal(
            {
                "symbol": "SOLUSDT",
                "strategy_mode": "trend_breakout",
                "direction": "Buy",
                "order_type": "Market",
                "price": Decimal("100"),
                "stop_loss": Decimal("95"),
                "take_profit": Decimal("110"),
                "time": "2026-03-20 10:00:00",
                "regime": "bull_trend",
                "setup_type": "breakout_close",
                "cluster": "l1_l2_beta",
                "risk_distance": Decimal("5"),
            },
            signal_bar_index=0,
        )

        simulator.process_candle(
            {
                "open_time": "2026-03-20 11:00:00",
                "close_time": "2026-03-20 11:59:59",
                "open": Decimal("101"),
                "high": Decimal("111"),
                "low": Decimal("100"),
                "close": Decimal("109"),
            },
            1,
        )

        self.assertEqual(len(simulator.trades), 1)
        trade = simulator.trades[0]
        self.assertEqual(trade.strategy_mode, "trend_breakout")
        self.assertEqual(trade.entry_price, Decimal("101"))
        self.assertEqual(trade.exit_price, Decimal("110"))
        self.assertEqual(trade.exit_reason, "take_profit")
        self.assertEqual(trade.pnl_pct, Decimal("8.91"))
        self.assertEqual(trade.regime, "bull_trend")
        self.assertEqual(trade.setup_type, "breakout_close")
        self.assertEqual(trade.cluster, "l1_l2_beta")
        self.assertEqual(trade.initial_risk_distance, Decimal("5"))
        self.assertEqual(trade.r_multiple, Decimal("1.80"))

    def test_limit_order_waits_until_price_is_touched(self):
        simulator = ExecutionSimulator()
        simulator.queue_signal(
            {
                "symbol": "SOLUSDT",
                "strategy_mode": "legacy_limit_setup",
                "direction": "Sell",
                "order_type": "Limit",
                "price": Decimal("105"),
                "stop_loss": Decimal("108"),
                "take_profit": Decimal("95"),
                "time": "2026-03-20 10:00:00",
            },
            signal_bar_index=0,
        )

        simulator.process_candle(
            {
                "open_time": "2026-03-20 11:00:00",
                "close_time": "2026-03-20 11:59:59",
                "open": Decimal("100"),
                "high": Decimal("104"),
                "low": Decimal("99"),
                "close": Decimal("103"),
            },
            1,
        )
        self.assertEqual(len(simulator.trades), 0)

        simulator.process_candle(
            {
                "open_time": "2026-03-20 12:00:00",
                "close_time": "2026-03-20 12:59:59",
                "open": Decimal("103"),
                "high": Decimal("106"),
                "low": Decimal("94"),
                "close": Decimal("96"),
            },
            2,
        )

        self.assertEqual(len(simulator.trades), 1)
        trade = simulator.trades[0]
        self.assertEqual(trade.strategy_mode, "legacy_limit_setup")
        self.assertEqual(trade.entry_price, Decimal("105"))
        self.assertEqual(trade.exit_price, Decimal("95"))
        self.assertEqual(trade.exit_reason, "take_profit")

    def test_intrabar_conflict_uses_stop_priority_by_default(self):
        simulator = ExecutionSimulator()
        simulator.queue_signal(
            {
                "symbol": "SOLUSDT",
                "strategy_mode": "trend_breakout",
                "direction": "Buy",
                "order_type": "Market",
                "price": Decimal("100"),
                "stop_loss": Decimal("95"),
                "take_profit": Decimal("110"),
                "time": "2026-03-20 10:00:00",
            },
            signal_bar_index=0,
        )

        simulator.process_candle(
            {
                "open_time": "2026-03-20 11:00:00",
                "close_time": "2026-03-20 11:59:59",
                "open": Decimal("100"),
                "high": Decimal("111"),
                "low": Decimal("94"),
                "close": Decimal("101"),
            },
            1,
        )

        self.assertEqual(simulator.trades[0].exit_reason, "stop_loss")

    def test_queue_signal_tracks_skip_reasons(self):
        simulator = ExecutionSimulator()
        simulator.position = types.SimpleNamespace(direction="buy")

        reason = simulator.queue_signal(
            {
                "symbol": "SOLUSDT",
                "strategy_mode": "legacy_limit_setup",
                "direction": "Buy",
                "order_type": "Market",
                "price": Decimal("100"),
                "stop_loss": Decimal("95"),
                "take_profit": Decimal("110"),
                "time": "2026-03-20 10:00:00",
            },
            signal_bar_index=0,
        )

        self.assertEqual(reason, "same_direction_position")
        self.assertEqual(simulator.skipped_signal_counts["same_direction_position"], 1)
        self.assertEqual(
            simulator.skipped_signal_counts_by_strategy["legacy_limit_setup"]["same_direction_position"],
            1,
        )

    def test_trend_breakout_signal_replaces_pending_limit_order(self):
        simulator = ExecutionSimulator()
        simulator.queue_signal(
            {
                "symbol": "SOLUSDT",
                "strategy_mode": "legacy_limit_setup",
                "direction": "Buy",
                "order_type": "Limit",
                "price": Decimal("99"),
                "stop_loss": Decimal("95"),
                "take_profit": Decimal("110"),
                "time": "2026-03-20 10:00:00",
            },
            signal_bar_index=0,
        )

        reason = simulator.queue_signal(
            {
                "symbol": "SOLUSDT",
                "strategy_mode": "trend_breakout",
                "direction": "Buy",
                "order_type": "Market",
                "price": Decimal("100"),
                "stop_loss": Decimal("96"),
                "take_profit": Decimal("108"),
                "time": "2026-03-20 11:00:00",
            },
            signal_bar_index=1,
        )

        self.assertEqual(reason, "queued")
        self.assertIsNotNone(simulator.pending_order)
        self.assertEqual(simulator.pending_order.order_type, "market")
        self.assertEqual(simulator.pending_order.signal_payload["strategy_mode"], "trend_breakout")


class BacktestingReportingTests(unittest.TestCase):
    @staticmethod
    def _trade(
        symbol: str,
        strategy_mode: str,
        pnl_pct: str,
        *,
        regime: str = "bull_trend",
        setup_type: str = "breakout_close",
        cluster: str = "l1_l2_beta",
        exit_time: str = "t2",
        r_multiple: str | None = None,
        bars_held: int = 2,
    ) -> Trade:
        return Trade(
            symbol,
            strategy_mode,
            "buy",
            "t1",
            exit_time,
            Decimal("100"),
            Decimal("103"),
            Decimal("95"),
            Decimal("110"),
            "tp",
            Decimal(pnl_pct),
            bars_held,
            "market",
            regime=regime,
            setup_type=setup_type,
            cluster=cluster,
            initial_risk_distance=Decimal("5"),
            r_multiple=Decimal(r_multiple) if r_multiple is not None else None,
        )

    def test_build_stats_calculates_win_loss_and_streaks(self):
        trades = [
            self._trade("SOLUSDT", "trend_breakout", "3.00", exit_time="t2", bars_held=2),
            self._trade("SOLUSDT", "trend_breakout", "2.00", exit_time="t4", bars_held=1),
            self._trade("SOLUSDT", "legacy_limit_setup", "-3.00", setup_type="legacy_limit", exit_time="t6", bars_held=2),
            self._trade("SOLUSDT", "legacy_limit_setup", "-4.00", setup_type="legacy_limit", exit_time="t8", bars_held=1),
            self._trade("SOLUSDT", "trend_breakout", "4.00", exit_time="t10", bars_held=1),
        ]

        stats = build_stats(trades)

        self.assertEqual(stats.total_trades, 5)
        self.assertEqual(stats.profitable_trades, 3)
        self.assertEqual(stats.losing_trades, 2)
        self.assertEqual(stats.profitable_trades_pct, Decimal("60.00"))
        self.assertEqual(stats.losing_trades_pct, Decimal("40.00"))
        self.assertEqual(stats.max_profit_streak, 2)
        self.assertEqual(stats.max_loss_streak, 2)
        self.assertEqual(stats.average_bars_held, Decimal("1.40"))

    def test_build_strategy_stats_groups_trades_by_strategy_mode(self):
        from backtesting.reporting import build_strategy_stats

        trades = [
            self._trade("SOLUSDT", "trend_breakout", "3.00", exit_time="t2"),
            self._trade("SOLUSDT", "trend_breakout", "-3.00", exit_time="t4"),
            self._trade("SOLUSDT", "legacy_limit_setup", "4.00", setup_type="legacy_limit", exit_time="t6"),
        ]

        strategy_stats = build_strategy_stats(trades)

        self.assertEqual(strategy_stats["trend_breakout"].total_trades, 2)
        self.assertEqual(strategy_stats["trend_breakout"].net_pnl_pct, Decimal("0.00"))
        self.assertEqual(strategy_stats["legacy_limit_setup"].total_trades, 1)
        self.assertEqual(strategy_stats["legacy_limit_setup"].profitable_trades, 1)

    def test_build_regime_stats_groups_pnl_by_regime(self):
        trades = [
            self._trade("SOLUSDT", "trend_breakout", "3.00", regime="bull_trend"),
            self._trade("ETHUSDT", "trend_breakout", "-1.50", regime="bull_trend", exit_time="t3"),
            self._trade("XRPUSDT", "trend_breakout", "2.25", regime="bear_trend", exit_time="t4"),
        ]

        result = build_regime_stats(trades)

        self.assertEqual(result["bull_trend"], Decimal("1.50"))
        self.assertEqual(result["bear_trend"], Decimal("2.25"))

    def test_build_setup_expectancy_groups_expectancy_and_win_rate(self):
        trades = [
            self._trade("SOLUSDT", "trend_breakout", "3.00", setup_type="breakout_close", r_multiple="0.60"),
            self._trade("ETHUSDT", "trend_breakout", "-1.00", setup_type="breakout_close", exit_time="t3", r_multiple="-0.20"),
            self._trade("XRPUSDT", "trend_breakout", "4.00", setup_type="pullback_reclaim", exit_time="t4", r_multiple="0.80"),
        ]

        result = build_setup_expectancy(trades)

        self.assertEqual(result["breakout_close"]["trade_count"], 2)
        self.assertEqual(result["breakout_close"]["win_rate_pct"], Decimal("50.00"))
        self.assertEqual(result["breakout_close"]["expectancy_pct"], Decimal("1.00"))
        self.assertEqual(result["breakout_close"]["expectancy_r"], Decimal("0.20"))
        self.assertEqual(result["pullback_reclaim"]["trade_count"], 1)

    def test_build_cluster_drawdown_computes_peak_to_trough_by_cluster(self):
        trades = [
            self._trade("SOLUSDT", "trend_breakout", "5.00", cluster="l1_l2_beta", exit_time="t1"),
            self._trade("AVAXUSDT", "trend_breakout", "-3.00", cluster="l1_l2_beta", exit_time="t2"),
            self._trade("ETHUSDT", "trend_breakout", "-4.00", cluster="majors", exit_time="t3"),
        ]

        result = build_cluster_drawdown(trades)

        self.assertEqual(result["l1_l2_beta"], Decimal("3.00"))
        self.assertEqual(result["majors"], Decimal("4.00"))

    def test_portfolio_result_aggregates_strategy_signal_and_exit_counts(self):
        trade = self._trade("SOLUSDT", "trend_breakout", "4.00", exit_time="t2", r_multiple="0.80")
        symbol_result = SymbolBacktestResult(
            symbol="SOLUSDT",
            trades=[trade],
            stats=build_stats([trade]),
            strategy_stats={"trend_breakout": build_stats([trade])},
            signal_counts_by_strategy={"trend_breakout": 3, "legacy_limit_setup": 1},
            filled_order_counts_by_strategy={"trend_breakout": 1},
            skipped_signal_counts={"same_direction_position": 2},
            skipped_signal_counts_by_strategy={"trend_breakout": {"same_direction_position": 2}},
            exit_reason_counts={"take_profit": 1},
            pnl_by_regime={"bull_trend": Decimal("4.00")},
            expectancy_by_setup={"breakout_close": {"trade_count": 1, "win_rate_pct": Decimal("100.00"), "expectancy_pct": Decimal("4.00"), "expectancy_r": Decimal("0.80")}},
            max_drawdown_by_cluster={"l1_l2_beta": Decimal("0.00")},
        )

        result = combine_symbol_results([symbol_result])

        self.assertEqual(result.signal_counts_by_strategy["trend_breakout"], 3)
        self.assertEqual(result.signal_counts_by_strategy["legacy_limit_setup"], 1)
        self.assertEqual(result.filled_order_counts_by_strategy["trend_breakout"], 1)
        self.assertEqual(result.skipped_signal_counts["same_direction_position"], 2)
        self.assertEqual(result.skipped_signal_counts_by_strategy["trend_breakout"]["same_direction_position"], 2)
        self.assertEqual(result.exit_reason_counts["take_profit"], 1)
        self.assertEqual(result.pnl_by_regime["bull_trend"], Decimal("4.00"))
        self.assertEqual(result.expectancy_by_setup["breakout_close"]["expectancy_r"], Decimal("0.80"))
        self.assertEqual(result.max_drawdown_by_cluster["l1_l2_beta"], Decimal("0.00"))

    def test_portfolio_result_to_dict_serializes_new_analytics_sections(self):
        trade = self._trade("SOLUSDT", "trend_breakout", "4.00", exit_time="t2", r_multiple="0.80")
        result = combine_symbol_results(
            [
                SymbolBacktestResult(
                    symbol="SOLUSDT",
                    trades=[trade],
                    stats=build_stats([trade]),
                    strategy_stats={"trend_breakout": build_stats([trade])},
                )
            ]
        )

        payload = result.to_dict()

        self.assertEqual(payload["pnl_by_regime"]["bull_trend"], "4.00")
        self.assertEqual(payload["expectancy_by_setup"]["breakout_close"]["expectancy_r"], "0.80")
        self.assertEqual(payload["max_drawdown_by_cluster"]["l1_l2_beta"], "0.00")
        self.assertEqual(payload["trades"][0]["r_multiple"], "0.80")


class BacktestingValidationTests(unittest.TestCase):
    @staticmethod
    def _portfolio_result(
        *,
        trade_count: int,
        net_pnl_pct: str,
        reclaim_trade_count: int = 0,
        max_cluster_drawdown: str = "0.00",
    ) -> PortfolioBacktestResult:
        stats = BacktestStats(
            total_trades=trade_count,
            profitable_trades=max(0, trade_count - 1),
            losing_trades=1 if trade_count else 0,
            breakeven_trades=0,
            profitable_trades_pct=Decimal("50.00") if trade_count else Decimal("0.00"),
            losing_trades_pct=Decimal("50.00") if trade_count else Decimal("0.00"),
            breakeven_trades_pct=Decimal("0.00"),
            max_profit_streak=1,
            max_loss_streak=1,
            average_profit_pct=Decimal("2.00"),
            average_loss_pct=Decimal("-1.00"),
            average_bars_held=Decimal("2.00"),
            net_pnl_pct=Decimal(net_pnl_pct),
        )
        expectancy_by_setup = {
            "breakout_close": {
                "trade_count": max(trade_count - reclaim_trade_count, 0),
                "win_rate_pct": Decimal("50.00"),
                "expectancy_pct": Decimal("1.00"),
                "expectancy_r": Decimal("0.20"),
            }
        }
        if reclaim_trade_count:
            expectancy_by_setup["breakout_reclaim"] = {
                "trade_count": reclaim_trade_count,
                "win_rate_pct": Decimal("50.00"),
                "expectancy_pct": Decimal("0.80"),
                "expectancy_r": Decimal("0.16"),
            }
        return PortfolioBacktestResult(
            symbols=["SOLUSDT"],
            symbol_results=[],
            trades=[],
            stats=stats,
            strategy_stats={},
            signal_counts_by_strategy={"trend_breakout": trade_count},
            filled_order_counts_by_strategy={"trend_breakout": trade_count},
            skipped_signal_counts={},
            skipped_signal_counts_by_strategy={},
            exit_reason_counts={},
            pnl_by_regime={"bull_trend": Decimal(net_pnl_pct)},
            expectancy_by_setup=expectancy_by_setup,
            max_drawdown_by_cluster={"l1_l2_beta": Decimal(max_cluster_drawdown)},
        )

    def test_adapter_passes_backtest_strategy_config_to_signal_generation(self):
        config = BacktestConfig(symbols=["SOLUSDT"], strategy_config=DEFAULT_STRATEGY_CONFIG)
        adapter = StrategyBacktestAdapter(config)
        class _CandlesStub:
            def __len__(self):
                return config.min_candles

            def copy(self):
                return None

        candles_df = _CandlesStub()

        adapter.bot = types.SimpleNamespace(
            _prepare_h1_indicators=lambda data: types.SimpleNamespace(tail=lambda count: "history"),
            _prepare_h4_indicators=lambda data: "h4",
            _prepare_d1_indicators=lambda data: "d1",
            _assemble_trend_result=lambda h1, h4, d1: "trend_result",
        )

        with patch("backtesting.adapters.generate_strategy_signal", new=AsyncMock(return_value={"ok": True})) as signal_mock:
            result = asyncio.run(adapter.build_signal("SOLUSDT", candles_df))

        self.assertEqual(result[0], {"ok": True})
        self.assertEqual(signal_mock.await_args.kwargs["strategy_config"], DEFAULT_STRATEGY_CONFIG)

    def test_build_v2_entry_validation_variants_builds_expected_profiles(self):
        variants = build_v2_entry_validation_variants(BacktestConfig(symbols=["SOLUSDT"]))

        self.assertEqual([variant.name for variant in variants], [
            "baseline",
            "breakout_close_upgraded",
            "breakout_close_plus_reclaim",
        ])
        self.assertFalse(variants[0].config.strategy_config.breakout.reclaim_enabled)
        self.assertFalse(variants[0].config.strategy_config.breakout.allow_h1_range_in_strong_h4_trend)
        self.assertEqual(
            variants[0].config.strategy_config.breakout.strong_trend_volume_ratio,
            variants[0].config.strategy_config.breakout.min_volume_spike_ratio,
        )
        self.assertEqual(
            variants[0].config.strategy_config.breakout.atr_breakout_buffer_fraction,
            Decimal("0"),
        )
        self.assertFalse(variants[1].config.strategy_config.breakout.reclaim_enabled)
        self.assertTrue(variants[2].config.strategy_config.breakout.reclaim_enabled)

    def test_run_v2_entry_validation_builds_comparison_summary(self):
        base_config = BacktestConfig(symbols=["SOLUSDT"])
        fake_results = [
            self._portfolio_result(trade_count=10, net_pnl_pct="5.00", max_cluster_drawdown="3.00"),
            self._portfolio_result(trade_count=12, net_pnl_pct="6.50", max_cluster_drawdown="4.00"),
            self._portfolio_result(
                trade_count=15,
                net_pnl_pct="8.00",
                reclaim_trade_count=4,
                max_cluster_drawdown="4.50",
            ),
        ]

        with patch("backtesting.validation.BacktestRunner.run", new=AsyncMock(side_effect=fake_results)):
            report = asyncio.run(run_v2_entry_validation(base_config, data_loader=object()))

        self.assertEqual(report.profile, "v2_entry")
        self.assertEqual([variant.name for variant in report.variants], [
            "baseline",
            "breakout_close_upgraded",
            "breakout_close_plus_reclaim",
        ])
        self.assertEqual(report.summary["trade_count_delta_vs_baseline"], 5)
        self.assertEqual(report.summary["net_pnl_delta_vs_baseline"], Decimal("3.00"))
        self.assertEqual(report.summary["max_cluster_drawdown_delta_vs_baseline"], Decimal("1.50"))
        self.assertEqual(report.summary["reclaim_trade_count"], 4)
        self.assertTrue(report.summary["acceptance"]["trade_count_improved"])
        self.assertTrue(report.summary["acceptance"]["drawdown_deterioration_limited"])
        self.assertTrue(report.summary["acceptance"]["reclaim_contributed"])
