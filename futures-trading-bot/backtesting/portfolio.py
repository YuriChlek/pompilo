from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List

from .models import PortfolioBacktestResult, SymbolBacktestResult, Trade
from .reporting import build_cluster_drawdown, build_regime_stats, build_setup_expectancy, build_stats, build_strategy_stats


def combine_symbol_results(results: Iterable[SymbolBacktestResult]) -> PortfolioBacktestResult:
    """Combine per-symbol results into one portfolio-level backtest report."""
    materialized = list(results)
    all_trades: List[Trade] = []
    symbols: List[str] = []
    signal_counts_by_strategy = defaultdict(int)
    filled_order_counts_by_strategy = defaultdict(int)
    skipped_signal_counts = defaultdict(int)
    skipped_signal_counts_by_strategy = defaultdict(lambda: defaultdict(int))
    exit_reason_counts = defaultdict(int)

    for result in materialized:
        symbols.append(result.symbol)
        all_trades.extend(result.trades)
        for strategy_mode, count in result.signal_counts_by_strategy.items():
            signal_counts_by_strategy[strategy_mode] += count
        for strategy_mode, count in result.filled_order_counts_by_strategy.items():
            filled_order_counts_by_strategy[strategy_mode] += count
        for reason, count in result.skipped_signal_counts.items():
            skipped_signal_counts[reason] += count
        for strategy_mode, reason_counts in result.skipped_signal_counts_by_strategy.items():
            for reason, count in reason_counts.items():
                skipped_signal_counts_by_strategy[strategy_mode][reason] += count
        for exit_reason, count in result.exit_reason_counts.items():
            exit_reason_counts[exit_reason] += count

    all_trades.sort(key=lambda trade: (str(trade.exit_time), trade.symbol))
    return PortfolioBacktestResult(
        symbols=symbols,
        symbol_results=materialized,
        trades=all_trades,
        stats=build_stats(all_trades),
        strategy_stats=build_strategy_stats(all_trades),
        signal_counts_by_strategy=dict(signal_counts_by_strategy),
        filled_order_counts_by_strategy=dict(filled_order_counts_by_strategy),
        skipped_signal_counts=dict(skipped_signal_counts),
        skipped_signal_counts_by_strategy={
            strategy_mode: dict(reason_counts)
            for strategy_mode, reason_counts in skipped_signal_counts_by_strategy.items()
        },
        exit_reason_counts=dict(exit_reason_counts),
        pnl_by_regime=build_regime_stats(all_trades),
        expectancy_by_setup=build_setup_expectancy(all_trades),
        max_drawdown_by_cluster=build_cluster_drawdown(all_trades),
    )
