from __future__ import annotations

from collections import defaultdict
import sys
from typing import Dict, Optional

from .adapters import StrategyBacktestAdapter
from .execution import ExecutionSimulator
from .models import BacktestConfig, SymbolBacktestResult
from .reporting import build_cluster_drawdown, build_regime_stats, build_setup_expectancy, build_stats, build_strategy_stats


def _print_replay_progress(symbol: str, current: int, total: int) -> None:
    """Render an in-place progress line for the current symbol replay."""
    if total <= 0:
        return

    progress_ratio = min(1.0, max(0.0, current / total))
    filled = int(progress_ratio * 20)
    bar = "#" * filled + "-" * (20 - filled)
    percent = progress_ratio * 100
    sys.stdout.write(f"\r[{bar}] {percent:5.1f}%  {symbol}  {current}/{total} bars")
    sys.stdout.flush()


async def replay_symbol(symbol: str, candles_df, config: BacktestConfig) -> SymbolBacktestResult:
    """Replay historical candles for a symbol and return the simulated trade result."""
    adapter = StrategyBacktestAdapter(config)
    execution = ExecutionSimulator(
        allow_reversal=config.allow_reversal,
        intrabar_exit_priority=config.intrabar_exit_priority,
    )

    records = candles_df.to_dict("records")
    if not records:
        return SymbolBacktestResult(symbol=symbol, trades=[], stats=build_stats([]))

    signal_counts_by_strategy: Dict[str, int] = defaultdict(int)
    total_records = len(records)
    progress_step = max(1, total_records // 100)
    print(f"▶️ Backtest replay started for {symbol}: {total_records} bars")

    for bar_index, candle in enumerate(records):
        execution.process_candle(candle, bar_index)
        if bar_index == 0 or (bar_index + 1) % progress_step == 0 or bar_index == total_records - 1:
            _print_replay_progress(symbol, bar_index + 1, total_records)

        if bar_index == len(records) - 1:
            break

        start_index = max(0, bar_index + 1 - config.lookback_candles)
        visible_df = candles_df.iloc[start_index:bar_index + 1].copy()
        signal, _, _ = await adapter.build_signal(symbol, visible_df)
        if signal:
            strategy_mode = str(signal.get("strategy_mode", "unknown"))
            signal_counts_by_strategy[strategy_mode] += 1
        execution.queue_signal(signal, bar_index)

    execution.finalize(records[-1], len(records) - 1)
    print()
    exit_reason_counts: Dict[str, int] = defaultdict(int)
    for trade in execution.trades:
        exit_reason_counts[str(trade.exit_reason)] += 1
    return SymbolBacktestResult(
        symbol=symbol,
        trades=execution.trades,
        stats=build_stats(execution.trades),
        strategy_stats=build_strategy_stats(execution.trades),
        signal_counts_by_strategy=dict(signal_counts_by_strategy),
        filled_order_counts_by_strategy=dict(execution.filled_order_counts_by_strategy),
        skipped_signal_counts=dict(execution.skipped_signal_counts),
        skipped_signal_counts_by_strategy={
            strategy_mode: dict(reason_counts)
            for strategy_mode, reason_counts in execution.skipped_signal_counts_by_strategy.items()
        },
        exit_reason_counts=dict(exit_reason_counts),
        pnl_by_regime=build_regime_stats(execution.trades),
        expectancy_by_setup=build_setup_expectancy(execution.trades),
        max_drawdown_by_cluster=build_cluster_drawdown(execution.trades),
    )
