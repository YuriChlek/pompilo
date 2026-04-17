from __future__ import annotations

from typing import Optional

from .data_loader import BacktestDataLoader
from .models import BacktestConfig, PortfolioBacktestResult
from .portfolio import combine_symbol_results
from .replay import replay_symbol
from .reporting import render_stats, write_result_json


def _render_counter_block(title: str, values: dict[str, int]) -> str:
    """Format a simple key-value counter block for CLI output."""
    lines = [title]
    if not values:
        lines.append("  none")
        return "\n".join(lines)
    for key, value in sorted(values.items()):
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def _render_nested_counter_block(title: str, values: dict[str, dict[str, int]]) -> str:
    """Format a nested strategy -> reason -> count block for CLI output."""
    lines = [title]
    if not values:
        lines.append("  none")
        return "\n".join(lines)
    for strategy_mode, reason_counts in sorted(values.items()):
        lines.append(f"  {strategy_mode}:")
        for reason, value in sorted(reason_counts.items()):
            lines.append(f"    {reason}: {value}")
    return "\n".join(lines)


def _render_decimal_block(title: str, values: dict[str, object], *, suffix: str = "") -> str:
    """Format a one-level analytics block for CLI output."""
    lines = [title]
    if not values:
        lines.append("  none")
        return "\n".join(lines)
    for key, value in sorted(values.items()):
        lines.append(f"  {key}: {value}{suffix}")
    return "\n".join(lines)


def _render_nested_metrics_block(title: str, values: dict[str, dict[str, object]]) -> str:
    """Format a nested analytics block for CLI output."""
    lines = [title]
    if not values:
        lines.append("  none")
        return "\n".join(lines)
    for key, metrics in sorted(values.items()):
        lines.append(f"  {key}:")
        for metric_name, value in sorted(metrics.items()):
            lines.append(f"    {metric_name}: {value}")
    return "\n".join(lines)


class BacktestRunner:
    """Coordinate data loading, replay, and reporting for portfolio backtests."""

    def __init__(self, config: BacktestConfig, data_loader: Optional[BacktestDataLoader] = None):
        """Initialize the backtest orchestrator for one or more symbols."""
        self.config = config
        self.data_loader = data_loader or BacktestDataLoader()

    async def run(self) -> PortfolioBacktestResult:
        """Run the full backtest and return the aggregated portfolio result."""
        symbol_results = []
        total_symbols = len(self.config.symbols)
        print(f"🚀 Backtest started for {total_symbols} symbol(s)")
        for index, symbol in enumerate(self.config.symbols, start=1):
            print(f"\n📥 Loading history for {symbol} ({index}/{total_symbols})")
            candles_df = self.data_loader.load_symbol_history(
                symbol,
                date_from=self.config.date_from,
                date_to=self.config.date_to,
            )
            print(f"📊 Loaded {len(candles_df)} candles for {symbol}")
            symbol_results.append(await replay_symbol(symbol, candles_df, self.config))
        print("\n✅ Backtest replay completed")
        return combine_symbol_results(symbol_results)

    @staticmethod
    def print_report(result: PortfolioBacktestResult) -> None:
        """Print a concise CLI report for the portfolio and each symbol."""
        print(render_stats("Portfolio Summary", result.stats))
        print()
        print(_render_counter_block("Portfolio Signals By Strategy", result.signal_counts_by_strategy))
        print()
        print(_render_counter_block("Portfolio Filled Orders By Strategy", result.filled_order_counts_by_strategy))
        print()
        print(_render_counter_block("Portfolio Skipped Signals", result.skipped_signal_counts))
        print()
        print(_render_nested_counter_block("Portfolio Skipped Signals By Strategy", result.skipped_signal_counts_by_strategy))
        print()
        print(_render_counter_block("Portfolio Exit Reasons", result.exit_reason_counts))
        print()
        print(_render_decimal_block("Portfolio PnL By Regime", result.pnl_by_regime, suffix="%"))
        print()
        print(_render_nested_metrics_block("Portfolio Expectancy By Setup", result.expectancy_by_setup))
        print()
        print(_render_decimal_block("Portfolio Max Drawdown By Cluster", result.max_drawdown_by_cluster, suffix="%"))
        for strategy_mode, strategy_stats in sorted(result.strategy_stats.items()):
            print()
            print(render_stats(f"Portfolio Strategy: {strategy_mode}", strategy_stats))
        for symbol_result in result.symbol_results:
            print()
            print(render_stats(f"Symbol Summary: {symbol_result.symbol}", symbol_result.stats))
            print()
            print(_render_counter_block("  Signals By Strategy", symbol_result.signal_counts_by_strategy))
            print()
            print(_render_counter_block("  Filled Orders By Strategy", symbol_result.filled_order_counts_by_strategy))
            print()
            print(_render_counter_block("  Skipped Signals", symbol_result.skipped_signal_counts))
            print()
            print(_render_nested_counter_block("  Skipped Signals By Strategy", symbol_result.skipped_signal_counts_by_strategy))
            print()
            print(_render_counter_block("  Exit Reasons", symbol_result.exit_reason_counts))
            print()
            print(_render_decimal_block("  PnL By Regime", symbol_result.pnl_by_regime, suffix="%"))
            print()
            print(_render_nested_metrics_block("  Expectancy By Setup", symbol_result.expectancy_by_setup))
            print()
            print(_render_decimal_block("  Max Drawdown By Cluster", symbol_result.max_drawdown_by_cluster, suffix="%"))
            for strategy_mode, strategy_stats in sorted(symbol_result.strategy_stats.items()):
                print()
                print(render_stats(f"  Strategy: {strategy_mode}", strategy_stats))

    @staticmethod
    def save_report(result: PortfolioBacktestResult, output_path: str) -> None:
        """Save the backtest result to a JSON file."""
        write_result_json(output_path, result.to_dict())
