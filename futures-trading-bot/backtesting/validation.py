from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Optional

from trading.domain.strategy_config import StrategyConfig

from .data_loader import BacktestDataLoader
from .models import (
    BacktestConfig,
    BacktestValidationReport,
    BacktestValidationVariant,
    BacktestValidationVariantResult,
)
from .reporting import write_result_json
from .runner import BacktestRunner


DECIMAL_ZERO = Decimal("0")


def _build_baseline_strategy_config(strategy_config: StrategyConfig) -> StrategyConfig:
    """Return a stricter close-only profile that mirrors the pre-v2 entry behavior."""
    breakout_config = strategy_config.breakout
    return replace(
        strategy_config,
        breakout=replace(
            breakout_config,
            strong_trend_volume_ratio=breakout_config.min_volume_spike_ratio,
            medium_trend_volume_ratio=breakout_config.min_volume_spike_ratio,
            reclaim_enabled=False,
            atr_breakout_buffer_fraction=DECIMAL_ZERO,
            allow_h1_range_in_strong_h4_trend=False,
            high_vol_risk_multiplier=Decimal("1.0"),
        ),
    )


def build_v2_entry_validation_variants(base_config: BacktestConfig) -> list[BacktestValidationVariant]:
    """Build the three replay variants used for v2-entry validation."""
    baseline_config = replace(
        base_config,
        strategy_config=_build_baseline_strategy_config(base_config.strategy_config),
    )
    upgraded_close_config = replace(
        base_config,
        strategy_config=replace(
            base_config.strategy_config,
            breakout=replace(base_config.strategy_config.breakout, reclaim_enabled=False),
        ),
    )
    full_v2_config = replace(base_config)

    return [
        BacktestValidationVariant(
            name="baseline",
            description="Strict breakout_close only with legacy-style confirmation thresholds.",
            config=baseline_config,
        ),
        BacktestValidationVariant(
            name="breakout_close_upgraded",
            description="Adaptive breakout_close without reclaim entries.",
            config=upgraded_close_config,
        ),
        BacktestValidationVariant(
            name="breakout_close_plus_reclaim",
            description="Full v2 entry model with adaptive breakout_close and breakout_reclaim.",
            config=full_v2_config,
        ),
    ]


def _max_cluster_drawdown(max_drawdown_by_cluster: dict[str, Decimal]) -> Decimal:
    """Return the worst cluster drawdown recorded for one backtest result."""
    if not max_drawdown_by_cluster:
        return DECIMAL_ZERO
    return max(max_drawdown_by_cluster.values())


def _build_validation_summary(variants: list[BacktestValidationVariantResult]) -> dict[str, object]:
    """Build a concise comparison summary across validation variants."""
    if not variants:
        return {}

    baseline = variants[0].result
    full_v2 = variants[-1].result
    baseline_trade_count = baseline.stats.total_trades
    full_v2_trade_count = full_v2.stats.total_trades
    baseline_drawdown = _max_cluster_drawdown(baseline.max_drawdown_by_cluster)
    full_v2_drawdown = _max_cluster_drawdown(full_v2.max_drawdown_by_cluster)
    reclaim_stats = full_v2.expectancy_by_setup.get("breakout_reclaim", {})
    reclaim_trade_count = int(reclaim_stats.get("trade_count", 0))
    trade_count_delta = full_v2_trade_count - baseline_trade_count
    net_pnl_delta = full_v2.stats.net_pnl_pct - baseline.stats.net_pnl_pct
    drawdown_delta = full_v2_drawdown - baseline_drawdown

    return {
        "baseline_variant": variants[0].name,
        "full_variant": variants[-1].name,
        "trade_count_delta_vs_baseline": trade_count_delta,
        "net_pnl_delta_vs_baseline": net_pnl_delta,
        "max_cluster_drawdown_delta_vs_baseline": drawdown_delta,
        "reclaim_trade_count": reclaim_trade_count,
        "acceptance": {
            "trade_count_improved": trade_count_delta > 0,
            "drawdown_deterioration_limited": drawdown_delta <= Decimal("5.00"),
            "reclaim_contributed": reclaim_trade_count > 0,
        },
    }


async def run_v2_entry_validation(
    base_config: BacktestConfig,
    data_loader: Optional[BacktestDataLoader] = None,
) -> BacktestValidationReport:
    """Run the v2-entry comparison profile on the existing backtest engine."""
    variants = build_v2_entry_validation_variants(base_config)
    results: list[BacktestValidationVariantResult] = []
    shared_loader = data_loader or BacktestDataLoader()

    for variant in variants:
        runner = BacktestRunner(variant.config, data_loader=shared_loader)
        result = await runner.run()
        results.append(
            BacktestValidationVariantResult(
                name=variant.name,
                description=variant.description,
                result=result,
            )
        )

    return BacktestValidationReport(
        profile="v2_entry",
        variants=results,
        summary=_build_validation_summary(results),
    )


def render_validation_report(report: BacktestValidationReport) -> str:
    """Render a concise CLI summary for a multi-variant validation report."""
    lines = [f"Validation Profile: {report.profile}"]
    for variant in report.variants:
        stats = variant.result.stats
        lines.append(
            (
                f"  {variant.name}: trades={stats.total_trades}, "
                f"net_pnl_pct={stats.net_pnl_pct}%, "
                f"max_cluster_drawdown={_max_cluster_drawdown(variant.result.max_drawdown_by_cluster)}%"
            )
        )
    if report.summary:
        acceptance = report.summary.get("acceptance", {})
        lines.append("Summary")
        lines.append(f"  trade_count_delta_vs_baseline: {report.summary['trade_count_delta_vs_baseline']}")
        lines.append(f"  net_pnl_delta_vs_baseline: {report.summary['net_pnl_delta_vs_baseline']}%")
        lines.append(
            "  max_cluster_drawdown_delta_vs_baseline: "
            f"{report.summary['max_cluster_drawdown_delta_vs_baseline']}%"
        )
        lines.append(f"  reclaim_trade_count: {report.summary['reclaim_trade_count']}")
        lines.append(f"  trade_count_improved: {acceptance.get('trade_count_improved', False)}")
        lines.append(
            "  drawdown_deterioration_limited: "
            f"{acceptance.get('drawdown_deterioration_limited', False)}"
        )
        lines.append(f"  reclaim_contributed: {acceptance.get('reclaim_contributed', False)}")
    return "\n".join(lines)


def save_validation_report(path: str, report: BacktestValidationReport) -> None:
    """Write the validation report to a JSON file."""
    write_result_json(path, report.to_dict())
