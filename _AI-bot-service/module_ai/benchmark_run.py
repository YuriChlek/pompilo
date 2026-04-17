from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from module_ai.console import ProgressBar, log_step, stage
from module_ai.data_access import filter_time_range
from module_ai.evaluate import evaluate_forecast_frame, evaluate_forecasts
from module_ai.regime_reporting import build_regime_group_reports
from module_ai.runtime import fit_runtime_bundle, predict_origin_frame
from module_ai.splits import SplitConfig, split_dataset
from module_ai.symbols import resolve_cli_symbols
from module_ai.training_universe import load_training_universe_bundle, summarize_training_universe
from module_ai.data_pipeline import MULTI_SYMBOL_SENTINEL
from module_ai.modeling import MODEL_BACKEND_GRADIENT_BOOSTING, SUPPORTED_MODEL_BACKENDS
from utils.config import TRADING_SYMBOLS


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark multiple model backends on the shared multi-symbol pipeline.")
    parser.add_argument("--symbol", default="ALL", help="Trading symbol for single-symbol benchmark or ALL for the training universe")
    parser.add_argument("--symbols", default="", help="Comma-separated symbol list for benchmark. Overrides --symbol when provided")
    parser.add_argument(
        "--model-backends",
        default="gradient_boosting,catboost",
        help="Comma-separated model backend list to benchmark",
    )
    parser.add_argument("--output-dir", default="./artifacts/benchmarks", help="Directory where benchmark reports will be saved")
    parser.add_argument("--encoder-len", type=int, default=48)
    parser.add_argument("--pred-len", type=int, default=6)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--signal-mode", default="quantile_barrier")
    parser.add_argument("--cost-threshold", type=float, default=0.0)
    parser.add_argument("--buy-threshold", type=float, default=0.0)
    parser.add_argument("--sell-threshold", type=float, default=0.0)
    parser.add_argument("--max-width", type=float, default=0.05)
    parser.add_argument("--execution-rule", default="close_to_close")
    parser.add_argument("--transaction-cost", type=float, default=0.0)
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument("--approval-max-validation-rmse", type=float, default=0.08)
    parser.add_argument("--approval-max-final-test-rmse", type=float, default=0.08)
    parser.add_argument("--approval-min-validation-coverage", type=float, default=0.50)
    parser.add_argument("--approval-min-final-test-coverage", type=float, default=0.50)
    parser.add_argument("--approval-max-per-symbol-validation-rmse", type=float, default=0.12)
    parser.add_argument("--approval-max-per-symbol-final-test-rmse", type=float, default=0.12)
    parser.add_argument("--approval-min-per-symbol-validation-coverage", type=float, default=0.10)
    parser.add_argument("--approval-min-per-symbol-final-test-coverage", type=float, default=0.10)
    parser.add_argument(
        "--compare-per-symbol-baseline",
        action="store_true",
        help="Additionally benchmark one separate model per symbol and compare it against the shared model",
    )
    return parser


def _resolve_symbols(args: argparse.Namespace) -> list[str]:
    return resolve_cli_symbols(
        symbol=getattr(args, "symbol", "ALL"),
        symbols=getattr(args, "symbols", ""),
        default_symbols=TRADING_SYMBOLS,
    )


def _resolve_backends(raw_value: str) -> list[str]:
    backends = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not backends:
        raise ValueError("at least one model backend must be provided")
    invalid = [backend for backend in backends if backend not in SUPPORTED_MODEL_BACKENDS]
    if invalid:
        raise ValueError(f"unsupported model backend(s): {invalid}")
    deduplicated: list[str] = []
    for backend in backends:
        if backend not in deduplicated:
            deduplicated.append(backend)
    return deduplicated


def _build_split_frames(feature_df: pd.DataFrame, split_meta: dict[str, str | None]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_raw = filter_time_range(
        feature_df,
        start_time=split_meta["train_start_time"],
        end_time=split_meta["train_end_time"],
    )
    val_raw = filter_time_range(
        feature_df,
        start_time=split_meta["val_start_time"],
        end_time=split_meta["val_end_time"],
    )
    test_raw = filter_time_range(
        feature_df,
        start_time=split_meta["test_start_time"],
        end_time=split_meta["test_end_time"],
    )
    return train_raw, val_raw, test_raw


def _build_per_symbol_forecast_metrics(forecast_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for symbol, group in forecast_df.groupby("symbol", sort=True):
        metrics[str(symbol)] = evaluate_forecasts(group)
    return metrics


def _passes_per_symbol_gate(
    per_symbol_metrics: dict[str, dict[str, float]],
    *,
    max_rmse: float,
    min_coverage: float,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for symbol, metrics in per_symbol_metrics.items():
        rmse = float(metrics.get("rmse_q50", float("inf")))
        coverage = float(metrics.get("interval_coverage_rate", float("-inf")))
        if rmse > max_rmse or coverage < min_coverage:
            failures.append(symbol)
    return len(failures) == 0, failures


def _benchmark_single_backend(
    *,
    backend: str,
    feature_df: pd.DataFrame,
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    train_origins: pd.DataFrame,
    val_origins: pd.DataFrame,
    test_origins: pd.DataFrame,
    raw_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    runtime_bundle = fit_runtime_bundle(
        feature_df,
        train_raw_df=train_raw,
        train_origin_df=train_origins,
        encoder_len=args.encoder_len,
        model_params={
            "model_backend": backend,
            "random_state": args.seed,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "compute_device": args.compute_device,
            "gpu_device_id": args.gpu_device_id,
        },
    )
    scaled_feature_df = runtime_bundle["scaled_feature_df"]
    model_bundle = runtime_bundle["model_bundle"]

    validation_forecasts, _ = predict_origin_frame(
        scaled_feature_df,
        origin_df=val_origins,
        encoder_len=args.encoder_len,
        model_bundle=model_bundle,
    )
    final_test_forecasts, _ = predict_origin_frame(
        scaled_feature_df,
        origin_df=test_origins,
        encoder_len=args.encoder_len,
        model_bundle=model_bundle,
    )

    signal_kwargs = {
        "cost_threshold": args.cost_threshold,
        "buy_threshold": args.buy_threshold,
        "sell_threshold": args.sell_threshold,
        "max_width": args.max_width,
    }
    validation_result = evaluate_forecast_frame(
        validation_forecasts,
        horizon_h=args.pred_len,
        signal_mode=args.signal_mode,
        signal_kwargs=signal_kwargs,
        execution_rule=args.execution_rule,
        transaction_cost=args.transaction_cost,
        slippage=args.slippage,
        price_frame=raw_df,
    )
    final_test_result = evaluate_forecast_frame(
        final_test_forecasts,
        horizon_h=args.pred_len,
        signal_mode=args.signal_mode,
        signal_kwargs=signal_kwargs,
        execution_rule=args.execution_rule,
        transaction_cost=args.transaction_cost,
        slippage=args.slippage,
        price_frame=raw_df,
    )
    validation_metrics = validation_result["report"]["forecast_metrics"]
    final_test_metrics = final_test_result["report"]["forecast_metrics"]
    validation_per_symbol = _build_per_symbol_forecast_metrics(validation_forecasts)
    final_test_per_symbol = _build_per_symbol_forecast_metrics(final_test_forecasts)
    validation_regime_reports = build_regime_group_reports(
        validation_forecasts,
        raw_df=raw_df,
        horizon_h=args.pred_len,
        signal_mode=args.signal_mode,
        signal_kwargs=signal_kwargs,
        execution_rule=args.execution_rule,
        transaction_cost=args.transaction_cost,
        slippage=args.slippage,
    )
    final_test_regime_reports = build_regime_group_reports(
        final_test_forecasts,
        raw_df=raw_df,
        horizon_h=args.pred_len,
        signal_mode=args.signal_mode,
        signal_kwargs=signal_kwargs,
        execution_rule=args.execution_rule,
        transaction_cost=args.transaction_cost,
        slippage=args.slippage,
    )
    validation_per_symbol_ok, validation_per_symbol_failures = _passes_per_symbol_gate(
        validation_per_symbol,
        max_rmse=args.approval_max_per_symbol_validation_rmse,
        min_coverage=args.approval_min_per_symbol_validation_coverage,
    )
    final_test_per_symbol_ok, final_test_per_symbol_failures = _passes_per_symbol_gate(
        final_test_per_symbol,
        max_rmse=args.approval_max_per_symbol_final_test_rmse,
        min_coverage=args.approval_min_per_symbol_final_test_coverage,
    )
    global_validation_ok = (
        validation_metrics["rmse_q50"] <= args.approval_max_validation_rmse
        and validation_metrics["interval_coverage_rate"] >= args.approval_min_validation_coverage
    )
    global_final_test_ok = (
        final_test_metrics["rmse_q50"] <= args.approval_max_final_test_rmse
        and final_test_metrics["interval_coverage_rate"] >= args.approval_min_final_test_coverage
    )
    overall_gate_ok = (
        global_validation_ok
        and global_final_test_ok
        and validation_per_symbol_ok
        and final_test_per_symbol_ok
    )
    return {
        "backend": backend,
        "model_type": model_bundle.get("model_type"),
        "model_backend": model_bundle.get("model_backend", backend),
        "compute_device_requested": model_bundle.get("compute_device_requested"),
        "compute_device_resolved": model_bundle.get("compute_device_resolved"),
        "model_params": model_bundle.get("model_params", {}),
        "validation_report": validation_result["report"],
        "final_test_report": final_test_result["report"],
        "validation_per_symbol_metrics": validation_per_symbol,
        "final_test_per_symbol_metrics": final_test_per_symbol,
        "validation_regime_reports": validation_regime_reports,
        "final_test_regime_reports": final_test_regime_reports,
        "approval_gate_results": {
            "global_validation_gate_passed": global_validation_ok,
            "global_final_test_gate_passed": global_final_test_ok,
            "per_symbol_validation_gate_passed": validation_per_symbol_ok,
            "per_symbol_final_test_gate_passed": final_test_per_symbol_ok,
            "per_symbol_validation_failures": validation_per_symbol_failures,
            "per_symbol_final_test_failures": final_test_per_symbol_failures,
            "overall_gate_passed": overall_gate_ok,
        },
    }, validation_forecasts, final_test_forecasts


def _aggregate_symbol_forecasts(forecast_frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not forecast_frames:
        raise ValueError("no per-symbol forecast frames were produced")
    return pd.concat(forecast_frames, ignore_index=True).sort_values(["symbol", "origin_timestamp"]).reset_index(drop=True)


def _benchmark_per_symbol_baseline(
    *,
    backend: str,
    feature_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    validation_forecast_frames: list[pd.DataFrame] = []
    final_test_forecast_frames: list[pd.DataFrame] = []
    symbol_results: dict[str, Any] = {}
    grouped_symbols = list(feature_df.groupby("symbol", sort=True))
    progress_bar = ProgressBar(len(grouped_symbols), f"per-symbol baseline [{backend}]")

    for symbol, symbol_feature_df in grouped_symbols:
        symbol_raw_df = raw_df.loc[raw_df["symbol"] == symbol].copy()
        log_step(f"[benchmark] Preparing per-symbol dataset symbol={symbol} raw_rows={len(symbol_raw_df)} feature_rows={len(symbol_feature_df)}")
        split_config = SplitConfig(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            encoder_len=args.encoder_len,
            horizon_h=args.pred_len,
        )
        train_origins, val_origins, test_origins, split_meta = split_dataset(symbol_feature_df.copy(), split_config)
        log_step(
            f"[benchmark] Per-symbol split symbol={symbol} train_origins={len(train_origins)} "
            f"validation_origins={len(val_origins)} test_origins={len(test_origins)}"
        )
        train_raw, _, _ = _build_split_frames(symbol_feature_df, split_meta)
        with stage(f"Training per-symbol baseline [{backend}] for {symbol}"):
            result, validation_forecasts, final_test_forecasts = _benchmark_single_backend(
                backend=backend,
                feature_df=symbol_feature_df,
                train_raw=train_raw,
                val_raw=symbol_feature_df.loc[val_origins.index].copy(),
                test_raw=symbol_feature_df.loc[test_origins.index].copy(),
                train_origins=train_origins,
                val_origins=val_origins,
                test_origins=test_origins,
                raw_df=symbol_raw_df,
                args=args,
            )
        symbol_results[str(symbol)] = result
        symbol_final_metrics = result["final_test_report"]["forecast_metrics"]
        log_step(
            f"[benchmark] Per-symbol training finished symbol={symbol} backend={backend} "
            f"final_test_rmse={symbol_final_metrics['rmse_q50']:.6f} "
            f"final_test_coverage={symbol_final_metrics['interval_coverage_rate']:.6f}"
        )
        validation_forecast_frames.append(validation_forecasts)
        final_test_forecast_frames.append(final_test_forecasts)
        progress_bar.advance(str(symbol))

    validation_forecasts = _aggregate_symbol_forecasts(validation_forecast_frames)
    final_test_forecasts = _aggregate_symbol_forecasts(final_test_forecast_frames)

    signal_kwargs = {
        "cost_threshold": args.cost_threshold,
        "buy_threshold": args.buy_threshold,
        "sell_threshold": args.sell_threshold,
        "max_width": args.max_width,
    }
    validation_result = evaluate_forecast_frame(
        validation_forecasts,
        horizon_h=args.pred_len,
        signal_mode=args.signal_mode,
        signal_kwargs=signal_kwargs,
        execution_rule=args.execution_rule,
        transaction_cost=args.transaction_cost,
        slippage=args.slippage,
        price_frame=raw_df,
    )
    final_test_result = evaluate_forecast_frame(
        final_test_forecasts,
        horizon_h=args.pred_len,
        signal_mode=args.signal_mode,
        signal_kwargs=signal_kwargs,
        execution_rule=args.execution_rule,
        transaction_cost=args.transaction_cost,
        slippage=args.slippage,
        price_frame=raw_df,
    )
    validation_per_symbol = _build_per_symbol_forecast_metrics(validation_forecasts)
    final_test_per_symbol = _build_per_symbol_forecast_metrics(final_test_forecasts)
    validation_regime_reports = build_regime_group_reports(
        validation_forecasts,
        raw_df=raw_df,
        horizon_h=args.pred_len,
        signal_mode=args.signal_mode,
        signal_kwargs=signal_kwargs,
        execution_rule=args.execution_rule,
        transaction_cost=args.transaction_cost,
        slippage=args.slippage,
    )
    final_test_regime_reports = build_regime_group_reports(
        final_test_forecasts,
        raw_df=raw_df,
        horizon_h=args.pred_len,
        signal_mode=args.signal_mode,
        signal_kwargs=signal_kwargs,
        execution_rule=args.execution_rule,
        transaction_cost=args.transaction_cost,
        slippage=args.slippage,
    )
    validation_per_symbol_ok, validation_per_symbol_failures = _passes_per_symbol_gate(
        validation_per_symbol,
        max_rmse=args.approval_max_per_symbol_validation_rmse,
        min_coverage=args.approval_min_per_symbol_validation_coverage,
    )
    final_test_per_symbol_ok, final_test_per_symbol_failures = _passes_per_symbol_gate(
        final_test_per_symbol,
        max_rmse=args.approval_max_per_symbol_final_test_rmse,
        min_coverage=args.approval_min_per_symbol_final_test_coverage,
    )
    validation_metrics = validation_result["report"]["forecast_metrics"]
    final_test_metrics = final_test_result["report"]["forecast_metrics"]
    global_validation_ok = (
        validation_metrics["rmse_q50"] <= args.approval_max_validation_rmse
        and validation_metrics["interval_coverage_rate"] >= args.approval_min_validation_coverage
    )
    global_final_test_ok = (
        final_test_metrics["rmse_q50"] <= args.approval_max_final_test_rmse
        and final_test_metrics["interval_coverage_rate"] >= args.approval_min_final_test_coverage
    )
    overall_gate_ok = (
        global_validation_ok
        and global_final_test_ok
        and validation_per_symbol_ok
        and final_test_per_symbol_ok
    )

    return {
        "backend": backend,
        "deployment_mode": "per_symbol_baseline",
        "validation_report": validation_result["report"],
        "final_test_report": final_test_result["report"],
        "validation_per_symbol_metrics": validation_per_symbol,
        "final_test_per_symbol_metrics": final_test_per_symbol,
        "validation_regime_reports": validation_regime_reports,
        "final_test_regime_reports": final_test_regime_reports,
        "symbol_results": symbol_results,
        "approval_gate_results": {
            "global_validation_gate_passed": global_validation_ok,
            "global_final_test_gate_passed": global_final_test_ok,
            "per_symbol_validation_gate_passed": validation_per_symbol_ok,
            "per_symbol_final_test_gate_passed": final_test_per_symbol_ok,
            "per_symbol_validation_failures": validation_per_symbol_failures,
            "per_symbol_final_test_failures": final_test_per_symbol_failures,
            "overall_gate_passed": overall_gate_ok,
        },
    }


def _hybrid_comparison_summary(
    shared_result: dict[str, Any],
    per_symbol_result: dict[str, Any],
) -> dict[str, Any]:
    shared_validation = shared_result["validation_report"]["forecast_metrics"]
    shared_final_test = shared_result["final_test_report"]["forecast_metrics"]
    per_symbol_validation = per_symbol_result["validation_report"]["forecast_metrics"]
    per_symbol_final_test = per_symbol_result["final_test_report"]["forecast_metrics"]

    shared_symbol_metrics = shared_result["final_test_per_symbol_metrics"]
    per_symbol_symbol_metrics = per_symbol_result["final_test_per_symbol_metrics"]
    common_symbols = sorted(set(shared_symbol_metrics) & set(per_symbol_symbol_metrics))
    shared_wins: list[str] = []
    per_symbol_wins: list[str] = []
    ties: list[str] = []
    for symbol in common_symbols:
        shared_rmse = float(shared_symbol_metrics[symbol]["rmse_q50"])
        per_symbol_rmse = float(per_symbol_symbol_metrics[symbol]["rmse_q50"])
        if shared_rmse < per_symbol_rmse:
            shared_wins.append(symbol)
        elif per_symbol_rmse < shared_rmse:
            per_symbol_wins.append(symbol)
        else:
            ties.append(symbol)

    if per_symbol_final_test["rmse_q50"] + 1e-12 < shared_final_test["rmse_q50"]:
        recommended_mode = "hybrid_or_per_symbol"
        reason = "per_symbol_baseline_has_lower_aggregated_final_test_rmse"
    else:
        recommended_mode = "shared"
        reason = "shared_model_is_not_worse_than_per_symbol_baseline_on_aggregated_final_test_rmse"

    return {
        "shared_validation_rmse": float(shared_validation["rmse_q50"]),
        "shared_final_test_rmse": float(shared_final_test["rmse_q50"]),
        "per_symbol_validation_rmse": float(per_symbol_validation["rmse_q50"]),
        "per_symbol_final_test_rmse": float(per_symbol_final_test["rmse_q50"]),
        "shared_gate_passed": bool(shared_result["approval_gate_results"]["overall_gate_passed"]),
        "per_symbol_gate_passed": bool(per_symbol_result["approval_gate_results"]["overall_gate_passed"]),
        "shared_wins_on_symbols": shared_wins,
        "per_symbol_wins_on_symbols": per_symbol_wins,
        "tied_symbols": ties,
        "recommended_deployment_mode": recommended_mode,
        "reason": reason,
    }


def _select_recommended_backend(backend_results: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [result for result in backend_results if result["approval_gate_results"]["overall_gate_passed"]]
    pool = eligible if eligible else backend_results

    def _ranking_key(result: dict[str, Any]) -> tuple[float, float, int, str]:
        final_test_metrics = result["final_test_report"]["forecast_metrics"]
        validation_metrics = result["validation_report"]["forecast_metrics"]
        gate = result["approval_gate_results"]
        failure_count = len(gate["per_symbol_validation_failures"]) + len(gate["per_symbol_final_test_failures"])
        return (
            float(final_test_metrics["rmse_q50"]),
            float(validation_metrics["rmse_q50"]),
            failure_count,
            str(result["backend"]),
        )

    recommended = sorted(pool, key=_ranking_key)[0]
    return {
        "recommended_backend": recommended["backend"],
        "recommended_model_type": recommended["model_type"],
        "reason": "best_final_test_rmse_with_gate_pass" if eligible else "best_final_test_rmse_no_backend_passed_gate",
    }


def _write_report(output_dir: Path, report: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "model_benchmark_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def _base_report(
    *,
    training_symbols: list[str],
    universe_bundle,
    args: argparse.Namespace,
    backends: list[str],
    status: str,
    backend_results: list[dict[str, Any]] | None = None,
    per_symbol_baseline_results: list[dict[str, Any]] | None = None,
    hybrid_comparison: list[dict[str, Any]] | None = None,
    recommendation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "training_symbols": training_symbols,
        "training_universe_summary": summarize_training_universe(universe_bundle),
        "split_config": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "encoder_len": args.encoder_len,
            "horizon_h": args.pred_len,
        },
        "benchmarked_backends": backends,
        "backend_results": backend_results or [],
        "per_symbol_baseline_results": per_symbol_baseline_results or [],
        "hybrid_comparison": hybrid_comparison or [],
        "recommendation": recommendation,
    }


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = build_argparser().parse_args()

    training_symbols = _resolve_symbols(args)
    backends = _resolve_backends(args.model_backends)
    with stage("Loading training universe"):
        universe_bundle = load_training_universe_bundle(
            horizon_h=args.pred_len,
            symbols=training_symbols,
            log_fn=log_step,
        )
    raw_df = universe_bundle.raw_df
    feature_df = universe_bundle.feature_df
    log_step(
        f"[benchmark] Training universe ready symbols={len(training_symbols)} raw_rows={len(raw_df)} "
        f"feature_rows={len(feature_df)}"
    )

    split_config = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        encoder_len=args.encoder_len,
        horizon_h=args.pred_len,
    )
    with stage("Creating benchmark splits"):
        train_origins, val_origins, test_origins, split_meta = split_dataset(feature_df, split_config)
        train_raw, val_raw, test_raw = _build_split_frames(feature_df, split_meta)
    log_step(
        f"[benchmark] Global split summary train_origins={len(train_origins)} "
        f"validation_origins={len(val_origins)} test_origins={len(test_origins)}"
    )
    log_step(
        f"[benchmark] Global split ranges "
        f"train={split_meta['train_start_time']}..{split_meta['train_end_time']} "
        f"validation={split_meta['val_start_time']}..{split_meta['val_end_time']} "
        f"test={split_meta['test_start_time']}..{split_meta['test_end_time']}"
    )

    output_dir = Path(args.output_dir)
    report_path = _write_report(
        output_dir,
        _base_report(
            training_symbols=training_symbols,
            universe_bundle=universe_bundle,
            args=args,
            backends=backends,
            status="running",
            recommendation=None,
        ),
    )
    print(f"[benchmark] initialized report: {report_path}", flush=True)
    backend_results: list[dict[str, Any]] = []
    for index, backend in enumerate(backends, start=1):
        print(f"[benchmark] shared backend {index}/{len(backends)} -> {backend}", flush=True)
        with stage(f"Training shared backend [{backend}]"):
            backend_result, _, _ = _benchmark_single_backend(
                backend=backend,
                feature_df=feature_df,
                train_raw=train_raw,
                val_raw=val_raw,
                test_raw=test_raw,
                train_origins=train_origins,
                val_origins=val_origins,
                test_origins=test_origins,
                raw_df=raw_df,
                args=args,
            )
        backend_results.append(backend_result)
        shared_final_metrics = backend_result["final_test_report"]["forecast_metrics"]
        log_step(
            f"[benchmark] Shared backend finished backend={backend} "
            f"device={backend_result.get('compute_device_resolved', 'n/a')} "
            f"final_test_rmse={shared_final_metrics['rmse_q50']:.6f} "
            f"final_test_coverage={shared_final_metrics['interval_coverage_rate']:.6f}"
        )
        report_path = _write_report(
            output_dir,
            _base_report(
                training_symbols=training_symbols,
                universe_bundle=universe_bundle,
                args=args,
                backends=backends,
                status="running",
                backend_results=backend_results,
                recommendation=_select_recommended_backend(backend_results),
            ),
        )
        print(f"[benchmark] partial report updated: {report_path}", flush=True)
    per_symbol_baseline_results: list[dict[str, Any]] = []
    hybrid_comparison: list[dict[str, Any]] = []
    if getattr(args, "compare_per_symbol_baseline", False):
        for index, backend in enumerate(backends, start=1):
            print(f"[benchmark] per-symbol baseline {index}/{len(backends)} -> {backend}", flush=True)
            per_symbol_baseline_results.append(
                _benchmark_per_symbol_baseline(
                    backend=backend,
                    feature_df=feature_df,
                    raw_df=raw_df,
                    args=args,
                )
            )
            baseline_final_metrics = per_symbol_baseline_results[-1]["final_test_report"]["forecast_metrics"]
            log_step(
                f"[benchmark] Per-symbol baseline aggregate finished backend={backend} "
                f"final_test_rmse={baseline_final_metrics['rmse_q50']:.6f} "
                f"final_test_coverage={baseline_final_metrics['interval_coverage_rate']:.6f}"
            )
            report_path = _write_report(
                output_dir,
                _base_report(
                    training_symbols=training_symbols,
                    universe_bundle=universe_bundle,
                    args=args,
                    backends=backends,
                    status="running",
                    backend_results=backend_results,
                    per_symbol_baseline_results=per_symbol_baseline_results,
                    recommendation=_select_recommended_backend(backend_results),
                ),
            )
            print(f"[benchmark] partial report updated: {report_path}", flush=True)
        for shared_result, per_symbol_result in zip(backend_results, per_symbol_baseline_results, strict=False):
            hybrid_comparison.append(
                {
                    "backend": shared_result["backend"],
                    "comparison": _hybrid_comparison_summary(shared_result, per_symbol_result),
                }
            )

    report = _base_report(
        training_symbols=training_symbols,
        universe_bundle=universe_bundle,
        args=args,
        backends=backends,
        status="completed",
        backend_results=backend_results,
        per_symbol_baseline_results=per_symbol_baseline_results,
        hybrid_comparison=hybrid_comparison,
        recommendation=_select_recommended_backend(backend_results),
    )

    report_path = _write_report(output_dir, report)

    print(f"Benchmark report saved to {report_path}")
    log_step(f"[benchmark] Completed successfully report={report_path}")
    print(json.dumps(report["recommendation"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
