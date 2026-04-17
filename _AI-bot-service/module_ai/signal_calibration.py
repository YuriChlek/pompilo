from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from module_ai.artifacts import default_artifact_dir
from module_ai.console import ProgressBar, log_step, stage
from module_ai.data_access import read_training_universe_candles
from module_ai.data_pipeline import (
    MULTI_SYMBOL_SENTINEL,
    build_feature_dataframe,
    load_scaler_artifact,
    transform_features,
    validate_artifact_metadata,
    validate_scaler_metadata_against_dataset_metadata,
)
from module_ai.evaluate import evaluate_forecast_frame
from module_ai.modeling import MODEL_ARTIFACT_NAME, load_model_bundle, validate_model_bundle_against_dataset_metadata
from module_ai.runtime import predict_origin_frame
from module_ai.splits import SplitConfig, split_dataset
from module_ai.symbols import normalize_cli_symbol, normalize_training_symbols


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate forecast->signal thresholds on validation and confirm the top candidates on final test."
    )
    parser.add_argument("--symbol", default="ALL", help="Trading symbol or ALL for shared multi-symbol calibration")
    parser.add_argument("--artifact-dir", default="", help="Candidate artifact directory to calibrate")
    parser.add_argument("--signal-modes", default="quantile_barrier,median_with_width", help="Comma-separated signal policies")
    parser.add_argument("--cost-threshold-grid", default="", help="Comma-separated grid for quantile_barrier cost_threshold")
    parser.add_argument("--buy-threshold-grid", default="", help="Comma-separated grid for median_with_width buy_threshold")
    parser.add_argument("--sell-threshold-grid", default="", help="Comma-separated grid for median_with_width sell_threshold")
    parser.add_argument("--max-width-grid", default="", help="Comma-separated grid for median_with_width max_width")
    parser.add_argument("--top-k", type=int, default=12, help="How many best validation candidates to keep in the report")
    parser.add_argument("--min-signal-share", type=float, default=0.002, help="Minimum BUY+SELL share for a viable candidate")
    parser.add_argument("--max-signal-share", type=float, default=0.35, help="Maximum BUY+SELL share for a viable candidate")
    parser.add_argument("--target-signal-share", type=float, default=0.05, help="Target BUY+SELL share used in validation scoring")
    parser.add_argument("--min-trade-count", type=int, default=25, help="Minimum number of trades for a viable candidate")
    parser.add_argument("--execution-rule", default="close_to_close", help="Trade simulation execution rule")
    parser.add_argument("--transaction-cost", type=float, default=0.0, help="Transaction cost per trade")
    parser.add_argument("--slippage", type=float, default=0.0, help="Slippage per trade")
    parser.add_argument(
        "--output-path",
        default="",
        help="Where to save the calibration JSON report. Defaults to <artifact-dir>/signal_calibration_report.json",
    )
    return parser


def _load_candidate_metadata(artifact_dir: Path) -> dict[str, Any]:
    meta_path = artifact_dir / "dataset_meta.pkl"
    if not meta_path.exists():
        raise ValueError(f"artifact metadata not found: {meta_path}")
    metadata = joblib.load(meta_path)
    validate_artifact_metadata(metadata)
    return metadata


def _load_candidate_components(artifact_dir: Path, dataset_meta: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    scalers, scaler_meta = load_scaler_artifact(artifact_dir / "scalers.pkl")
    validate_scaler_metadata_against_dataset_metadata(scaler_meta, dataset_meta)
    model_bundle = load_model_bundle(artifact_dir / MODEL_ARTIFACT_NAME)
    validate_model_bundle_against_dataset_metadata(model_bundle, dataset_meta)
    return scalers, model_bundle


def _resolve_symbol_filter(requested_symbol: str, dataset_meta: dict[str, Any]) -> list[str]:
    requested_symbol_normalized = normalize_cli_symbol(requested_symbol)
    training_symbols = normalize_training_symbols(list(dataset_meta.get("training_symbols", [])))
    artifact_symbol = str(dataset_meta.get("symbol", "")).strip()
    artifact_symbol_normalized = artifact_symbol.upper() if artifact_symbol != MULTI_SYMBOL_SENTINEL else MULTI_SYMBOL_SENTINEL

    if artifact_symbol and artifact_symbol_normalized != MULTI_SYMBOL_SENTINEL and requested_symbol_normalized != artifact_symbol_normalized:
        raise ValueError("symbol argument does not match candidate artifact symbol")

    if artifact_symbol_normalized == MULTI_SYMBOL_SENTINEL:
        if requested_symbol_normalized in {"", "ALL"}:
            return training_symbols
        if requested_symbol_normalized not in training_symbols:
            raise ValueError("symbol argument is not present in the shared artifact training universe")
        return [requested_symbol_normalized]

    return [artifact_symbol_normalized or requested_symbol_normalized]


def _load_split_config(dataset_meta: dict[str, Any]) -> SplitConfig:
    split_config_data = dataset_meta.get("split_config", {})
    if not split_config_data:
        raise ValueError("candidate artifact metadata is missing split_config")
    return SplitConfig(
        train_ratio=float(split_config_data["train_ratio"]),
        val_ratio=float(split_config_data["val_ratio"]),
        test_ratio=float(split_config_data["test_ratio"]),
        encoder_len=int(split_config_data["encoder_len"]),
        horizon_h=int(split_config_data["horizon_h"]),
    )


def _predict_saved_artifact(
    feature_df: pd.DataFrame,
    *,
    origin_df: pd.DataFrame,
    scaled_features: list[str],
    scalers: dict[str, Any],
    model_bundle: dict[str, Any],
    encoder_len: int,
) -> pd.DataFrame:
    scaled_feature_df = transform_features(feature_df, scalers, scaled_features)
    forecast_df, _ = predict_origin_frame(
        scaled_feature_df,
        origin_df=origin_df,
        encoder_len=encoder_len,
        model_bundle=model_bundle,
    )
    return forecast_df


def _parse_mode_list(raw_value: str) -> list[str]:
    modes = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not modes:
        raise ValueError("at least one signal mode must be provided")
    invalid = [mode for mode in modes if mode not in {"quantile_barrier", "median_with_width"}]
    if invalid:
        raise ValueError(f"unsupported signal mode(s): {invalid}")
    deduplicated: list[str] = []
    for mode in modes:
        if mode not in deduplicated:
            deduplicated.append(mode)
    return deduplicated


def _parse_float_grid(raw_value: str) -> list[float]:
    values = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    return [float(item) for item in values]


def _rounded_unique(values: list[float], *, min_value: float | None = None) -> list[float]:
    normalized = []
    for value in values:
        if not np.isfinite(value):
            continue
        rounded = round(float(value), 6)
        if min_value is not None and rounded < min_value:
            continue
        normalized.append(rounded)
    unique = sorted(set(normalized))
    return unique


def _derive_default_grids(validation_forecasts: pd.DataFrame) -> dict[str, list[float]]:
    abs_q50 = validation_forecasts["q50_return_h"].abs().to_numpy(dtype=np.float64)
    interval_width = (validation_forecasts["q90_return_h"] - validation_forecasts["q10_return_h"]).to_numpy(dtype=np.float64)

    if len(abs_q50) == 0 or len(interval_width) == 0:
        raise ValueError("validation forecasts are empty")

    q50_quantiles = np.quantile(abs_q50, [0.0, 0.25, 0.5, 0.75, 0.9])
    width_quantiles = np.quantile(interval_width, [0.15, 0.3, 0.5, 0.65, 0.8])
    median_abs_q50 = float(np.quantile(abs_q50, 0.5))

    cost_grid = _rounded_unique(
        [
            0.0,
            median_abs_q50 * 0.15,
            median_abs_q50 * 0.3,
            median_abs_q50 * 0.5,
            median_abs_q50 * 0.75,
        ],
        min_value=0.0,
    )
    threshold_grid = _rounded_unique([0.0, *list(q50_quantiles)], min_value=0.0)
    width_grid = _rounded_unique(list(width_quantiles), min_value=0.0)
    if not width_grid:
        width_grid = [0.05]

    return {
        "cost_threshold_grid": cost_grid or [0.0],
        "buy_threshold_grid": threshold_grid or [0.0],
        "sell_threshold_grid": threshold_grid or [0.0],
        "max_width_grid": width_grid,
    }


def _summarize_forecast_distribution(forecast_df: pd.DataFrame) -> dict[str, Any]:
    width = forecast_df["q90_return_h"] - forecast_df["q10_return_h"]
    q50 = forecast_df["q50_return_h"]
    q10 = forecast_df["q10_return_h"]
    q90 = forecast_df["q90_return_h"]
    return {
        "forecast_count": int(len(forecast_df)),
        "q50_abs_quantiles": {
            "p25": float(q50.abs().quantile(0.25)),
            "p50": float(q50.abs().quantile(0.50)),
            "p75": float(q50.abs().quantile(0.75)),
            "p90": float(q50.abs().quantile(0.90)),
        },
        "interval_width_quantiles": {
            "p15": float(width.quantile(0.15)),
            "p30": float(width.quantile(0.30)),
            "p50": float(width.quantile(0.50)),
            "p65": float(width.quantile(0.65)),
            "p80": float(width.quantile(0.80)),
        },
        "neutral_crossing_shares": {
            "q10_gt_zero_share": float((q10 > 0.0).mean()),
            "q90_lt_zero_share": float((q90 < 0.0).mean()),
            "interval_crosses_zero_share": float(((q10 <= 0.0) & (q90 >= 0.0)).mean()),
        },
    }


def _build_candidate_configs(
    *,
    signal_modes: list[str],
    cost_threshold_grid: list[float],
    buy_threshold_grid: list[float],
    sell_threshold_grid: list[float],
    max_width_grid: list[float],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if "quantile_barrier" in signal_modes:
        for cost_threshold in cost_threshold_grid:
            candidates.append(
                {
                    "signal_mode": "quantile_barrier",
                    "signal_kwargs": {
                        "cost_threshold": float(cost_threshold),
                    },
                }
            )
    if "median_with_width" in signal_modes:
        for buy_threshold in buy_threshold_grid:
            for sell_threshold in sell_threshold_grid:
                for max_width in max_width_grid:
                    candidates.append(
                        {
                            "signal_mode": "median_with_width",
                            "signal_kwargs": {
                                "buy_threshold": float(buy_threshold),
                                "sell_threshold": float(sell_threshold),
                                "max_width": float(max_width),
                            },
                        }
                    )
    if not candidates:
        raise ValueError("no candidate threshold configurations were generated")
    return candidates


def _extract_candidate_metrics(report: dict[str, Any]) -> dict[str, float]:
    forecast_metrics = report.get("forecast_metrics", {})
    signal_metrics = report.get("signal_metrics", {})
    trading_metrics = report.get("trading_metrics", {})
    signal_count = float(signal_metrics.get("signal_count", 0.0))
    buy_signal_count = float(signal_metrics.get("buy_signal_count", 0.0))
    sell_signal_count = float(signal_metrics.get("sell_signal_count", 0.0))
    active_signal_count = buy_signal_count + sell_signal_count
    signal_share = float(active_signal_count / signal_count) if signal_count > 0 else 0.0
    return {
        "rmse_q50": float(forecast_metrics.get("rmse_q50", float("inf"))),
        "interval_coverage_rate": float(forecast_metrics.get("interval_coverage_rate", float("-inf"))),
        "buy_signal_count": buy_signal_count,
        "sell_signal_count": sell_signal_count,
        "hold_signal_count": float(signal_metrics.get("hold_signal_count", 0.0)),
        "signal_share": signal_share,
        "directional_accuracy": float(signal_metrics.get("directional_accuracy", 0.0)),
        "trade_count": float(trading_metrics.get("trade_count", 0.0)),
        "turnover": float(trading_metrics.get("turnover", 0.0)),
        "cumulative_return": float(trading_metrics.get("cumulative_return", 0.0)),
        "average_trade_return": float(trading_metrics.get("average_trade_return", 0.0)),
        "hit_rate": float(trading_metrics.get("hit_rate", 0.0)),
        "max_drawdown": float(trading_metrics.get("max_drawdown", 0.0)),
        "profit_factor": float(trading_metrics.get("profit_factor", 0.0)),
        "sharpe_like": float(trading_metrics.get("sharpe_like", 0.0)),
    }


def _score_candidate(
    metrics: dict[str, float],
    *,
    min_signal_share: float,
    max_signal_share: float,
    target_signal_share: float,
    min_trade_count: int,
) -> tuple[float, bool, list[str]]:
    reasons: list[str] = []
    trade_count = int(metrics["trade_count"])
    signal_share = metrics["signal_share"]
    viable = True

    if trade_count < min_trade_count:
        viable = False
        reasons.append("trade_count_below_minimum")
    if signal_share < min_signal_share:
        viable = False
        reasons.append("signal_share_below_minimum")
    if signal_share > max_signal_share:
        viable = False
        reasons.append("signal_share_above_maximum")

    density_penalty = abs(signal_share - target_signal_share)
    score = (
        metrics["cumulative_return"] * 100.0
        + metrics["sharpe_like"] * 10.0
        + metrics["hit_rate"] * 2.0
        + metrics["directional_accuracy"] * 1.0
        - metrics["max_drawdown"] * 50.0
        - density_penalty * 5.0
    )
    if not viable:
        score -= 1000.0
    return float(score), viable, reasons


def _evaluate_candidate(
    candidate: dict[str, Any],
    *,
    validation_forecasts: pd.DataFrame,
    final_test_forecasts: pd.DataFrame,
    horizon_h: int,
    execution_rule: str,
    transaction_cost: float,
    slippage: float,
    price_frame: pd.DataFrame,
    min_signal_share: float,
    max_signal_share: float,
    target_signal_share: float,
    min_trade_count: int,
) -> dict[str, Any]:
    validation_result = evaluate_forecast_frame(
        validation_forecasts,
        horizon_h=horizon_h,
        signal_mode=candidate["signal_mode"],
        signal_kwargs=candidate["signal_kwargs"],
        execution_rule=execution_rule,
        transaction_cost=transaction_cost,
        slippage=slippage,
        price_frame=price_frame,
    )
    final_test_result = evaluate_forecast_frame(
        final_test_forecasts,
        horizon_h=horizon_h,
        signal_mode=candidate["signal_mode"],
        signal_kwargs=candidate["signal_kwargs"],
        execution_rule=execution_rule,
        transaction_cost=transaction_cost,
        slippage=slippage,
        price_frame=price_frame,
    )

    validation_metrics = _extract_candidate_metrics(validation_result["report"])
    final_test_metrics = _extract_candidate_metrics(final_test_result["report"])
    validation_score, viable, reasons = _score_candidate(
        validation_metrics,
        min_signal_share=min_signal_share,
        max_signal_share=max_signal_share,
        target_signal_share=target_signal_share,
        min_trade_count=min_trade_count,
    )

    return {
        "signal_mode": candidate["signal_mode"],
        "signal_kwargs": candidate["signal_kwargs"],
        "validation": validation_metrics,
        "final_test": final_test_metrics,
        "validation_score": validation_score,
        "viable_on_validation": viable,
        "validation_rejection_reasons": reasons,
    }


def _candidate_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    validation = item["validation"]
    return (
        0 if item["viable_on_validation"] else 1,
        -float(item["validation_score"]),
        -float(validation["cumulative_return"]),
        float(validation["max_drawdown"]),
        -float(validation["trade_count"]),
        item["signal_mode"],
        json.dumps(item["signal_kwargs"], sort_keys=True),
    )


def _print_summary(report: dict[str, Any]) -> None:
    recommendation = report["recommendation"]
    log_step("Calibration complete.")
    log_step(f"Artifact: {report['artifact_dir']}")
    log_step(f"Symbols: {', '.join(report['training_symbols'])}")
    log_step(f"Selected candidate: mode={recommendation['signal_mode']} kwargs={recommendation['signal_kwargs']}")
    log_step(
        "Validation: "
        f"score={recommendation['validation_score']:.4f} "
        f"trades={int(recommendation['validation']['trade_count'])} "
        f"signal_share={recommendation['validation']['signal_share']:.4f} "
        f"cumret={recommendation['validation']['cumulative_return']:.6f}"
    )
    log_step(
        "Final test: "
        f"trades={int(recommendation['final_test']['trade_count'])} "
        f"signal_share={recommendation['final_test']['signal_share']:.4f} "
        f"cumret={recommendation['final_test']['cumulative_return']:.6f}"
    )
    log_step(f"Report saved to: {report['output_path']}")


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = build_argparser().parse_args()

    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else default_artifact_dir(args.symbol)
    with stage("Loading candidate artifact metadata"):
        dataset_meta = _load_candidate_metadata(artifact_dir)
    symbol_filter = _resolve_symbol_filter(args.symbol, dataset_meta)
    split_config = _load_split_config(dataset_meta)
    log_step(
        f"Calibration target symbols: {len(symbol_filter)} -> "
        f"{', '.join(symbol_filter[:8])}{' ...' if len(symbol_filter) > 8 else ''}"
    )
    symbol_id_map = {
        str(key).strip().upper(): int(value)
        for key, value in dict(dataset_meta.get("symbol_id_map", {})).items()
    }
    with stage("Loading calibration candles from PostgreSQL"):
        raw_df = read_training_universe_candles(symbol_filter, order="ASC", clean=True)
    with stage("Building calibration feature dataframe"):
        feature_df = build_feature_dataframe(
            raw_df,
            horizon_h=split_config.horizon_h,
            symbol_id_map=symbol_id_map,
        )
    with stage("Rebuilding validation and final test splits"):
        _, val_origins, test_origins, _ = split_dataset(feature_df, split_config)
    with stage("Loading scaler and model artifacts"):
        scalers, model_bundle = _load_candidate_components(artifact_dir, dataset_meta)
    scaled_features = list(dataset_meta["scaled_features"])

    with stage("Generating validation forecasts for calibration"):
        validation_forecasts = _predict_saved_artifact(
            feature_df,
            origin_df=val_origins,
            scaled_features=scaled_features,
            scalers=scalers,
            model_bundle=model_bundle,
            encoder_len=split_config.encoder_len,
        )
    with stage("Generating final test forecasts for calibration"):
        final_test_forecasts = _predict_saved_artifact(
            feature_df,
            origin_df=test_origins,
            scaled_features=scaled_features,
            scalers=scalers,
            model_bundle=model_bundle,
            encoder_len=split_config.encoder_len,
        )
    log_step(
        f"Forecast rows ready: validation={len(validation_forecasts)} final_test={len(final_test_forecasts)}"
    )

    with stage("Deriving default threshold grids"):
        derived_grids = _derive_default_grids(validation_forecasts)
    cost_threshold_grid = _parse_float_grid(args.cost_threshold_grid) or derived_grids["cost_threshold_grid"]
    buy_threshold_grid = _parse_float_grid(args.buy_threshold_grid) or derived_grids["buy_threshold_grid"]
    sell_threshold_grid = _parse_float_grid(args.sell_threshold_grid) or derived_grids["sell_threshold_grid"]
    max_width_grid = _parse_float_grid(args.max_width_grid) or derived_grids["max_width_grid"]
    signal_modes = _parse_mode_list(args.signal_modes)

    with stage("Building threshold candidate grid"):
        candidates = _build_candidate_configs(
            signal_modes=signal_modes,
            cost_threshold_grid=cost_threshold_grid,
            buy_threshold_grid=buy_threshold_grid,
            sell_threshold_grid=sell_threshold_grid,
            max_width_grid=max_width_grid,
        )
    log_step(f"Calibration candidates: {len(candidates)}")

    progress_bar = ProgressBar(len(candidates), "threshold calibration")
    evaluated_candidates: list[dict[str, Any]] = []
    with stage("Evaluating threshold candidates"):
        for candidate in candidates:
            evaluated_candidates.append(
                _evaluate_candidate(
                    candidate,
                    validation_forecasts=validation_forecasts,
                    final_test_forecasts=final_test_forecasts,
                    horizon_h=split_config.horizon_h,
                    execution_rule=args.execution_rule,
                    transaction_cost=args.transaction_cost,
                    slippage=args.slippage,
                    price_frame=raw_df,
                    min_signal_share=args.min_signal_share,
                    max_signal_share=args.max_signal_share,
                    target_signal_share=args.target_signal_share,
                    min_trade_count=args.min_trade_count,
                )
            )
            progress_bar.advance(
                f"{candidate['signal_mode']} {json.dumps(candidate['signal_kwargs'], sort_keys=True)}"
            )
    ranked_candidates = sorted(evaluated_candidates, key=_candidate_sort_key)
    top_candidates = ranked_candidates[: max(int(args.top_k), 1)]
    best_candidate = top_candidates[0]

    output_path = Path(args.output_path) if args.output_path else artifact_dir / "signal_calibration_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "artifact_dir": str(artifact_dir),
        "output_path": str(output_path),
        "training_symbols": symbol_filter,
        "artifact_training_symbols": list(dataset_meta.get("training_symbols", [])),
        "model_backend": model_bundle.get("model_backend"),
        "model_type": model_bundle.get("model_type"),
        "horizon_h": split_config.horizon_h,
        "execution_rule": args.execution_rule,
        "transaction_cost": args.transaction_cost,
        "slippage": args.slippage,
        "candidate_count": len(evaluated_candidates),
        "search_space": {
            "signal_modes": signal_modes,
            "cost_threshold_grid": cost_threshold_grid,
            "buy_threshold_grid": buy_threshold_grid,
            "sell_threshold_grid": sell_threshold_grid,
            "max_width_grid": max_width_grid,
            "min_signal_share": args.min_signal_share,
            "max_signal_share": args.max_signal_share,
            "target_signal_share": args.target_signal_share,
            "min_trade_count": args.min_trade_count,
        },
        "validation_distribution_summary": _summarize_forecast_distribution(validation_forecasts),
        "recommendation": best_candidate,
        "top_candidates": top_candidates,
    }
    with stage("Saving calibration report"):
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
