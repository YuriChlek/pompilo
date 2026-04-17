from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from module_ai.artifacts import default_artifact_dir
from module_ai.console import log_step, stage
from module_ai.data_access import (
    read_training_universe_candles,
)
from module_ai.data_pipeline import (
    MULTI_SYMBOL_SENTINEL,
    build_feature_dataframe,
    load_scaler_artifact,
    transform_features,
    validate_artifact_metadata,
    validate_scaler_metadata_against_dataset_metadata,
)
from module_ai.evaluate import evaluate_forecast_frame
from module_ai.modeling import (
    MODEL_ARTIFACT_NAME,
    load_model_bundle,
    validate_model_bundle_against_dataset_metadata,
)
from module_ai.regime_reporting import build_regime_group_reports
from module_ai.runtime import predict_origin_frame
from module_ai.splits import SplitConfig, split_dataset
from module_ai.symbols import normalize_cli_symbol, normalize_training_symbols
from module_ai.training_universe import TrainingUniverseBundle, summarize_training_universe


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run canonical offline evaluation for a saved candidate artifact.")
    parser.add_argument("--symbol", default="ALL", help="Trading symbol or ALL for shared multi-symbol evaluation")
    parser.add_argument("--artifact-dir", default="", help="Candidate artifact directory to evaluate and approve")
    parser.add_argument("--signal-mode", default="quantile_barrier")
    parser.add_argument("--cost-threshold", type=float, default=0.0)
    parser.add_argument("--buy-threshold", type=float, default=0.0)
    parser.add_argument("--sell-threshold", type=float, default=0.0)
    parser.add_argument("--max-width", type=float, default=0.05)
    parser.add_argument("--execution-rule", default="close_to_close")
    parser.add_argument("--transaction-cost", type=float, default=0.0)
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument("--approval-max-validation-rmse", "--approval-max-walk-forward-rmse", dest="approval_max_validation_rmse", type=float, default=0.08)
    parser.add_argument("--approval-max-final-test-rmse", type=float, default=0.08)
    parser.add_argument("--approval-min-validation-coverage", "--approval-min-walk-forward-coverage", dest="approval_min_validation_coverage", type=float, default=0.50)
    parser.add_argument("--approval-min-final-test-coverage", type=float, default=0.50)
    parser.add_argument("--approval-max-per-symbol-validation-rmse", type=float, default=0.12)
    parser.add_argument("--approval-max-per-symbol-final-test-rmse", type=float, default=0.12)
    parser.add_argument("--approval-min-per-symbol-validation-coverage", type=float, default=0.10)
    parser.add_argument("--approval-min-per-symbol-final-test-coverage", type=float, default=0.10)
    return parser


def _build_per_symbol_reports(
    forecast_df: pd.DataFrame,
    *,
    horizon_h: int,
    signal_mode: str,
    signal_kwargs: dict[str, float],
    execution_rule: str,
    transaction_cost: float,
    slippage: float,
    price_frame: pd.DataFrame,
) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    for symbol, group in forecast_df.groupby("symbol", sort=True):
        symbol_key = str(symbol)
        symbol_price_frame = price_frame.loc[price_frame["symbol"].astype(str) == symbol_key].copy()
        reports[symbol_key] = evaluate_forecast_frame(
            group,
            horizon_h=horizon_h,
            signal_mode=signal_mode,
            signal_kwargs=signal_kwargs,
            execution_rule=execution_rule,
            transaction_cost=transaction_cost,
            slippage=slippage,
            price_frame=symbol_price_frame,
        )["report"]
    return reports


def _passes_per_symbol_gate(
    per_symbol_report: dict[str, dict],
    *,
    max_rmse: float,
    min_coverage: float,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for symbol, report in per_symbol_report.items():
        forecast_metrics = report.get("forecast_metrics", {})
        rmse = float(forecast_metrics.get("rmse_q50", float("inf")))
        coverage = float(forecast_metrics.get("interval_coverage_rate", float("-inf")))
        if rmse > max_rmse or coverage < min_coverage:
            failures.append(symbol)
    return len(failures) == 0, failures


def _extract_symbol_scoreboard(per_symbol_report: dict[str, dict]) -> list[dict[str, float | str]]:
    scoreboard: list[dict[str, float | str]] = []
    for symbol, report in per_symbol_report.items():
        forecast_metrics = report.get("forecast_metrics", {})
        scoreboard.append(
            {
                "symbol": symbol,
                "rmse_q50": float(forecast_metrics.get("rmse_q50", float("inf"))),
                "mae_q50": float(forecast_metrics.get("mae_q50", float("inf"))),
                "interval_coverage_rate": float(forecast_metrics.get("interval_coverage_rate", float("-inf"))),
            }
        )
    return sorted(scoreboard, key=lambda item: (item["rmse_q50"], -item["interval_coverage_rate"], item["symbol"]))


def _load_candidate_metadata(artifact_dir: Path) -> dict:
    meta_path = artifact_dir / "dataset_meta.pkl"
    if not meta_path.exists():
        raise ValueError(f"artifact metadata not found: {meta_path}")
    metadata = joblib.load(meta_path)
    validate_artifact_metadata(metadata)
    return metadata


def _load_candidate_components(artifact_dir: Path, dataset_meta: dict) -> tuple[dict, dict]:
    scalers, scaler_meta = load_scaler_artifact(artifact_dir / "scalers.pkl")
    validate_scaler_metadata_against_dataset_metadata(scaler_meta, dataset_meta)
    model_bundle = load_model_bundle(artifact_dir / MODEL_ARTIFACT_NAME)
    validate_model_bundle_against_dataset_metadata(model_bundle, dataset_meta)
    return scalers, model_bundle


def _predict_saved_artifact(
    feature_df: pd.DataFrame,
    *,
    origin_df: pd.DataFrame,
    scaled_features: list[str],
    scalers: dict,
    model_bundle: dict,
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


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = build_argparser().parse_args()
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else default_artifact_dir(args.symbol)
    with stage("Loading candidate artifact metadata"):
        dataset_meta = _load_candidate_metadata(artifact_dir)

    requested_symbol = normalize_cli_symbol(args.symbol)
    training_symbols = normalize_training_symbols(list(dataset_meta.get("training_symbols", [])))
    artifact_symbol = str(dataset_meta.get("symbol", "")).strip()
    artifact_symbol_normalized = artifact_symbol.upper() if artifact_symbol != MULTI_SYMBOL_SENTINEL else MULTI_SYMBOL_SENTINEL
    if artifact_symbol and artifact_symbol_normalized != MULTI_SYMBOL_SENTINEL and requested_symbol != artifact_symbol_normalized:
        raise ValueError("symbol argument does not match candidate artifact symbol")
    if artifact_symbol_normalized == MULTI_SYMBOL_SENTINEL:
        if requested_symbol in {"", "ALL"}:
            symbol_filter = training_symbols
        else:
            if requested_symbol not in training_symbols:
                raise ValueError("symbol argument is not present in the shared artifact training universe")
            symbol_filter = [requested_symbol]
    else:
        symbol_filter = [artifact_symbol_normalized or requested_symbol]

    split_config_data = dataset_meta.get("split_config", {})
    if not split_config_data:
        raise ValueError("candidate artifact metadata is missing split_config")
    split_config = SplitConfig(
        train_ratio=float(split_config_data["train_ratio"]),
        val_ratio=float(split_config_data["val_ratio"]),
        test_ratio=float(split_config_data["test_ratio"]),
        encoder_len=int(split_config_data["encoder_len"]),
        horizon_h=int(split_config_data["horizon_h"]),
    )
    log_step(
        f"Evaluation target symbols: {len(symbol_filter)} -> "
        f"{', '.join(symbol_filter[:8])}{' ...' if len(symbol_filter) > 8 else ''}"
    )

    symbol_id_map = {
        str(key).strip().upper(): int(value)
        for key, value in dict(dataset_meta.get("symbol_id_map", {})).items()
    }
    with stage("Loading evaluation candles from PostgreSQL"):
        raw_df = read_training_universe_candles(symbol_filter, order="ASC", clean=True)
    with stage("Building evaluation feature dataframe"):
        feature_df = build_feature_dataframe(
            raw_df,
            horizon_h=split_config.horizon_h,
            symbol_id_map=symbol_id_map,
        )
    with stage("Rebuilding validation and test splits"):
        _, val_origins, test_origins, _ = split_dataset(feature_df, split_config)
    log_step(
        f"Evaluation origins: validation={len(val_origins)} test={len(test_origins)}"
    )

    with stage("Loading scaler and model artifacts"):
        scalers, model_bundle = _load_candidate_components(artifact_dir, dataset_meta)
    scaled_features = list(dataset_meta["scaled_features"])

    with stage("Generating validation forecasts"):
        validation_forecasts = _predict_saved_artifact(
            feature_df,
            origin_df=val_origins,
            scaled_features=scaled_features,
            scalers=scalers,
            model_bundle=model_bundle,
            encoder_len=split_config.encoder_len,
        )
    with stage("Generating final test forecasts"):
        final_test_forecasts = _predict_saved_artifact(
            feature_df,
            origin_df=test_origins,
            scaled_features=scaled_features,
            scalers=scalers,
            model_bundle=model_bundle,
            encoder_len=split_config.encoder_len,
        )
    log_step(
        f"Forecast rows: validation={len(validation_forecasts)} final_test={len(final_test_forecasts)}"
    )

    signal_kwargs = {
        "cost_threshold": args.cost_threshold,
        "buy_threshold": args.buy_threshold,
        "sell_threshold": args.sell_threshold,
        "max_width": args.max_width,
    }

    with stage("Evaluating validation forecasts"):
        validation_result = evaluate_forecast_frame(
            validation_forecasts,
            horizon_h=split_config.horizon_h,
            signal_mode=args.signal_mode,
            signal_kwargs=signal_kwargs,
            execution_rule=args.execution_rule,
            transaction_cost=args.transaction_cost,
            slippage=args.slippage,
            price_frame=raw_df,
        )
    with stage("Evaluating final test forecasts"):
        final_test_result = evaluate_forecast_frame(
            final_test_forecasts,
            horizon_h=split_config.horizon_h,
            signal_mode=args.signal_mode,
            signal_kwargs=signal_kwargs,
            execution_rule=args.execution_rule,
            transaction_cost=args.transaction_cost,
            slippage=args.slippage,
            price_frame=raw_df,
        )

    validation_report = validation_result["report"]
    final_report = final_test_result["report"]
    validation_forecast = validation_report["forecast_metrics"]
    final_forecast = final_report["forecast_metrics"]
    with stage("Building per-symbol validation reports"):
        validation_per_symbol_report = _build_per_symbol_reports(
            validation_forecasts,
            horizon_h=split_config.horizon_h,
            signal_mode=args.signal_mode,
            signal_kwargs=signal_kwargs,
            execution_rule=args.execution_rule,
            transaction_cost=args.transaction_cost,
            slippage=args.slippage,
            price_frame=raw_df,
        )
    with stage("Building per-symbol final test reports"):
        final_test_per_symbol_report = _build_per_symbol_reports(
            final_test_forecasts,
            horizon_h=split_config.horizon_h,
            signal_mode=args.signal_mode,
            signal_kwargs=signal_kwargs,
            execution_rule=args.execution_rule,
            transaction_cost=args.transaction_cost,
            slippage=args.slippage,
            price_frame=raw_df,
        )
    validation_per_symbol_ok, validation_per_symbol_failures = _passes_per_symbol_gate(
        validation_per_symbol_report,
        max_rmse=args.approval_max_per_symbol_validation_rmse,
        min_coverage=args.approval_min_per_symbol_validation_coverage,
    )
    final_test_per_symbol_ok, final_test_per_symbol_failures = _passes_per_symbol_gate(
        final_test_per_symbol_report,
        max_rmse=args.approval_max_per_symbol_final_test_rmse,
        min_coverage=args.approval_min_per_symbol_final_test_coverage,
    )
    with stage("Building validation regime reports"):
        validation_regime_reports = build_regime_group_reports(
            validation_forecasts,
            raw_df=raw_df,
            horizon_h=split_config.horizon_h,
            signal_mode=args.signal_mode,
            signal_kwargs=signal_kwargs,
            execution_rule=args.execution_rule,
            transaction_cost=args.transaction_cost,
            slippage=args.slippage,
        )
    with stage("Building final test regime reports"):
        final_test_regime_reports = build_regime_group_reports(
            final_test_forecasts,
            raw_df=raw_df,
            horizon_h=split_config.horizon_h,
            signal_mode=args.signal_mode,
            signal_kwargs=signal_kwargs,
            execution_rule=args.execution_rule,
            transaction_cost=args.transaction_cost,
            slippage=args.slippage,
        )

    approved = (
        validation_forecast["rmse_q50"] <= args.approval_max_validation_rmse
        and final_forecast["rmse_q50"] <= args.approval_max_final_test_rmse
        and validation_forecast["interval_coverage_rate"] >= args.approval_min_validation_coverage
        and final_forecast["interval_coverage_rate"] >= args.approval_min_final_test_coverage
        and validation_per_symbol_ok
        and final_test_per_symbol_ok
    )

    dataset_meta.update(
        {
            "evaluation_status": "approved" if approved else "rejected",
            "evaluation_approved": bool(approved),
            "deployable": bool(approved),
            "evaluation_timestamp": pd.Timestamp.utcnow().isoformat(),
            "evaluation_summary": {
                "evaluation_mode": "saved_candidate_artifact",
                "evaluated_artifact_dir": str(artifact_dir),
                "artifact_validation_report": validation_report,
                "final_test_report": final_report,
                "validation_per_symbol_report": validation_per_symbol_report,
                "final_test_per_symbol_report": final_test_per_symbol_report,
                "validation_symbol_scoreboard": _extract_symbol_scoreboard(validation_per_symbol_report),
                "final_test_symbol_scoreboard": _extract_symbol_scoreboard(final_test_per_symbol_report),
                "validation_regime_reports": validation_regime_reports,
                "final_test_regime_reports": final_test_regime_reports,
                "training_universe_summary": summarize_training_universe(
                    TrainingUniverseBundle(
                        raw_df=raw_df,
                        feature_df=feature_df,
                        symbol_id_map=symbol_id_map,
                        symbols=symbol_filter,
                    )
                ),
                "approval_criteria": {
                    "approval_max_validation_rmse": args.approval_max_validation_rmse,
                    "approval_max_final_test_rmse": args.approval_max_final_test_rmse,
                    "approval_min_validation_coverage": args.approval_min_validation_coverage,
                    "approval_min_final_test_coverage": args.approval_min_final_test_coverage,
                    "approval_max_per_symbol_validation_rmse": args.approval_max_per_symbol_validation_rmse,
                    "approval_max_per_symbol_final_test_rmse": args.approval_max_per_symbol_final_test_rmse,
                    "approval_min_per_symbol_validation_coverage": args.approval_min_per_symbol_validation_coverage,
                    "approval_min_per_symbol_final_test_coverage": args.approval_min_per_symbol_final_test_coverage,
                },
                "approval_gate_results": {
                    "global_validation_gate_passed": validation_forecast["rmse_q50"] <= args.approval_max_validation_rmse
                    and validation_forecast["interval_coverage_rate"] >= args.approval_min_validation_coverage,
                    "global_final_test_gate_passed": final_forecast["rmse_q50"] <= args.approval_max_final_test_rmse
                    and final_forecast["interval_coverage_rate"] >= args.approval_min_final_test_coverage,
                    "per_symbol_validation_gate_passed": validation_per_symbol_ok,
                    "per_symbol_final_test_gate_passed": final_test_per_symbol_ok,
                    "per_symbol_validation_failures": validation_per_symbol_failures,
                    "per_symbol_final_test_failures": final_test_per_symbol_failures,
                },
            },
        }
    )
    validate_artifact_metadata(dataset_meta)

    meta_path = artifact_dir / "dataset_meta.pkl"
    with stage("Saving evaluation report and updated metadata"):
        joblib.dump(dataset_meta, meta_path)
        joblib.dump(dataset_meta["evaluation_summary"], artifact_dir / "evaluation_report.pkl")

    log_step("Artifact validation report:")
    print(validation_report)
    log_step("Final test report:")
    print(final_report)
    log_step(f"Deployment approval: {'APPROVED' if approved else 'REJECTED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
