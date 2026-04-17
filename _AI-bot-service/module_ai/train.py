from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.metrics import mean_absolute_error

from module_ai.console import log_step, stage
from module_ai.data_access import filter_time_range
from module_ai.data_pipeline import (
    MULTI_SYMBOL_SENTINEL,
    TARGET_NAME,
    build_dataset_metadata,
    save_scaler_artifact,
)
from module_ai.evaluate import evaluate_forecasts
from module_ai.modeling import (
    MODEL_ARTIFACT_NAME,
    MODEL_BACKEND_GRADIENT_BOOSTING,
    SUPPORTED_MODEL_BACKENDS,
    predict_quantiles,
    save_model_bundle,
)
from module_ai.runtime import fit_runtime_bundle, predict_origin_frame
from module_ai.symbols import resolve_cli_symbols
from module_ai.splits import SplitConfig, split_dataset
from module_ai.training_universe import load_training_universe_bundle, summarize_training_universe
from utils.config import TRADING_SYMBOLS


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the canonical ML forecast model.")
    parser.add_argument(
        "--symbol",
        default="ALL",
        help="Trading symbol for single-symbol experiments, or ALL for the full training universe",
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbol list for multi-symbol training. Overrides --symbol when provided",
    )
    parser.add_argument("--save-dir", default="./artifacts", help="Base directory for trained artifacts")
    parser.add_argument("--encoder-len", type=int, default=48)
    parser.add_argument("--pred-len", type=int, default=6, help="Forecast horizon H")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--model-backend",
        default=MODEL_BACKEND_GRADIENT_BOOSTING,
        choices=SUPPORTED_MODEL_BACKENDS,
        help="Model backend for quantile forecasting",
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    return parser


def _resolve_training_symbols(args: argparse.Namespace) -> list[str]:
    return resolve_cli_symbols(
        symbol=getattr(args, "symbol", "ALL"),
        symbols=getattr(args, "symbols", ""),
        default_symbols=TRADING_SYMBOLS,
    )


def _artifact_name_for_symbols(symbols: list[str]) -> str:
    if len(symbols) == 1:
        return symbols[0].lower()
    return MULTI_SYMBOL_SENTINEL


def _build_split_frames(feature_df, split_meta: dict[str, str | None]):
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


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = build_argparser().parse_args()

    training_symbols = _resolve_training_symbols(args)
    artifact_name = _artifact_name_for_symbols(training_symbols)
    log_step(
        f"Training universe: {len(training_symbols)} symbol(s) -> "
        f"{', '.join(training_symbols[:8])}{' ...' if len(training_symbols) > 8 else ''}"
    )
    log_step(
        f"Config: model_backend={args.model_backend}, encoder_len={args.encoder_len}, pred_len={args.pred_len}, "
        f"ratios={args.train_ratio:.2f}/{args.val_ratio:.2f}/{args.test_ratio:.2f}, "
        f"compute_device={args.compute_device}"
    )

    with stage("Loading training universe and building feature dataframe"):
        universe_bundle = load_training_universe_bundle(
            horizon_h=args.pred_len,
            symbols=training_symbols,
        )
    raw_df = universe_bundle.raw_df
    feature_df = universe_bundle.feature_df
    symbol_id_map = universe_bundle.symbol_id_map
    training_summary = summarize_training_universe(universe_bundle)
    save_dir = Path(args.save_dir) / artifact_name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_step(f"Raw rows after cleanup: {len(raw_df)}")
    log_step(f"Feature rows: {len(feature_df)}")

    split_config = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        encoder_len=args.encoder_len,
        horizon_h=args.pred_len,
    )
    with stage("Creating chronological train/validation/test splits"):
        train_origins, val_origins, test_origins, split_meta = split_dataset(feature_df, split_config)
        train_raw, val_raw, test_raw = _build_split_frames(feature_df, split_meta)
    log_step(
        f"Origins: train={len(train_origins)}, val={len(val_origins)}, test={len(test_origins)}"
    )

    with stage("Fitting scalers and training quantile model"):
        runtime_bundle = fit_runtime_bundle(
            feature_df,
            train_raw_df=train_raw,
            train_origin_df=train_origins,
            encoder_len=args.encoder_len,
            model_params={
                "model_backend": args.model_backend,
                "random_state": args.seed,
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "learning_rate": args.learning_rate,
                "subsample": args.subsample,
                "compute_device": args.compute_device,
                "gpu_device_id": args.gpu_device_id,
            },
        )
    scalers = runtime_bundle["scalers"]
    scaler_meta = runtime_bundle["scaler_meta"]
    with stage("Saving scaler artifact"):
        save_scaler_artifact(save_dir / "scalers.pkl", scalers, scaler_meta)

    scaled_feature_df = runtime_bundle["scaled_feature_df"]
    train_samples = runtime_bundle["train_samples"]
    model_bundle = runtime_bundle["model_bundle"]
    log_step(
        "Resolved training device: "
        f"requested={model_bundle.get('compute_device_requested', 'n/a')} "
        f"resolved={model_bundle.get('compute_device_resolved', 'n/a')}"
    )
    with stage("Running validation and test forecasts"):
        val_forecasts, val_samples = predict_origin_frame(
            scaled_feature_df,
            origin_df=val_origins,
            encoder_len=args.encoder_len,
            model_bundle=model_bundle,
        )
        test_forecasts, test_samples = predict_origin_frame(
            scaled_feature_df,
            origin_df=test_origins,
            encoder_len=args.encoder_len,
            model_bundle=model_bundle,
        )

    with stage("Calculating evaluation metrics"):
        forecast_metrics = evaluate_forecasts(val_forecasts)
        holdout_metrics = evaluate_forecasts(test_forecasts)

    with stage("Preparing metadata and model artifacts"):
        dataset_meta = build_dataset_metadata(
            symbol=artifact_name,
            training_symbols=training_symbols,
            symbol_id_map=symbol_id_map,
            horizon_h=args.pred_len,
            encoder_len=args.encoder_len,
            scaler_config=scaler_meta["scaler_config"],
            train_start_time=split_meta["train_start_time"],
            train_end_time=split_meta["train_end_time"],
            validation_start_time=split_meta["val_start_time"],
            validation_end_time=split_meta["val_end_time"],
            test_start_time=split_meta["test_start_time"],
            test_end_time=split_meta["test_end_time"],
        )
    dataset_meta.update(
        {
            "model_type": model_bundle["model_type"],
            "model_backend": model_bundle.get("model_backend"),
            "quantiles": model_bundle["quantiles"],
            "window_feature_names": train_samples.feature_names,
            "train_sample_count": int(len(train_samples.X)),
            "validation_sample_count": int(len(val_samples.X)),
            "test_sample_count": int(len(test_samples.X)),
            "split_config": {
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "test_ratio": args.test_ratio,
                "model_backend": args.model_backend,
                "compute_device": args.compute_device,
                "gpu_device_id": args.gpu_device_id,
                "encoder_len": args.encoder_len,
                "horizon_h": args.pred_len,
            },
            "training_universe_summary": training_summary,
            "validation_forecast_metrics": forecast_metrics,
            "test_forecast_metrics": holdout_metrics,
            "target": TARGET_NAME,
            "candidate_artifact": True,
        }
    )

    with stage("Saving model bundle and metadata"):
        save_model_bundle(
            save_dir / MODEL_ARTIFACT_NAME,
            {
                **model_bundle,
                "encoder_len": args.encoder_len,
                "horizon_h": args.pred_len,
                "window_feature_names": train_samples.feature_names,
                "symbol": artifact_name,
                "training_symbols": training_symbols,
                "symbol_id_map": symbol_id_map,
            },
        )
        joblib.dump(dataset_meta, save_dir / "dataset_meta.pkl")

    with stage("Writing training summary"):
        summary = {
            "train_mae": mean_absolute_error(train_samples.y, predict_quantiles(model_bundle, train_samples.X)[:, 1]),
            "validation_mae": forecast_metrics["mae_q50"],
            "validation_rmse": forecast_metrics["rmse_q50"],
            "test_mae": holdout_metrics["mae_q50"],
            "test_rmse": holdout_metrics["rmse_q50"],
        }
        joblib.dump(summary, save_dir / "training_summary.pkl")

    log_step(f"Saved candidate model artifacts to {save_dir}")
    log_step("Artifact status: pending canonical evaluation approval")
    log_step(f"Validation MAE: {forecast_metrics['mae_q50']:.6f}")
    log_step(f"Validation RMSE: {forecast_metrics['rmse_q50']:.6f}")
    log_step(f"Test MAE: {holdout_metrics['mae_q50']:.6f}")
    log_step(f"Test RMSE: {holdout_metrics['rmse_q50']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
