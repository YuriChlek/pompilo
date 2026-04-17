from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from module_ai.artifacts import default_artifact_dir
from module_ai.console import log_step, stage
from module_ai.data_access import (
    drop_incomplete_latest_candles,
    keep_latest_regular_time_block,
    normalize_symbol,
    read_candles_postgres,
)
from module_ai.data_pipeline import (
    MULTI_SYMBOL_SENTINEL,
    add_symbol_id_feature,
    add_technical_features,
    add_time_features,
    load_scaler_artifact,
    normalize_candle_schema,
    transform_features,
    validate_artifact_metadata,
    validate_scaler_metadata_against_dataset_metadata,
)
from module_ai.modeling import (
    ForecastResult,
    MODEL_ARTIFACT_NAME,
    load_model_bundle,
    predict_quantiles,
    validate_model_bundle_against_dataset_metadata,
)
from module_ai.windowing import build_latest_window
from module_ai.symbols import normalize_training_symbols


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a canonical ML forecast.")
    parser.add_argument("--model-dir", default="", help="Directory with trained artifacts for one symbol")
    parser.add_argument("--symbol", required=True, help="Trading symbol")
    parser.add_argument("--limit", type=int, default=256, help="Number of recent candles to load")
    return parser


def _build_inference_features(raw_df: pd.DataFrame, *, symbol_id_map: dict[str, int]) -> pd.DataFrame:
    out = normalize_candle_schema(raw_df)
    out = add_symbol_id_feature(out, symbol_id_map=symbol_id_map)
    out = add_time_features(out)
    out = add_technical_features(out)
    return out


def get_return_forecast(
    *,
    model_dir: str | Path = "",
    symbol: str,
    limit: int = 256,
) -> ForecastResult:
    normalized_symbol = normalize_symbol(symbol)
    artifact_dir = Path(model_dir) if model_dir else default_artifact_dir(symbol)
    if not artifact_dir.exists():
        raise ValueError(f"model_dir does not exist: {artifact_dir}")

    meta_path = artifact_dir / "dataset_meta.pkl"
    scalers_path = artifact_dir / "scalers.pkl"
    model_path = artifact_dir / MODEL_ARTIFACT_NAME
    if not meta_path.exists() or not scalers_path.exists() or not model_path.exists():
        raise ValueError(f"missing forecast artifacts in {artifact_dir}")

    dataset_meta = joblib.load(meta_path)
    validate_artifact_metadata(dataset_meta)
    scalers, scaler_meta = load_scaler_artifact(scalers_path)
    validate_scaler_metadata_against_dataset_metadata(scaler_meta, dataset_meta)
    model_bundle = load_model_bundle(model_path)
    validate_model_bundle_against_dataset_metadata(model_bundle, dataset_meta)
    scaled_features = dataset_meta["scaled_features"]
    encoder_len = int(dataset_meta["encoder_len"])
    horizon_h = int(dataset_meta["horizon_h"])
    symbol_id_map = {
        str(key).strip().upper(): int(value)
        for key, value in dict(dataset_meta.get("symbol_id_map", {})).items()
    }
    training_symbols = normalize_training_symbols(list(dataset_meta.get("training_symbols", [])))
    artifact_symbol = str(dataset_meta.get("symbol", "")).lower().replace("/", "").replace("-", "")
    request_symbol = normalized_symbol
    if artifact_symbol and artifact_symbol != MULTI_SYMBOL_SENTINEL and artifact_symbol != request_symbol:
        raise ValueError("artifact symbol does not match requested symbol")
    if training_symbols and symbol.upper() not in training_symbols:
        raise ValueError(f"requested symbol {symbol.upper()} is not present in the artifact training universe")
    if not dataset_meta.get("evaluation_approved", False) or not dataset_meta.get("deployable", False):
        raise ValueError(
            "artifact is not deployment-approved. "
            "Run the canonical evaluation gate via module_ai/evaluate_run.py first."
        )

    raw_df = read_candles_postgres(symbol, limit=limit, order="ASC")
    raw_df = drop_incomplete_latest_candles(raw_df)
    raw_df = keep_latest_regular_time_block(raw_df)
    if raw_df.empty:
        raise ValueError("no closed candles available for inference")

    feature_df = _build_inference_features(raw_df, symbol_id_map=symbol_id_map)
    scaled_feature_df = transform_features(feature_df, scalers, scaled_features)
    X_latest, latest_row, feature_names = build_latest_window(
        scaled_feature_df,
        encoder_len=encoder_len,
    )

    artifact_feature_names = model_bundle.get("window_feature_names")
    if artifact_feature_names and artifact_feature_names != feature_names:
        raise ValueError("window feature names do not match the trained artifact")

    predicted_quantiles = predict_quantiles(model_bundle, X_latest)[0]
    last_real_close = float(raw_df.iloc[-1]["close"])

    return ForecastResult(
        symbol=str(latest_row["symbol"]),
        origin_timestamp=pd.Timestamp(latest_row["open_time"]).isoformat(),
        horizon_h=horizon_h,
        q10_target_return_h=float(predicted_quantiles[0]),
        q50_target_return_h=float(predicted_quantiles[1]),
        q90_target_return_h=float(predicted_quantiles[2]),
        last_real_close=last_real_close,
        projected_price_h_q10=last_real_close * (1.0 + float(predicted_quantiles[0])),
        projected_price_h_q50=last_real_close * (1.0 + float(predicted_quantiles[1])),
        projected_price_h_q90=last_real_close * (1.0 + float(predicted_quantiles[2])),
    )


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = build_argparser().parse_args()
    log_step(f"Forecast request: symbol={args.symbol.upper()} limit={int(args.limit)}")
    with stage("Running forecast pipeline"):
        forecast = get_return_forecast(
            model_dir=args.model_dir,
            symbol=args.symbol,
            limit=args.limit,
        )
    print(forecast.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
