from __future__ import annotations

import pandas as pd

from module_ai.data_pipeline import fit_feature_scalers, get_scaled_feature_list, transform_features
from module_ai.modeling import predict_quantiles, train_quantile_models
from module_ai.windowing import WindowedSamples, build_window_samples


def fit_runtime_bundle(
    feature_df: pd.DataFrame,
    *,
    train_raw_df: pd.DataFrame,
    train_origin_df: pd.DataFrame,
    encoder_len: int,
    model_params: dict | None = None,
) -> dict:
    scaled_features = get_scaled_feature_list()
    scalers, scaler_meta = fit_feature_scalers(train_raw_df, scaled_features)
    scaled_feature_df = transform_features(feature_df, scalers, scaled_features)
    train_samples = build_window_samples(
        scaled_feature_df,
        train_origin_df.loc[:, ["symbol", "open_time"]],
        encoder_len=encoder_len,
    )
    model_bundle = train_quantile_models(
        train_samples.X,
        train_samples.y,
        **(model_params or {}),
    )
    return {
        "scalers": scalers,
        "scaler_meta": scaler_meta,
        "scaled_feature_df": scaled_feature_df,
        "train_samples": train_samples,
        "model_bundle": model_bundle,
    }


def predict_origin_frame(
    scaled_feature_df: pd.DataFrame,
    *,
    origin_df: pd.DataFrame,
    encoder_len: int,
    model_bundle: dict,
) -> tuple[pd.DataFrame, WindowedSamples]:
    samples = build_window_samples(
        scaled_feature_df,
        origin_df.loc[:, ["symbol", "open_time"]],
        encoder_len=encoder_len,
    )
    predicted_quantiles = predict_quantiles(model_bundle, samples.X)
    forecast_df = samples.origin_frame.copy()
    forecast_df["q10_return_h"] = predicted_quantiles[:, 0]
    forecast_df["q50_return_h"] = predicted_quantiles[:, 1]
    forecast_df["q90_return_h"] = predicted_quantiles[:, 2]
    return forecast_df, samples


__all__ = [
    "fit_runtime_bundle",
    "predict_origin_frame",
]
