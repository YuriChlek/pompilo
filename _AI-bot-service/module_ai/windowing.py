from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from module_ai.data_pipeline import ENCODER_FEATURES, TARGET_NAME


@dataclass(frozen=True)
class WindowedSamples:
    X: np.ndarray
    y: np.ndarray
    origin_frame: pd.DataFrame
    feature_names: list[str]


def _ensure_single_symbol(df: pd.DataFrame) -> str:
    if "symbol" not in df.columns:
        raise ValueError("windowing dataframe must contain symbol")
    symbols = df["symbol"].dropna().astype(str).unique().tolist()
    if len(symbols) != 1:
        raise ValueError("windowing operates on one symbol at a time")
    return symbols[0]


def _normalize_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _normalize_origin_frame(
    origin_source: pd.DataFrame | Iterable[pd.Timestamp | str],
) -> pd.DataFrame:
    if isinstance(origin_source, pd.DataFrame):
        if "symbol" not in origin_source.columns or "open_time" not in origin_source.columns:
            raise ValueError("origin dataframe must contain symbol and open_time")
        origin_frame = origin_source.loc[:, ["symbol", "open_time"]].copy()
        origin_frame["symbol"] = origin_frame["symbol"].astype(str).str.strip().str.upper()
        origin_frame["open_time"] = origin_frame["open_time"].map(_normalize_timestamp)
        return origin_frame.reset_index(drop=True)

    normalized_origins = [_normalize_timestamp(ts) for ts in origin_source]
    if not normalized_origins:
        raise ValueError("origin_timestamps is empty")
    return pd.DataFrame({"symbol": [None] * len(normalized_origins), "open_time": normalized_origins})


def build_window_feature_names(encoder_len: int, encoder_features: list[str] | None = None) -> list[str]:
    if encoder_len <= 0:
        raise ValueError("encoder_len must be > 0")
    features = encoder_features or ENCODER_FEATURES
    names: list[str] = []
    for lag in range(encoder_len - 1, -1, -1):
        for feature in features:
            names.append(f"{feature}__t_minus_{lag}")
    return names


def build_window_samples(
    feature_df: pd.DataFrame,
    origin_timestamps: pd.DataFrame | Iterable[pd.Timestamp | str],
    *,
    encoder_len: int,
    encoder_features: list[str] | None = None,
    target_name: str = TARGET_NAME,
) -> WindowedSamples:
    if encoder_len <= 0:
        raise ValueError("encoder_len must be > 0")

    df = feature_df.sort_values(["symbol", "open_time"]).reset_index(drop=True).copy()
    features = encoder_features or ENCODER_FEATURES
    missing = [col for col in features + ["symbol", "open_time", target_name] if col not in df.columns]
    if missing:
        raise ValueError(f"windowing input missing required columns: {missing}")

    origin_frame = _normalize_origin_frame(origin_timestamps)
    if origin_frame["symbol"].isna().all():
        symbol = _ensure_single_symbol(df)
        origin_frame["symbol"] = symbol
    else:
        origin_frame["symbol"] = origin_frame["symbol"].astype(str).str.strip().str.upper()

    origin_index_lookup = {
        (str(symbol).strip().upper(), _normalize_timestamp(open_time)): idx
        for idx, (symbol, open_time) in enumerate(
            df.loc[:, ["symbol", "open_time"]].itertuples(index=False, name=None)
        )
    }

    rows: list[np.ndarray] = []
    targets: list[float] = []
    origin_records: list[pd.Series] = []

    for origin_symbol, origin_time in origin_frame.itertuples(index=False):
        origin_key = (str(origin_symbol).strip().upper(), _normalize_timestamp(origin_time))
        if origin_key not in origin_index_lookup:
            raise ValueError(
                f"origin ({origin_key[0]}, {origin_key[1].isoformat()}) not found in feature dataframe"
            )
        origin_idx = origin_index_lookup[origin_key]
        start_idx = origin_idx - encoder_len + 1
        if start_idx < 0:
            raise ValueError(
                f"insufficient encoder history for origin {origin_key[0]} @ {origin_key[1].isoformat()}"
            )

        window = df.iloc[start_idx:origin_idx + 1].copy()
        if len(window) != encoder_len:
            raise ValueError(f"invalid window length for origin {origin_key[0]} @ {origin_key[1].isoformat()}")
        if window["symbol"].nunique() != 1 or str(window["symbol"].iloc[-1]).strip().upper() != origin_key[0]:
            raise ValueError(
                f"encoder window crosses symbol boundaries for origin {origin_key[0]} @ {origin_key[1].isoformat()}"
            )
        if window[features].isna().any().any():
            raise ValueError(
                f"encoder window contains NaN values for origin {origin_key[0]} @ {origin_key[1].isoformat()}"
            )

        target_value = df.iloc[origin_idx][target_name]
        if pd.isna(target_value):
            raise ValueError(f"target value is missing for origin {origin_key[0]} @ {origin_key[1].isoformat()}")

        rows.append(window[features].to_numpy(dtype=np.float64).reshape(-1))
        targets.append(float(target_value))
        origin_records.append(df.iloc[origin_idx][["symbol", "open_time", "close", target_name]].copy())

    origin_frame = pd.DataFrame(origin_records).reset_index(drop=True)
    origin_frame = origin_frame.rename(columns={"open_time": "origin_timestamp", target_name: "actual_return_h"})
    return WindowedSamples(
        X=np.asarray(rows, dtype=np.float64),
        y=np.asarray(targets, dtype=np.float64),
        origin_frame=origin_frame,
        feature_names=build_window_feature_names(encoder_len, features),
    )


def build_latest_window(
    feature_df: pd.DataFrame,
    *,
    encoder_len: int,
    encoder_features: list[str] | None = None,
) -> tuple[np.ndarray, pd.Series, list[str]]:
    if encoder_len <= 0:
        raise ValueError("encoder_len must be > 0")

    df = feature_df.sort_values(["symbol", "open_time"]).reset_index(drop=True).copy()
    _ensure_single_symbol(df)

    features = encoder_features or ENCODER_FEATURES
    missing = [col for col in features + ["open_time", "close"] if col not in df.columns]
    if missing:
        raise ValueError(f"latest window input missing required columns: {missing}")
    if len(df) < encoder_len:
        raise ValueError(f"need at least {encoder_len} rows for latest window, got {len(df)}")

    window = df.iloc[-encoder_len:].copy()
    if window[features].isna().any().any():
        raise ValueError("latest encoder window contains NaN values")

    vector = window[features].to_numpy(dtype=np.float64).reshape(1, -1)
    latest_row = df.iloc[-1].copy()
    return vector, latest_row, build_window_feature_names(encoder_len, features)


__all__ = [
    "WindowedSamples",
    "build_latest_window",
    "build_window_feature_names",
    "build_window_samples",
]
