from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


DATASET_CONTRACT_VERSION = "3.0.0"
FEATURE_VERSION = "3.0.0"
EVALUATION_CONTRACT_VERSION = "1.0.0"
TARGET_NAME = "target_return_h"
SUPPORTED_INFERENCE_ENTRYPOINT = "module_ai.forecast:get_return_forecast"
MULTI_SYMBOL_SENTINEL = "__multi__"

NUMERIC_CANDLE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "cvd",
]

SYMBOL_FEATURES = [
    "symbol_id",
]

TIME_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]

RELATIVE_PRICE_FEATURES = [
    "return_1",
    "return_3",
    "return_6",
    "hl_range_ratio",
    "oc_change_ratio",
    "close_vs_ma_20",
    "close_vs_ma_50",
]

VOLUME_FEATURES = [
    "volume_vs_ma_20",
    "volume_zscore_20",
]

FLOW_FEATURES = [
    "cvd_change_1",
    "cvd_change_6",
    "cvd_zscore_20",
]

TECHNICAL_FEATURES = [
    "rsi",
    "bb_position",
    "bb_width_ratio",
    "volatility_24",
]

ENCODER_FEATURES = (
    SYMBOL_FEATURES
    + TIME_FEATURES
    + RELATIVE_PRICE_FEATURES
    + VOLUME_FEATURES
    + FLOW_FEATURES
    + TECHNICAL_FEATURES
)

DECODER_KNOWN_FEATURES = [
    "time_idx",
    "symbol",
    "symbol_id",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]

DEPRECATED_FEATURES = [
    "future_high_ratio",
    "future_low_ratio",
    "future_volatility",
    "direction_target",
    "week_of_year",
]

WARMUP_SENSITIVE_FEATURES = [
    "rsi",
    "bb_position",
    "bb_width_ratio",
    "return_1",
    "return_3",
    "return_6",
    "close_vs_ma_20",
    "close_vs_ma_50",
    "volume_vs_ma_20",
    "volume_zscore_20",
    "cvd_change_1",
    "cvd_change_6",
    "cvd_zscore_20",
    "volatility_24",
]

BASE_REQUIRED_COLUMNS = [
    "symbol",
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

OPTIONAL_SUPPORTED_COLUMNS = [
    "cvd",
]


@dataclass(frozen=True)
class ScalerArtifact:
    scalers: dict[str, Any]
    metadata: dict[str, Any]


def _ensure_columns_exist(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _ensure_no_duplicate_feature_names(columns: list[str]) -> None:
    duplicates = [col for col in columns if columns.count(col) > 1]
    if duplicates:
        raise ValueError(f"duplicate feature names detected: {sorted(set(duplicates))}")


def _assert_no_forbidden_active_fields(columns: list[str] | pd.Index, label: str) -> None:
    forbidden = [col for col in columns if col in DEPRECATED_FEATURES]
    if forbidden:
        raise ValueError(f"{label} contains forbidden active ML fields: {forbidden}")


def _normalize_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized = df.copy()
    for col in columns:
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
    return normalized


def _validate_symbol_timestamps(df: pd.DataFrame) -> None:
    if not pd.api.types.is_datetime64_any_dtype(df["open_time"]):
        sample_values = df["open_time"].head(5).tolist()
        raise ValueError(
            f"open_time must be datetime-like; dtype={df['open_time'].dtype}; sample_values={sample_values}"
        )

    duplicated = df.duplicated(subset=["symbol", "open_time"])
    if duplicated.any():
        raise ValueError("duplicate timestamps detected for the same symbol")


def _sort_canonical(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["symbol", "open_time"]).reset_index(drop=True)


def _extract_symbol_universe(df: pd.DataFrame) -> list[str]:
    _ensure_columns_exist(df, ["symbol"], "dataframe")
    symbols = df["symbol"].dropna().astype(str).str.strip().str.upper().unique().tolist()
    if not symbols:
        raise ValueError("symbol column contains no non-null values")
    return sorted(symbols)


def _extract_single_symbol(df: pd.DataFrame) -> str:
    symbols = _extract_symbol_universe(df)
    if len(symbols) != 1:
        raise ValueError("scaler artifacts must be symbol-specific; expected exactly one symbol")
    return symbols[0]


def _iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _normalize_symbol_list(symbols: list[str]) -> list[str]:
    normalized = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    deduplicated = sorted(set(normalized))
    if not deduplicated:
        raise ValueError("training_symbols must contain at least one non-empty symbol")
    return deduplicated


def build_symbol_id_map(symbols: list[str]) -> dict[str, int]:
    normalized_symbols = _normalize_symbol_list(symbols)
    return {symbol: idx for idx, symbol in enumerate(normalized_symbols)}


def normalize_candle_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize the raw candle schema for active ML usage.

    Passive legacy columns may exist in raw input, but they are removed from the
    returned dataframe and must never participate in active ML paths.
    """
    if df.empty:
        raise ValueError("input dataframe is empty")

    _ensure_columns_exist(df, BASE_REQUIRED_COLUMNS, "input dataframe")

    normalized = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(normalized["open_time"]):
        normalized["open_time"] = pd.to_datetime(normalized["open_time"], utc=True, errors="coerce")
    if normalized["open_time"].isna().any():
        raise ValueError("open_time contains invalid timestamps")

    normalized["symbol"] = normalized["symbol"].astype("string").str.strip().str.upper()
    if normalized["symbol"].isna().any():
        raise ValueError("symbol column contains null values")
    if (normalized["symbol"] == "").any():
        raise ValueError("symbol column contains empty values")

    for optional_column in OPTIONAL_SUPPORTED_COLUMNS:
        if optional_column not in normalized.columns:
            normalized[optional_column] = 0.0

    numeric_cols = [col for col in NUMERIC_CANDLE_COLUMNS if col in normalized.columns]
    normalized = _normalize_numeric_columns(normalized, numeric_cols)

    _validate_symbol_timestamps(normalized)
    normalized = _sort_canonical(normalized)
    _assert_no_forbidden_active_fields(normalized.columns, "normalized candle schema")
    return normalized


def add_symbol_id_feature(
    df: pd.DataFrame,
    symbol_id_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    normalized = normalize_candle_schema(df)
    out = normalized.copy()
    resolved_symbol_id_map = symbol_id_map or build_symbol_id_map(_extract_symbol_universe(out))
    missing_symbols = sorted(set(_extract_symbol_universe(out)) - set(resolved_symbol_id_map))
    if missing_symbols:
        raise ValueError(f"symbol_id_map is missing symbols: {missing_symbols}")
    out["symbol_id"] = (
        out["symbol"]
        .astype("string")
        .str.strip()
        .str.upper()
        .map(resolved_symbol_id_map)
        .astype("Int64")
    )
    if out["symbol_id"].isna().any():
        raise ValueError("symbol_id contains null values after encoding")
    out["symbol_id"] = out["symbol_id"].astype(np.int32)
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the canonical deterministic time features.
    """
    normalized = normalize_candle_schema(df)
    out = normalized.copy()

    out["time_idx"] = out.groupby("symbol").cumcount()
    hour = out["open_time"].dt.hour.astype(np.int16)
    dow = out["open_time"].dt.dayofweek.astype(np.int16)
    month = out["open_time"].dt.month.astype(np.int16)

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)

    return out


def _rolling_zscore(series: pd.Series, window: int, *, min_periods: int | None = None) -> pd.Series:
    min_periods = min_periods or window
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()
    zscore = (series - rolling_mean) / rolling_std
    zscore = zscore.replace([np.inf, -np.inf], np.nan)
    return zscore


def _add_grouped_technical_features(group: pd.DataFrame) -> pd.DataFrame:
    out = group.copy()
    if "symbol" not in out.columns:
        out["symbol"] = getattr(group, "name", None)

    close = out["close"].astype(float)
    open_price = out["open"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    volume = out["volume"].astype(float)
    cvd = out["cvd"].astype(float)

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    out["rsi"] = 100 - (100 / (1 + rs))

    rolling_mean = close.rolling(window=20, min_periods=20).mean()
    rolling_std = close.rolling(window=20, min_periods=20).std()
    out["bb_upper"] = rolling_mean + (rolling_std * 2)
    out["bb_lower"] = rolling_mean - (rolling_std * 2)

    band_width = out["bb_upper"] - out["bb_lower"]
    out["bb_position"] = (close - out["bb_lower"]) / band_width
    out.loc[band_width == 0, "bb_position"] = np.nan
    out["bb_width_ratio"] = band_width / rolling_mean
    out.loc[rolling_mean == 0, "bb_width_ratio"] = np.nan

    out["return_1"] = close.pct_change(1)
    out["return_3"] = close.pct_change(3)
    out["return_6"] = close.pct_change(6)
    out["hl_range_ratio"] = (high - low) / close
    out["oc_change_ratio"] = (close - open_price) / open_price
    out["close_vs_ma_20"] = close / rolling_mean - 1.0

    ma_50 = close.rolling(window=50, min_periods=50).mean()
    out["close_vs_ma_50"] = close / ma_50 - 1.0

    volume_ma = volume.rolling(window=20, min_periods=20).mean()
    out["volume_vs_ma_20"] = volume / volume_ma - 1.0
    out["volume_zscore_20"] = _rolling_zscore(volume, 20)

    out["cvd_change_1"] = cvd.diff(1)
    out["cvd_change_6"] = cvd.diff(6)
    out["cvd_zscore_20"] = _rolling_zscore(cvd, 20)
    out["volatility_24"] = out["return_1"].rolling(window=24, min_periods=24).std()

    ratio_columns = [
        "hl_range_ratio",
        "oc_change_ratio",
        "close_vs_ma_20",
        "close_vs_ma_50",
        "volume_vs_ma_20",
        "bb_width_ratio",
    ]
    out[ratio_columns] = out[ratio_columns].replace([np.inf, -np.inf], np.nan)

    return out


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the canonical technical features.

    Warmup NaNs are preserved and never zero-filled.
    """
    required = ["symbol", "close", "volume"]
    _ensure_columns_exist(df, required, "technical feature input")
    _assert_no_forbidden_active_fields(df.columns, "technical feature input")

    normalized = _sort_canonical(df)
    result = (
        normalized.groupby("symbol", group_keys=False, sort=False)
        .apply(_add_grouped_technical_features)
        .reset_index(drop=True)
    )
    return result


def add_target_return_h(df: pd.DataFrame, horizon_h: int) -> pd.DataFrame:
    """
    Add the single scalar target column target_return_h.
    """
    if horizon_h <= 0:
        raise ValueError("horizon_h must be > 0")

    _ensure_columns_exist(df, ["symbol", "close"], "target generation input")
    _assert_no_forbidden_active_fields(df.columns, "target generation input")

    out = _sort_canonical(df)
    future_close = out.groupby("symbol")["close"].shift(-horizon_h)
    out[TARGET_NAME] = future_close / out["close"] - 1
    return out


def build_feature_dataframe(
    df: pd.DataFrame,
    horizon_h: int,
    *,
    symbol_id_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Build the canonical active-contract feature dataframe.
    """
    out = normalize_candle_schema(df)
    out = add_symbol_id_feature(out, symbol_id_map=symbol_id_map)
    out = add_time_features(out)
    out = add_technical_features(out)
    out = add_target_return_h(out, horizon_h)
    validate_feature_schema(out)
    return out


def get_required_encoder_features() -> list[str]:
    return ENCODER_FEATURES.copy()


def get_required_decoder_features() -> list[str]:
    return DECODER_KNOWN_FEATURES.copy()


def get_scaled_feature_list() -> list[str]:
    scaled = [feature for feature in ENCODER_FEATURES if feature != "symbol_id"]
    _assert_no_forbidden_active_fields(scaled, "scaled feature list")
    if TARGET_NAME in scaled:
        raise ValueError(f"{TARGET_NAME} must not appear in scaled feature list")
    return scaled


def get_unscaled_feature_list() -> list[str]:
    return [TARGET_NAME, "symbol_id"].copy()


def validate_feature_schema(df: pd.DataFrame) -> None:
    """
    Validate that a dataframe conforms to the active ML schema.
    """
    if df.empty:
        raise ValueError("feature dataframe is empty")

    _ensure_columns_exist(df, BASE_REQUIRED_COLUMNS, "feature dataframe")
    _ensure_columns_exist(df, ENCODER_FEATURES, "feature dataframe")
    _ensure_columns_exist(df, ["time_idx", TARGET_NAME], "feature dataframe")

    _assert_no_forbidden_active_fields(df.columns, "feature dataframe")
    _ensure_no_duplicate_feature_names(get_required_encoder_features())
    _ensure_no_duplicate_feature_names(get_required_decoder_features())


def get_invalid_origin_mask(df: pd.DataFrame) -> pd.Series:
    """
    Return a boolean mask marking invalid prediction origins.
    """
    validate_feature_schema(df)
    required_cols = get_required_encoder_features() + [TARGET_NAME]
    invalid = df[required_cols].isna().any(axis=1)
    invalid = invalid | ~np.isfinite(df[TARGET_NAME].astype(float))
    return invalid


def _build_scaler(kind: str) -> Any:
    if kind == "minmax":
        return MinMaxScaler()
    if kind == "standard":
        return StandardScaler()
    raise ValueError(f"unsupported scaler kind: {kind}")


def _normalize_scaler_config(
    scaler_config: dict[str, Any] | None,
    feature_list: list[str],
) -> dict[str, Any]:
    config = {"kind": "minmax"}
    if scaler_config:
        config.update(scaler_config)

    per_feature = config.get("per_feature", {})
    normalized_per_feature = {}
    for feature in feature_list:
        normalized_per_feature[feature] = per_feature.get(feature, config["kind"])
    config["per_feature"] = normalized_per_feature
    return config


def _validate_scaler_feature_list(feature_list: list[str]) -> None:
    _ensure_no_duplicate_feature_names(feature_list)
    _assert_no_forbidden_active_fields(feature_list, "scaler feature list")
    if TARGET_NAME in feature_list:
        raise ValueError(f"{TARGET_NAME} must not be scaled")


def validate_scaler_metadata(metadata: dict[str, Any]) -> None:
    required_metadata = [
        "symbol",
        "training_symbols",
        "feature_version",
        "scaled_features",
        "dataset_contract_version",
        "scaler_config",
        "scaling_scope",
    ]
    missing = [key for key in required_metadata if key not in metadata]
    if missing:
        raise ValueError(f"scaler metadata missing required keys: {missing}")

    if not metadata["symbol"]:
        raise ValueError("scaler metadata symbol must be non-empty")
    training_symbols = _normalize_symbol_list(list(metadata["training_symbols"]))
    _validate_scaler_feature_list(list(metadata["scaled_features"]))
    if metadata["feature_version"] != FEATURE_VERSION:
        raise ValueError("feature_version in scaler metadata does not match active contract")
    if metadata["dataset_contract_version"] != DATASET_CONTRACT_VERSION:
        raise ValueError("dataset_contract_version in scaler metadata does not match active contract")
    if not isinstance(metadata["scaler_config"], dict):
        raise ValueError("scaler_config in scaler metadata must be a dict")
    if metadata["scaling_scope"] not in {"single_symbol", "per_symbol"}:
        raise ValueError("scaling_scope in scaler metadata must be 'single_symbol' or 'per_symbol'")
    if metadata["scaling_scope"] == "single_symbol" and len(training_symbols) != 1:
        raise ValueError("single_symbol scaling requires exactly one training symbol")


def fit_feature_scalers(
    df: pd.DataFrame,
    feature_list: list[str],
    scaler_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Fit per-feature scalers on train-only data.
    """
    _validate_scaler_feature_list(feature_list)
    _ensure_columns_exist(df, feature_list + ["symbol"], "scaler fit input")

    invalid_mask = df[feature_list].isna().any(axis=1)
    fit_df = df.loc[~invalid_mask, ["symbol"] + feature_list].copy()
    if fit_df.empty:
        raise ValueError("no valid rows available for scaler fitting")

    normalized_config = _normalize_scaler_config(scaler_config, feature_list)
    training_symbols = _extract_symbol_universe(df)
    scalers: dict[str, Any]
    if len(training_symbols) == 1:
        scalers = {}
        for feature in feature_list:
            scaler_kind = normalized_config["per_feature"][feature]
            scaler = _build_scaler(scaler_kind)
            scaler.fit(fit_df[[feature]])
            scalers[feature] = scaler
        scaling_scope = "single_symbol"
        primary_symbol = training_symbols[0]
    else:
        scalers = {}
        for symbol, symbol_df in fit_df.groupby("symbol", sort=False):
            symbol_local = str(symbol).strip().upper()
            symbol_scalers: dict[str, Any] = {}
            for feature in feature_list:
                scaler_kind = normalized_config["per_feature"][feature]
                scaler = _build_scaler(scaler_kind)
                scaler.fit(symbol_df[[feature]])
                symbol_scalers[feature] = scaler
            scalers[symbol_local] = symbol_scalers
        scaling_scope = "per_symbol"
        primary_symbol = MULTI_SYMBOL_SENTINEL

    metadata = {
        "symbol": primary_symbol,
        "training_symbols": training_symbols,
        "dataset_contract_version": DATASET_CONTRACT_VERSION,
        "feature_version": FEATURE_VERSION,
        "scaled_features": feature_list.copy(),
        "scaler_config": normalized_config,
        "scaling_scope": scaling_scope,
    }
    return scalers, metadata


def transform_features(
    df: pd.DataFrame,
    fitted_scalers: dict[str, Any],
    feature_list: list[str],
) -> pd.DataFrame:
    """
    Transform only the listed features and preserve all other columns.
    """
    _validate_scaler_feature_list(feature_list)
    _ensure_columns_exist(df, feature_list + ["symbol"], "scaler transform input")

    transformed = df.copy()
    is_per_symbol_scalers = bool(fitted_scalers) and all(
        isinstance(scaler_bundle, dict) for scaler_bundle in fitted_scalers.values()
    )

    if is_per_symbol_scalers:
        available_symbols = {str(symbol).strip().upper() for symbol in fitted_scalers}
        frame_symbols = {
            str(symbol).strip().upper()
            for symbol in transformed["symbol"].dropna().astype(str).unique().tolist()
        }
        missing_symbols = sorted(frame_symbols - available_symbols)
        if missing_symbols:
            raise ValueError(f"missing per-symbol scalers for symbols: {missing_symbols}")

        for symbol, symbol_df in transformed.groupby("symbol", sort=False):
            symbol_key = str(symbol).strip().upper()
            symbol_scalers = fitted_scalers[symbol_key]
            for feature in feature_list:
                if feature not in symbol_scalers:
                    raise ValueError(f"missing fitted scaler for feature: {feature} in symbol {symbol_key}")
                non_null_mask = symbol_df[feature].notna()
                if not non_null_mask.any():
                    continue
                local_index = symbol_df.index[non_null_mask]
                scaled_values = symbol_scalers[feature].transform(transformed.loc[local_index, [feature]])
                transformed.loc[local_index, feature] = scaled_values.reshape(-1)
    else:
        for feature in feature_list:
            if feature not in fitted_scalers:
                raise ValueError(f"missing fitted scaler for feature: {feature}")

            non_null_mask = transformed[feature].notna()
            if not non_null_mask.any():
                continue

            scaled_values = fitted_scalers[feature].transform(transformed.loc[non_null_mask, [feature]])
            transformed.loc[non_null_mask, feature] = scaled_values.reshape(-1)

    return transformed


def save_scaler_artifact(path: str | Path, fitted_scalers: dict[str, Any], metadata: dict[str, Any]) -> None:
    validate_scaler_metadata(metadata)

    artifact = ScalerArtifact(scalers=fitted_scalers, metadata=metadata)
    joblib.dump({"scalers": artifact.scalers, "metadata": artifact.metadata}, Path(path))


def load_scaler_artifact(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    loaded = joblib.load(Path(path))
    if not isinstance(loaded, dict):
        raise ValueError("invalid scaler artifact format")
    if "scalers" not in loaded or "metadata" not in loaded:
        raise ValueError("scaler artifact must contain scalers and metadata")

    scalers = loaded["scalers"]
    metadata = loaded["metadata"]
    validate_scaler_metadata(metadata)

    scaled_features = metadata.get("scaled_features", [])
    if metadata.get("scaling_scope") == "per_symbol":
        training_symbols = metadata.get("training_symbols", [])
        for symbol in training_symbols:
            if symbol not in scalers:
                raise ValueError(f"scaler artifact missing scaler bundle for symbol: {symbol}")
            for feature in scaled_features:
                if feature not in scalers[symbol]:
                    raise ValueError(f"scaler artifact missing scaler for feature: {feature} in symbol {symbol}")
    else:
        for feature in scaled_features:
            if feature not in scalers:
                raise ValueError(f"scaler artifact missing scaler for feature: {feature}")

    return scalers, metadata


def validate_scaler_metadata_against_dataset_metadata(
    scaler_metadata: dict[str, Any],
    dataset_metadata: dict[str, Any],
) -> None:
    required_scaler_keys = [
        "symbol",
        "training_symbols",
        "feature_version",
        "dataset_contract_version",
        "scaled_features",
        "scaling_scope",
    ]
    missing = [key for key in required_scaler_keys if key not in scaler_metadata]
    if missing:
        raise ValueError(f"scaler metadata missing required keys: {missing}")

    scaler_symbol = str(scaler_metadata["symbol"]).strip()
    dataset_symbol = str(dataset_metadata["symbol"]).strip()
    if scaler_symbol != MULTI_SYMBOL_SENTINEL:
        scaler_symbol = scaler_symbol.upper()
    if dataset_symbol != MULTI_SYMBOL_SENTINEL:
        dataset_symbol = dataset_symbol.upper()
    if scaler_symbol != dataset_symbol:
        raise ValueError("scaler metadata symbol does not match dataset metadata symbol")
    if list(scaler_metadata["training_symbols"]) != list(dataset_metadata["training_symbols"]):
        raise ValueError("scaler metadata training_symbols do not match dataset metadata")
    if scaler_metadata["feature_version"] != dataset_metadata["feature_version"]:
        raise ValueError("scaler metadata feature_version does not match dataset metadata")
    if scaler_metadata["dataset_contract_version"] != dataset_metadata["dataset_contract_version"]:
        raise ValueError("scaler metadata dataset_contract_version does not match dataset metadata")
    if list(scaler_metadata["scaled_features"]) != list(dataset_metadata["scaled_features"]):
        raise ValueError("scaler metadata scaled_features do not match dataset metadata")
    if scaler_metadata["scaling_scope"] != dataset_metadata["scaling_scope"]:
        raise ValueError("scaler metadata scaling_scope does not match dataset metadata")


def build_dataset_metadata(
    *,
    symbol: str,
    training_symbols: list[str] | None = None,
    symbol_id_map: dict[str, int] | None = None,
    horizon_h: int,
    encoder_len: int,
    scaler_config: dict[str, Any] | None,
    train_start_time: Any,
    train_end_time: Any,
    validation_start_time: Any,
    validation_end_time: Any,
    test_start_time: Any = None,
    test_end_time: Any = None,
) -> dict[str, Any]:
    if horizon_h <= 0:
        raise ValueError("horizon_h must be > 0")
    if encoder_len <= 0:
        raise ValueError("encoder_len must be > 0")

    resolved_training_symbols = _normalize_symbol_list(training_symbols or [symbol])
    resolved_symbol = str(symbol).strip()
    if not resolved_symbol:
        raise ValueError("symbol is required for dataset metadata")
    if resolved_symbol != MULTI_SYMBOL_SENTINEL:
        resolved_symbol = resolved_symbol.upper()
    resolved_symbol_id_map = symbol_id_map or build_symbol_id_map(resolved_training_symbols)
    if set(resolved_symbol_id_map) != set(resolved_training_symbols):
        raise ValueError("symbol_id_map keys must exactly match training_symbols")
    scaling_scope = "single_symbol" if len(resolved_training_symbols) == 1 else "per_symbol"

    metadata = {
        "symbol": resolved_symbol,
        "training_symbols": resolved_training_symbols,
        "symbol_count": len(resolved_training_symbols),
        "symbol_id_map": resolved_symbol_id_map,
        "scaling_scope": scaling_scope,
        "target_name": TARGET_NAME,
        "horizon_h": horizon_h,
        "encoder_len": encoder_len,
        "dataset_contract_version": DATASET_CONTRACT_VERSION,
        "feature_version": FEATURE_VERSION,
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "supported_inference_entrypoint": SUPPORTED_INFERENCE_ENTRYPOINT,
        "encoder_features": get_required_encoder_features(),
        "decoder_features": get_required_decoder_features(),
        "scaled_features": get_scaled_feature_list(),
        "unscaled_features": get_unscaled_feature_list(),
        "scaler_config": scaler_config or {"kind": "minmax"},
        "evaluation_status": "pending",
        "evaluation_approved": False,
        "deployable": False,
        "evaluation_summary": None,
        "train_start_time": _iso_or_none(train_start_time),
        "train_end_time": _iso_or_none(train_end_time),
        "validation_start_time": _iso_or_none(validation_start_time),
        "validation_end_time": _iso_or_none(validation_end_time),
        "test_start_time": _iso_or_none(test_start_time),
        "test_end_time": _iso_or_none(test_end_time),
    }
    validate_artifact_metadata(metadata)
    return metadata


def validate_artifact_metadata(metadata: dict[str, Any]) -> None:
    required_keys = [
        "symbol",
        "training_symbols",
        "symbol_count",
        "symbol_id_map",
        "scaling_scope",
        "target_name",
        "horizon_h",
        "encoder_len",
        "dataset_contract_version",
        "feature_version",
        "evaluation_contract_version",
        "supported_inference_entrypoint",
        "encoder_features",
        "decoder_features",
        "scaled_features",
        "unscaled_features",
        "scaler_config",
        "evaluation_status",
        "evaluation_approved",
        "deployable",
    ]
    missing = [key for key in required_keys if key not in metadata]
    if missing:
        raise ValueError(f"artifact metadata missing required keys: {missing}")

    if metadata["target_name"] != TARGET_NAME:
        raise ValueError(f"target_name must equal {TARGET_NAME}")
    if metadata["feature_version"] != FEATURE_VERSION:
        raise ValueError("artifact metadata feature_version does not match active contract")
    if metadata["dataset_contract_version"] != DATASET_CONTRACT_VERSION:
        raise ValueError("artifact metadata dataset_contract_version does not match active contract")
    if metadata["evaluation_contract_version"] != EVALUATION_CONTRACT_VERSION:
        raise ValueError("artifact metadata evaluation_contract_version does not match active contract")
    if metadata["supported_inference_entrypoint"] != SUPPORTED_INFERENCE_ENTRYPOINT:
        raise ValueError("artifact metadata supported_inference_entrypoint does not match active contract")
    if not isinstance(metadata["evaluation_approved"], bool):
        raise ValueError("evaluation_approved must be a bool")
    if not isinstance(metadata["deployable"], bool):
        raise ValueError("deployable must be a bool")
    if metadata["deployable"] and not metadata["evaluation_approved"]:
        raise ValueError("deployable artifacts must be evaluation-approved")
    if metadata.get("evaluation_summary") is not None and not isinstance(metadata["evaluation_summary"], dict):
        raise ValueError("evaluation_summary must be a dict or None")
    training_symbols = _normalize_symbol_list(list(metadata["training_symbols"]))
    if int(metadata["symbol_count"]) != len(training_symbols):
        raise ValueError("symbol_count must match training_symbols length")
    if metadata["scaling_scope"] not in {"single_symbol", "per_symbol"}:
        raise ValueError("scaling_scope must be 'single_symbol' or 'per_symbol'")
    if not isinstance(metadata["symbol_id_map"], dict):
        raise ValueError("symbol_id_map must be a dict")
    normalized_symbol_id_map = {
        str(symbol).strip().upper(): int(symbol_id)
        for symbol, symbol_id in metadata["symbol_id_map"].items()
    }
    if set(normalized_symbol_id_map) != set(training_symbols):
        raise ValueError("symbol_id_map keys must exactly match training_symbols")
    if sorted(normalized_symbol_id_map.values()) != list(range(len(training_symbols))):
        raise ValueError("symbol_id_map values must be a contiguous range starting at 0")
    if metadata["scaling_scope"] == "single_symbol" and len(training_symbols) != 1:
        raise ValueError("single_symbol scaling_scope requires exactly one training symbol")

    for key in ("encoder_features", "decoder_features", "scaled_features", "unscaled_features"):
        values = metadata.get(key, [])
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} must be a non-empty list")
        _assert_no_forbidden_active_fields(values, key)

    if TARGET_NAME in metadata["scaled_features"]:
        raise ValueError(f"{TARGET_NAME} must not appear in scaled_features")
    if TARGET_NAME not in metadata["unscaled_features"]:
        raise ValueError(f"{TARGET_NAME} must appear in unscaled_features")

    start_end_pairs = [
        ("train_start_time", "train_end_time"),
        ("validation_start_time", "validation_end_time"),
        ("test_start_time", "test_end_time"),
    ]
    for start_key, end_key in start_end_pairs:
        start_val = metadata.get(start_key)
        end_val = metadata.get(end_key)
        if start_val is None and end_val is None:
            continue
        if start_val is None or end_val is None:
            raise ValueError(f"{start_key} and {end_key} must both be set or both be None")


__all__ = [
    "DATASET_CONTRACT_VERSION",
    "DEPRECATED_FEATURES",
    "DECODER_KNOWN_FEATURES",
    "ENCODER_FEATURES",
    "EVALUATION_CONTRACT_VERSION",
    "FEATURE_VERSION",
    "MULTI_SYMBOL_SENTINEL",
    "FLOW_FEATURES",
    "RELATIVE_PRICE_FEATURES",
    "SUPPORTED_INFERENCE_ENTRYPOINT",
    "SYMBOL_FEATURES",
    "TARGET_NAME",
    "TECHNICAL_FEATURES",
    "TIME_FEATURES",
    "VOLUME_FEATURES",
    "WARMUP_SENSITIVE_FEATURES",
    "add_symbol_id_feature",
    "add_target_return_h",
    "add_technical_features",
    "add_time_features",
    "build_symbol_id_map",
    "build_dataset_metadata",
    "build_feature_dataframe",
    "fit_feature_scalers",
    "get_invalid_origin_mask",
    "get_required_decoder_features",
    "get_required_encoder_features",
    "get_scaled_feature_list",
    "get_unscaled_feature_list",
    "load_scaler_artifact",
    "normalize_candle_schema",
    "save_scaler_artifact",
    "transform_features",
    "validate_artifact_metadata",
    "validate_scaler_metadata",
    "validate_scaler_metadata_against_dataset_metadata",
    "validate_feature_schema",
]
