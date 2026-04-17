from __future__ import annotations

from dataclasses import asdict, dataclass
from math import floor, isclose
from typing import Any, Iterable

import pandas as pd


HELPER_COLUMNS = {
    "target_return_h",
    "is_context",
    "is_valid_origin",
    "split_name",
}


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float
    val_ratio: float
    test_ratio: float
    encoder_len: int
    horizon_h: int
    warmup_buffer: int = 0
    walk_forward_val_size: int | None = None

    @classmethod
    def from_obj(cls, config: "SplitConfig | dict[str, Any]") -> "SplitConfig":
        if isinstance(config, cls):
            return config
        if not isinstance(config, dict):
            raise TypeError("config must be a SplitConfig instance or a dict")
        return cls(**config)


@dataclass(frozen=True)
class SplitMetadata:
    symbol: str | None
    symbol_count: int
    total_rows: int
    train_rows_raw: int
    val_rows_raw: int
    test_rows_raw: int
    train_rows_valid: int
    val_rows_valid: int
    test_rows_valid: int
    train_start_time: str | None
    train_end_time: str | None
    val_start_time: str | None
    val_end_time: str | None
    test_start_time: str | None
    test_end_time: str | None
    encoder_len: int
    horizon_h: int
    warmup_buffer: int


def _validate_config(config: SplitConfig) -> None:
    for name, value in (
        ("train_ratio", config.train_ratio),
        ("val_ratio", config.val_ratio),
        ("test_ratio", config.test_ratio),
    ):
        if value <= 0 or value >= 1:
            raise ValueError(f"{name} must be between 0 and 1")

    ratio_sum = config.train_ratio + config.val_ratio + config.test_ratio
    if not isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    if config.encoder_len <= 0:
        raise ValueError("encoder_len must be a positive integer")
    if config.horizon_h <= 0:
        raise ValueError("horizon_h must be a positive integer")
    if config.warmup_buffer < 0:
        raise ValueError("warmup_buffer must be >= 0")
    if config.walk_forward_val_size is not None and config.walk_forward_val_size <= 0:
        raise ValueError("walk_forward_val_size must be a positive integer when provided")


def _ensure_non_empty(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("input dataframe is empty")


def _ensure_single_symbol(df: pd.DataFrame) -> str | None:
    if "symbol" not in df.columns:
        return None
    unique_symbols = df["symbol"].dropna().unique()
    if len(unique_symbols) == 0:
        raise ValueError("symbol column exists but contains no non-null values")
    if len(unique_symbols) > 1:
        raise ValueError("split module operates on one symbol at a time")
    return str(unique_symbols[0])


def _extract_symbols(df: pd.DataFrame) -> list[str]:
    if "symbol" not in df.columns:
        return []
    unique_symbols = df["symbol"].dropna().astype(str).str.strip().str.upper().unique().tolist()
    if len(unique_symbols) == 0:
        raise ValueError("symbol column exists but contains no non-null values")
    return sorted(unique_symbols)


def _require_datetime_series(df: pd.DataFrame) -> pd.Series:
    if "open_time" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["open_time"]):
            raise ValueError("open_time must be datetime-like")
        series = df["open_time"]
    elif isinstance(df.index, pd.DatetimeIndex):
        series = pd.Series(df.index, index=df.index, name="open_time")
    else:
        raise ValueError("dataframe must contain a datetime-like open_time column or DatetimeIndex")

    if series.isna().any():
        raise ValueError("timestamp series contains NaT values")
    return series


def _validate_time_axis(df: pd.DataFrame) -> None:
    times = _require_datetime_series(df)
    if not times.is_monotonic_increasing:
        raise ValueError("dataframe must be sorted by time in ascending order")
    if times.duplicated().any():
        raise ValueError("duplicate timestamps detected")

    diffs = times.diff().dropna()
    if diffs.empty:
        return

    unique_diffs = diffs.unique()
    if len(unique_diffs) != 1:
        raise ValueError("missing or irregular timestamps detected")


def _validate_grouped_time_axis(df: pd.DataFrame) -> None:
    if "symbol" not in df.columns:
        _validate_time_axis(df)
        return
    for _, group in df.groupby("symbol", sort=False):
        _validate_time_axis(group)


def _time_bounds(df: pd.DataFrame) -> tuple[str | None, str | None]:
    if df.empty:
        return None, None
    times = _require_datetime_series(df)
    return times.iloc[0].isoformat(), times.iloc[-1].isoformat()


def _feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(HELPER_COLUMNS) | {"open_time"}
    feature_cols = [col for col in df.columns if col not in excluded]
    if not feature_cols:
        raise ValueError("no usable columns available for validity checks")
    return feature_cols


def _minimum_rows_for_one_origin(encoder_len: int, horizon_h: int) -> int:
    return encoder_len + horizon_h


def get_valid_origins(df: pd.DataFrame, encoder_len: int, horizon_h: int) -> pd.Index:
    """
    Return origin indices that satisfy:
    - enough encoder history exists
    - target close[t + H] exists
    - no NaN values appear in any required columns inside the encoder window
    - the time axis is regular and strictly increasing
    """
    if encoder_len <= 0:
        raise ValueError("encoder_len must be positive")
    if horizon_h <= 0:
        raise ValueError("horizon_h must be positive")

    _ensure_non_empty(df)
    symbols = _extract_symbols(df)
    if len(symbols) > 1:
        valid_indexes: list[pd.Index] = []
        for _, group in df.groupby("symbol", sort=False):
            valid_indexes.append(get_valid_origins(group.copy(), encoder_len, horizon_h))
        if not valid_indexes:
            raise ValueError("no valid origins available after grouped validation")
        combined_index = valid_indexes[0]
        for index in valid_indexes[1:]:
            combined_index = combined_index.append(index)
        if len(combined_index) == 0:
            raise ValueError("no valid origins available after grouped validation")
        return combined_index

    _ensure_single_symbol(df)
    _validate_time_axis(df)

    if "close" not in df.columns:
        raise ValueError("close column is required to validate target availability")

    min_rows = _minimum_rows_for_one_origin(encoder_len, horizon_h)
    if len(df) < min_rows:
        raise ValueError(
            f"insufficient history: need at least {min_rows} rows, got {len(df)}"
        )

    feature_cols = _feature_columns(df)
    feature_has_nan = df[feature_cols].isna().any(axis=1)
    encoder_window_has_nan = (
        feature_has_nan.rolling(window=encoder_len, min_periods=encoder_len).max().fillna(1).astype(bool)
    )

    n_rows = len(df)
    valid_mask = pd.Series(False, index=df.index)
    close_values = df["close"]

    for pos in range(encoder_len - 1, n_rows - horizon_h):
        target_pos = pos + horizon_h
        if encoder_window_has_nan.iloc[pos]:
            continue
        if pd.isna(close_values.iloc[target_pos]):
            continue
        valid_mask.iloc[pos] = True

    valid_origins = df.index[valid_mask]
    if len(valid_origins) == 0:
        raise ValueError("no valid origins available after applying encoder, horizon, and NaN rules")
    return valid_origins


def apply_warmup_context(
    target_df: pd.DataFrame,
    context_df: pd.DataFrame,
    encoder_len: int,
    warmup_buffer: int,
) -> pd.DataFrame:
    """
    Prepend context rows to a labeled target dataframe and mark them explicitly.

    Context rows are not labeled samples and must only come from prior history.
    """
    if encoder_len <= 0:
        raise ValueError("encoder_len must be positive")
    if warmup_buffer < 0:
        raise ValueError("warmup_buffer must be >= 0")
    if target_df.empty:
        raise ValueError("target_df is empty")

    required_context = max(encoder_len - 1, warmup_buffer)
    context_tail = context_df.tail(required_context).copy() if required_context > 0 else context_df.iloc[0:0].copy()

    if required_context > 0 and len(context_tail) < required_context:
        raise ValueError(
            f"insufficient context: need {required_context} rows, got {len(context_tail)}"
        )

    context_tail = context_tail.copy()
    labeled = target_df.copy()

    context_tail["is_context"] = True
    labeled["is_context"] = False

    combined = pd.concat([context_tail, labeled], axis=0)
    _validate_grouped_time_axis(combined)
    return combined


def _split_time_boundaries(
    df: pd.DataFrame,
    config: SplitConfig,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    unique_times = (
        pd.Series(pd.to_datetime(df["open_time"], utc=True, errors="raise").unique())
        .sort_values()
        .reset_index(drop=True)
    )
    min_rows = _minimum_rows_for_one_origin(config.encoder_len, config.horizon_h)
    if len(unique_times) < min_rows * 3:
        raise ValueError("dataset is too small to build train/validation/test splits with valid labeled origins")

    total_times = len(unique_times)
    val_size = max(floor(total_times * config.val_ratio), min_rows)
    test_size = max(floor(total_times * config.test_ratio), min_rows)
    train_size = total_times - val_size - test_size
    if train_size < min_rows:
        raise ValueError("dataset is too small to satisfy minimum train/validation/test window sizes")

    train_end_pos = train_size
    val_end_pos = train_size + val_size
    if train_end_pos <= 0 or val_end_pos <= train_end_pos or val_end_pos >= total_times:
        raise ValueError("invalid split boundaries derived from ratios")

    train_end_time = pd.Timestamp(unique_times.iloc[train_end_pos - 1])
    val_end_time = pd.Timestamp(unique_times.iloc[val_end_pos - 1])
    return train_end_time, val_end_time


def _filter_time_window(
    df: pd.DataFrame,
    *,
    start_exclusive: pd.Timestamp | None = None,
    end_inclusive: pd.Timestamp | None = None,
) -> pd.DataFrame:
    out = df.copy()
    times = pd.to_datetime(out["open_time"], utc=True, errors="raise")
    if start_exclusive is not None:
        out = out.loc[times > start_exclusive].copy()
        times = pd.to_datetime(out["open_time"], utc=True, errors="raise")
    if end_inclusive is not None:
        out = out.loc[times <= end_inclusive].copy()
    return out.copy()


def _validate_split_minimum_rows_per_symbol(
    split_df: pd.DataFrame,
    *,
    min_rows: int,
    split_name: str,
) -> None:
    if split_df.empty:
        raise ValueError(f"{split_name} split is empty")
    counts = split_df.groupby("symbol", sort=False).size()
    too_small = counts[counts < min_rows]
    if not too_small.empty:
        rendered = ", ".join(f"{symbol}={count}" for symbol, count in too_small.items())
        raise ValueError(f"{split_name} split is too small for some symbols: {rendered}")


def split_dataset(
    df: pd.DataFrame,
    config: SplitConfig | dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Split a dataframe into strict chronological train/validation/test subsets.

    Each returned dataframe contains only valid labeled origins for that split.
    """
    split_config = SplitConfig.from_obj(config)
    _validate_config(split_config)
    _ensure_non_empty(df)

    df_local = df.copy()
    symbols = _extract_symbols(df_local)
    symbol = symbols[0] if len(symbols) == 1 else "__multi__"
    _validate_grouped_time_axis(df_local)
    total_rows = len(df_local)

    min_rows = _minimum_rows_for_one_origin(split_config.encoder_len, split_config.horizon_h)
    train_end_time, val_end_time = _split_time_boundaries(df_local, split_config)

    train_raw = _filter_time_window(df_local, end_inclusive=train_end_time)
    val_raw = _filter_time_window(df_local, start_exclusive=train_end_time, end_inclusive=val_end_time)
    test_raw = _filter_time_window(df_local, start_exclusive=val_end_time)

    for name, split_df in (("train", train_raw), ("validation", val_raw), ("test", test_raw)):
        _validate_split_minimum_rows_per_symbol(split_df, min_rows=min_rows, split_name=name)

    train_valid = train_raw.loc[get_valid_origins(train_raw, split_config.encoder_len, split_config.horizon_h)].copy()
    val_valid = val_raw.loc[get_valid_origins(val_raw, split_config.encoder_len, split_config.horizon_h)].copy()
    test_valid = test_raw.loc[get_valid_origins(test_raw, split_config.encoder_len, split_config.horizon_h)].copy()

    metadata = SplitMetadata(
        symbol=symbol,
        symbol_count=len(symbols),
        total_rows=total_rows,
        train_rows_raw=len(train_raw),
        val_rows_raw=len(val_raw),
        test_rows_raw=len(test_raw),
        train_rows_valid=len(train_valid),
        val_rows_valid=len(val_valid),
        test_rows_valid=len(test_valid),
        train_start_time=_time_bounds(train_raw)[0],
        train_end_time=_time_bounds(train_raw)[1],
        val_start_time=_time_bounds(val_raw)[0],
        val_end_time=_time_bounds(val_raw)[1],
        test_start_time=_time_bounds(test_raw)[0],
        test_end_time=_time_bounds(test_raw)[1],
        encoder_len=split_config.encoder_len,
        horizon_h=split_config.horizon_h,
        warmup_buffer=split_config.warmup_buffer,
    )

    return train_valid, val_valid, test_valid, asdict(metadata)


def _walk_forward_block_size(n_rows: int, config: SplitConfig) -> int:
    if config.walk_forward_val_size is not None:
        return config.walk_forward_val_size

    train_val_ratio = config.train_ratio + config.val_ratio
    if train_val_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be positive for walk-forward folds")

    proportional_val_ratio = config.val_ratio / train_val_ratio
    block_size = floor(n_rows * proportional_val_ratio)
    return max(block_size, _minimum_rows_for_one_origin(config.encoder_len, config.horizon_h))


def generate_walk_forward_folds(
    train_val_df: pd.DataFrame,
    config: SplitConfig | dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Generate expanding-window walk-forward folds from a single-symbol train+validation dataframe.

    Each fold contains:
    - fold_train_df: valid labeled origins from the raw train window
    - fold_val_df: valid labeled origins from the raw validation block with prepended context rows
    - metadata: fold-specific boundaries and counts
    """
    split_config = SplitConfig.from_obj(config)
    _validate_config(split_config)
    _ensure_non_empty(train_val_df)

    df_local = train_val_df.copy()
    symbol = _ensure_single_symbol(df_local)
    _validate_time_axis(df_local)

    min_rows = _minimum_rows_for_one_origin(split_config.encoder_len, split_config.horizon_h)
    n_rows = len(df_local)
    if n_rows < min_rows * 2:
        raise ValueError("train_val_df is too small to build walk-forward folds")

    train_val_ratio = split_config.train_ratio + split_config.val_ratio
    if train_val_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be positive")

    initial_train_ratio = split_config.train_ratio / train_val_ratio
    initial_train_end = floor(n_rows * initial_train_ratio)
    if initial_train_end < min_rows:
        raise ValueError("initial expanding train window is too small")

    block_size = _walk_forward_block_size(n_rows, split_config)
    folds: list[dict[str, Any]] = []
    fold_start = initial_train_end
    fold_number = 1

    while fold_start < n_rows:
        fold_end = min(fold_start + block_size, n_rows)
        raw_train = df_local.iloc[:fold_start].copy()
        raw_val = df_local.iloc[fold_start:fold_end].copy()

        if len(raw_val) < min_rows:
            raise ValueError(
                f"validation fold {fold_number} is too small: need at least {min_rows} rows, got {len(raw_val)}"
            )

        fold_train_df = raw_train.loc[
            get_valid_origins(raw_train, split_config.encoder_len, split_config.horizon_h)
        ].copy()
        fold_val_labeled = raw_val.loc[
            get_valid_origins(raw_val, split_config.encoder_len, split_config.horizon_h)
        ].copy()
        fold_val_df = apply_warmup_context(
            fold_val_labeled,
            raw_train,
            split_config.encoder_len,
            split_config.warmup_buffer,
        )

        train_bounds = _time_bounds(raw_train)
        val_bounds = _time_bounds(raw_val)
        folds.append(
            {
                "raw_train_df": raw_train.copy(),
                "raw_val_df": raw_val.copy(),
                "fold_train_df": fold_train_df,
                "fold_val_df": fold_val_df,
                "metadata": {
                    "fold_number": fold_number,
                    "symbol": symbol,
                    "train_rows_raw": len(raw_train),
                    "val_rows_raw": len(raw_val),
                    "train_rows_valid": len(fold_train_df),
                    "val_rows_valid": len(fold_val_labeled),
                    "train_start_time": train_bounds[0],
                    "train_end_time": train_bounds[1],
                    "val_start_time": val_bounds[0],
                    "val_end_time": val_bounds[1],
                    "encoder_len": split_config.encoder_len,
                    "horizon_h": split_config.horizon_h,
                    "warmup_buffer": split_config.warmup_buffer,
                },
            }
        )

        fold_start = fold_end
        fold_number += 1

    if not folds:
        raise ValueError("no walk-forward folds could be generated")
    return folds


__all__ = [
    "SplitConfig",
    "apply_warmup_context",
    "generate_walk_forward_folds",
    "get_valid_origins",
    "split_dataset",
]
