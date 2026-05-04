from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from utils.config import build_db_connection_url as _build_db_connection_url
from utils.config import TRADING_SYMBOLS


ACTIVE_DB_COLUMNS = [
    "open_time",
    "close_time",
    "symbol",
    "open",
    "close",
    "high",
    "low",
    "cvd",
    "volume",
]

LogFn = Callable[[str], None]


def normalize_timestamp(series: pd.Series) -> pd.Series:
    sample_values = series.head(5).tolist()
    series_dtype = str(series.dtype)
    if pd.api.types.is_datetime64_any_dtype(series):
        converted = pd.Series(series, index=series.index, name=series.name)
    else:
        try:
            if pd.api.types.is_integer_dtype(series) or pd.api.types.is_unsigned_integer_dtype(series):
                converted = pd.to_datetime(series, unit="us", errors="raise", utc=True)
            else:
                converted = pd.to_datetime(series, errors="raise", utc=True)
        except Exception as exc:
            raise ValueError(
                f"failed to convert timestamp series to datetime; dtype={series_dtype}; "
                f"sample_values={sample_values}"
            ) from exc

    converted = pd.Series(converted, index=series.index, name=series.name)
    if pd.api.types.is_datetime64_any_dtype(converted):
        tz = getattr(converted.dt, "tz", None)
        if tz is None:
            converted = converted.dt.tz_localize("UTC")
        else:
            converted = converted.dt.tz_convert("UTC")
    if converted.isna().any():
        raise ValueError(
            f"timestamp series contains invalid values after datetime conversion; "
            f"dtype={series_dtype}; sample_values={sample_values}"
        )
    if not pd.api.types.is_datetime64_any_dtype(converted):
        raise ValueError(
            f"timestamp series is not pandas datetime dtype after conversion; "
            f"dtype={converted.dtype}; sample_values={sample_values}"
        )
    return converted


def coerce_datetime_series(series: pd.Series, column_name: str) -> pd.Series:
    try:
        return normalize_timestamp(series)
    except Exception as exc:
        raise ValueError(
            f"failed to convert {column_name} to datetime; dtype={series.dtype}; "
            f"sample_values={series.head(5).tolist()}"
        ) from exc


def normalize_symbol(symbol: str) -> str:
    normalized = str(symbol).strip().lower().replace("/", "").replace("-", "")
    if not normalized:
        raise ValueError("symbol must be a non-empty string")
    return normalized


def resolve_candles_table(symbol: str) -> str:
    return f"_candles_trading_data.{normalize_symbol(symbol)}_1h"


def build_db_connection_url() -> str:
    return _build_db_connection_url()


def _validate_order(order: str) -> str:
    order_upper = order.upper()
    if order_upper not in {"ASC", "DESC"}:
        raise ValueError("order must be ASC or DESC")
    return order_upper


def _validate_limit(limit: int | None) -> None:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be > 0 when provided")


def _selected_columns_for_symbol(inspector, normalized_symbol: str) -> tuple[str, list[str]]:
    table_name = f"{normalized_symbol}_1h"
    table = resolve_candles_table(normalized_symbol)
    table_columns = [
        column["name"]
        for column in inspector.get_columns(table_name, schema="_candles_trading_data")
    ]
    selected_db_columns = [column for column in ACTIVE_DB_COLUMNS if column in table_columns]
    if "open_time" not in selected_db_columns:
        raise ValueError(f"table {table} is missing required column: open_time")
    if "close" not in selected_db_columns:
        raise ValueError(f"table {table} is missing required column: close")
    return table, selected_db_columns


def _normalize_loaded_candle_frame(df: pd.DataFrame, symbol_upper: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"no candle data loaded for symbol {symbol_upper}")
    if "open_time" not in df.columns:
        raise ValueError("loaded candle frame is missing required column: open_time")

    normalized = df.copy()
    normalized["open_time"] = normalize_timestamp(normalized["open_time"])
    if "close_time" in normalized.columns:
        normalized["close_time"] = normalize_timestamp(normalized["close_time"])

    if "symbol" not in normalized.columns:
        normalized["symbol"] = symbol_upper
    else:
        normalized["symbol"] = normalized["symbol"].astype("string")
        normalized["symbol"] = normalized["symbol"].fillna(symbol_upper)
        normalized["symbol"] = normalized["symbol"].str.strip()
        normalized.loc[normalized["symbol"] == "", "symbol"] = symbol_upper
        normalized["symbol"] = normalized["symbol"].fillna(symbol_upper)
    normalized["symbol"] = normalized["symbol"].astype("string").str.strip().str.upper()

    if normalized["symbol"].isna().any():
        raise ValueError(f"symbol column still contains null values after autofill for {symbol_upper}")
    if (normalized["symbol"].str.strip() == "").any():
        raise ValueError(f"symbol column contains empty values after normalization for {symbol_upper}")

    normalized = normalized.dropna(subset=["open_time", "close"])
    if normalized.empty:
        raise ValueError(f"no valid candle rows remain for symbol {symbol_upper}")

    normalized = normalized.sort_values("open_time").reset_index(drop=True)
    if not normalized["open_time"].is_monotonic_increasing:
        raise ValueError(f"loaded candle timestamps are not sorted in ascending order for {symbol_upper}")
    return normalized


def _read_candles_postgres_with_engine(
    engine,
    inspector,
    symbol: str,
    *,
    limit: int | None = None,
    order: str = "ASC",
    log_fn: LogFn | None = None,
) -> pd.DataFrame:
    order_upper = _validate_order(order)
    _validate_limit(limit)

    normalized_symbol = normalize_symbol(symbol)
    symbol_upper = normalized_symbol.upper()
    table, selected_db_columns = _selected_columns_for_symbol(inspector, normalized_symbol)
    if log_fn is not None:
        log_fn(
            f"[data] Loading symbol={symbol_upper} table={table} order={order_upper} "
            f"limit={'ALL' if limit is None else limit} columns={','.join(selected_db_columns)}"
        )
    selected_columns = ", ".join(selected_db_columns)

    limit_clause = ""
    params: dict[str, object] = {}
    if limit is not None:
        limit_clause = "LIMIT :limit"
        params["limit"] = limit

    query = text(
        f"""
        SELECT {selected_columns}
        FROM {table}
        ORDER BY open_time {order_upper}
        {limit_clause}
        """
    )

    parse_dates = [column for column in ["open_time", "close_time"] if column in selected_db_columns]
    with engine.begin() as conn:
        df = pd.read_sql(query, conn, params=params, parse_dates=parse_dates)
    normalized_df = _normalize_loaded_candle_frame(df, symbol_upper)
    if log_fn is not None:
        log_fn(
            f"[data] Loaded symbol={symbol_upper} rows={len(normalized_df)} "
            f"from={normalized_df['open_time'].iloc[0].isoformat()} to={normalized_df['open_time'].iloc[-1].isoformat()}"
        )
    return normalized_df


def read_candles_postgres(
    symbol: str,
    *,
    limit: int | None = None,
    order: str = "ASC",
    log_fn: LogFn | None = None,
) -> pd.DataFrame:
    db_url = build_db_connection_url()
    engine = create_engine(db_url)
    inspector = inspect(engine)
    return _read_candles_postgres_with_engine(
        engine,
        inspector,
        symbol,
        limit=limit,
        order=order,
        log_fn=log_fn,
    )


def read_candles_postgres_many(
    symbols: list[str],
    *,
    limit: int | None = None,
    order: str = "ASC",
    log_fn: LogFn | None = None,
) -> pd.DataFrame:
    if not symbols:
        raise ValueError("symbols list must be non-empty")

    normalized_symbols: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        normalized = normalize_symbol(symbol)
        if normalized in seen:
            continue
        seen.add(normalized)
        normalized_symbols.append(normalized)

    db_url = build_db_connection_url()
    engine = create_engine(db_url)
    inspector = inspect(engine)
    if log_fn is not None:
        log_fn(f"[data] Starting multi-symbol load for {len(normalized_symbols)} symbols")
    frames = [
        _read_candles_postgres_with_engine(
            engine,
            inspector,
            symbol,
            limit=limit,
            order=order,
            log_fn=log_fn,
        )
        for symbol in normalized_symbols
    ]
    if not frames:
        raise ValueError("no candle data loaded for the requested symbols")
    combined = pd.concat(frames, ignore_index=True)
    if log_fn is not None:
        log_fn(f"[data] Combined raw rows across universe={len(combined)}")
    return combined.sort_values(["symbol", "open_time"]).reset_index(drop=True)


def read_training_universe_candles(
    symbols: list[str] | None = None,
    *,
    limit: int | None = None,
    order: str = "ASC",
    clean: bool = True,
    log_fn: LogFn | None = None,
) -> pd.DataFrame:
    requested_symbols = symbols or list(TRADING_SYMBOLS)
    combined = read_candles_postgres_many(requested_symbols, limit=limit, order=order, log_fn=log_fn)
    if not clean:
        return combined

    if log_fn is not None:
        log_fn(f"[data] Cleaning training universe raw_rows={len(combined)}")
    cleaned = drop_incomplete_latest_candles(combined)
    if log_fn is not None:
        log_fn(f"[data] After dropping incomplete latest candles rows={len(cleaned)}")
    cleaned = keep_latest_regular_time_block(cleaned)
    if log_fn is not None:
        log_fn(f"[data] After keeping latest regular time blocks rows={len(cleaned)}")
    if cleaned.empty:
        raise ValueError("no closed candles remain after training universe cleanup")
    return cleaned


def drop_incomplete_latest_candles(
    df: pd.DataFrame,
    *,
    now_utc: datetime | None = None,
) -> pd.DataFrame:
    if df.empty or "close_time" not in df.columns:
        return df.copy()

    now = now_utc or datetime.now(timezone.utc)
    cleaned_groups: list[pd.DataFrame] = []

    for _, group in df.groupby("symbol", sort=False):
        group_local = group.sort_values("open_time").copy()
        if group_local.empty:
            continue

        last_close_time = group_local["close_time"].iloc[-1]
        if pd.notna(last_close_time) and last_close_time.tzinfo is not None and last_close_time > now:
            group_local = group_local.iloc[:-1].copy()

        cleaned_groups.append(group_local)

    if not cleaned_groups:
        return df.iloc[0:0].copy()
    return pd.concat(cleaned_groups, ignore_index=True)


def keep_latest_regular_time_block(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "open_time" not in df.columns:
        return df.copy()

    cleaned_groups: list[pd.DataFrame] = []

    for _, group in df.groupby("symbol", sort=False):
        group_local = group.sort_values("open_time").reset_index(drop=True).copy()
        if len(group_local) < 2:
            cleaned_groups.append(group_local)
            continue

        diffs = group_local["open_time"].diff().dropna()
        if diffs.empty:
            cleaned_groups.append(group_local)
            continue

        expected_delta = diffs.mode().iloc[0]
        if pd.isna(expected_delta) or expected_delta <= pd.Timedelta(0):
            raise ValueError("failed to infer a valid candle interval from open_time")

        irregular_positions = diffs[diffs != expected_delta].index
        if len(irregular_positions) == 0:
            cleaned_groups.append(group_local)
            continue

        # Keep the latest contiguous block so training and inference use a regular time axis.
        latest_block_start = int(irregular_positions.max())
        trimmed = group_local.iloc[latest_block_start:].reset_index(drop=True).copy()
        if trimmed.empty:
            raise ValueError("latest regular candle block is empty after removing irregular timestamps")
        cleaned_groups.append(trimmed)

    if not cleaned_groups:
        return df.iloc[0:0].copy()
    return pd.concat(cleaned_groups, ignore_index=True)


def filter_time_range(
    df: pd.DataFrame,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    if "open_time" in out.columns:
        out["open_time"] = normalize_timestamp(out["open_time"])
    if start_time is not None:
        start = pd.to_datetime(start_time, utc=True, errors="raise")
        out = out.loc[out["open_time"] >= start]
    if end_time is not None:
        end = pd.to_datetime(end_time, utc=True, errors="raise")
        out = out.loc[out["open_time"] <= end]
    return out.copy()


__all__ = [
    "ACTIVE_DB_COLUMNS",
    "build_db_connection_url",
    "coerce_datetime_series",
    "drop_incomplete_latest_candles",
    "filter_time_range",
    "keep_latest_regular_time_block",
    "normalize_timestamp",
    "normalize_symbol",
    "read_candles_postgres",
    "read_candles_postgres_many",
    "read_training_universe_candles",
    "resolve_candles_table",
]
