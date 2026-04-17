from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from module_ai.data_access import read_training_universe_candles
from module_ai.data_pipeline import build_feature_dataframe, build_symbol_id_map
from module_ai.symbols import normalize_training_symbols
from utils.config import TRADING_SYMBOLS

LogFn = Callable[[str], None]


@dataclass(frozen=True)
class TrainingUniverseBundle:
    raw_df: pd.DataFrame
    feature_df: pd.DataFrame
    symbol_id_map: dict[str, int]
    symbols: list[str]


def _resolve_symbols(symbols: list[str] | None = None) -> list[str]:
    requested_symbols = symbols or list(TRADING_SYMBOLS)
    normalized = normalize_training_symbols(list(requested_symbols))
    if not normalized:
        raise ValueError("training universe must contain at least one symbol")
    return normalized


def load_training_universe_bundle(
    *,
    horizon_h: int,
    symbols: list[str] | None = None,
    limit: int | None = None,
    log_fn: LogFn | None = None,
) -> TrainingUniverseBundle:
    resolved_symbols = _resolve_symbols(symbols)
    if log_fn is not None:
        log_fn(f"[universe] Resolved {len(resolved_symbols)} symbols for benchmark/training universe")
    raw_df = read_training_universe_candles(resolved_symbols, limit=limit, order="ASC", clean=True, log_fn=log_fn)
    symbol_id_map = build_symbol_id_map(resolved_symbols)
    if log_fn is not None:
        log_fn(f"[universe] Building feature dataframe horizon_h={horizon_h}")
    feature_df = build_feature_dataframe(raw_df, horizon_h=horizon_h, symbol_id_map=symbol_id_map)
    if log_fn is not None:
        log_fn(
            f"[universe] Feature dataframe ready rows={len(feature_df)} "
            f"symbols={len(resolved_symbols)} symbol_id_map_size={len(symbol_id_map)}"
        )
    return TrainingUniverseBundle(
        raw_df=raw_df,
        feature_df=feature_df,
        symbol_id_map=symbol_id_map,
        symbols=resolved_symbols,
    )


def summarize_training_universe(bundle: TrainingUniverseBundle) -> dict[str, Any]:
    per_symbol_rows = (
        bundle.feature_df.groupby("symbol", sort=True)
        .size()
        .astype(int)
        .to_dict()
    )
    invalid_rows = (
        bundle.feature_df.groupby("symbol", sort=True)[["target_return_h"]]
        .apply(lambda frame: int(frame["target_return_h"].isna().sum()))
        .to_dict()
    )
    return {
        "symbol_count": len(bundle.symbols),
        "symbols": bundle.symbols,
        "raw_rows": int(len(bundle.raw_df)),
        "feature_rows": int(len(bundle.feature_df)),
        "per_symbol_feature_rows": per_symbol_rows,
        "per_symbol_missing_target_rows": invalid_rows,
        "symbol_id_map": bundle.symbol_id_map,
    }


__all__ = [
    "TrainingUniverseBundle",
    "load_training_universe_bundle",
    "summarize_training_universe",
]
