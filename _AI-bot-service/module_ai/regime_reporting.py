from __future__ import annotations

from typing import Any

import pandas as pd

from module_ai.evaluate import evaluate_forecast_frame


REGIME_LABELS = ("low", "mid", "high")


def _normalize_raw_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "open_time", "close", "volume"}
    missing = sorted(required - set(raw_df.columns))
    if missing:
        raise ValueError(f"raw_df is missing required columns for regime reporting: {missing}")
    out = raw_df.copy()
    out["symbol"] = out["symbol"].astype(str)
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True, errors="coerce")
    if out["open_time"].isna().any():
        raise ValueError("raw_df contains invalid open_time values")
    out = out.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    return out


def _assign_tercile_labels(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if clean.nunique(dropna=True) <= 1:
        return pd.Series(["mid"] * len(clean), index=series.index, dtype="object")
    ranked = clean.rank(method="first")
    bins = pd.qcut(ranked, q=min(3, len(clean)), labels=list(REGIME_LABELS[: min(3, len(clean))]), duplicates="drop")
    return bins.astype(str)


def build_symbol_regime_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    frame = _normalize_raw_frame(raw_df)
    close_return = frame.groupby("symbol", sort=True)["close"].pct_change()
    enriched = frame.assign(close_return=close_return.fillna(0.0))
    grouped = enriched.groupby("symbol", sort=True)
    stats = grouped.agg(
        raw_rows=("symbol", "size"),
        median_volume=("volume", "median"),
        mean_volume=("volume", "mean"),
        return_volatility=("close_return", "std"),
    ).reset_index()
    stats["return_volatility"] = stats["return_volatility"].fillna(0.0)
    stats["volatility_regime"] = _assign_tercile_labels(stats["return_volatility"])
    stats["liquidity_regime"] = _assign_tercile_labels(stats["median_volume"])
    return stats.sort_values("symbol").reset_index(drop=True)


def _build_group_reports_for_dimension(
    forecast_df: pd.DataFrame,
    *,
    raw_df: pd.DataFrame,
    regime_frame: pd.DataFrame,
    dimension: str,
    horizon_h: int,
    signal_mode: str,
    signal_kwargs: dict[str, float],
    execution_rule: str,
    transaction_cost: float,
    slippage: float,
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for regime_name, symbol_frame in regime_frame.groupby(dimension, sort=True):
        symbols = sorted(symbol_frame["symbol"].astype(str).tolist())
        group_forecast_df = forecast_df.loc[forecast_df["symbol"].astype(str).isin(symbols)].copy()
        if group_forecast_df.empty:
            continue
        group_price_frame = raw_df.loc[raw_df["symbol"].astype(str).isin(symbols)].copy()
        report = evaluate_forecast_frame(
            group_forecast_df,
            horizon_h=horizon_h,
            signal_mode=signal_mode,
            signal_kwargs=signal_kwargs,
            execution_rule=execution_rule,
            transaction_cost=transaction_cost,
            slippage=slippage,
            price_frame=group_price_frame,
        )["report"]
        reports[str(regime_name)] = {
            "symbols": symbols,
            "symbol_count": len(symbols),
            "report": report,
        }
    return reports


def build_regime_group_reports(
    forecast_df: pd.DataFrame,
    *,
    raw_df: pd.DataFrame,
    horizon_h: int,
    signal_mode: str,
    signal_kwargs: dict[str, float],
    execution_rule: str,
    transaction_cost: float,
    slippage: float,
) -> dict[str, Any]:
    regime_frame = build_symbol_regime_frame(raw_df)
    return {
        "symbol_regimes": regime_frame.to_dict(orient="records"),
        "volatility_regime_reports": _build_group_reports_for_dimension(
            forecast_df,
            raw_df=raw_df,
            regime_frame=regime_frame,
            dimension="volatility_regime",
            horizon_h=horizon_h,
            signal_mode=signal_mode,
            signal_kwargs=signal_kwargs,
            execution_rule=execution_rule,
            transaction_cost=transaction_cost,
            slippage=slippage,
        ),
        "liquidity_regime_reports": _build_group_reports_for_dimension(
            forecast_df,
            raw_df=raw_df,
            regime_frame=regime_frame,
            dimension="liquidity_regime",
            horizon_h=horizon_h,
            signal_mode=signal_mode,
            signal_kwargs=signal_kwargs,
            execution_rule=execution_rule,
            transaction_cost=transaction_cost,
            slippage=slippage,
        ),
    }


__all__ = [
    "build_regime_group_reports",
    "build_symbol_regime_frame",
]
