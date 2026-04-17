from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from module_ai.data_pipeline import TARGET_NAME
from module_ai.splits import SplitConfig, generate_walk_forward_folds, split_dataset


FORECAST_Q10 = "q10_return_h"
FORECAST_Q50 = "q50_return_h"
FORECAST_Q90 = "q90_return_h"
FORECAST_SYMBOL = "symbol"
FORECAST_TIME = "origin_timestamp"
FORECAST_ACTUAL = "actual_return_h"

SIGNAL_COLUMN = "signal"
INTERVAL_WIDTH_COLUMN = "interval_width"
TRADE_EXECUTION_CLOSE_TO_CLOSE = "close_to_close"
TRADE_EXECUTION_NEXT_OPEN_TO_H_CLOSE = "next_open_to_h_close"

SIGNAL_BUY = "BUY"
SIGNAL_SELL = "SELL"
SIGNAL_HOLD = "HOLD"

REQUIRED_FORECAST_COLUMNS = [
    FORECAST_SYMBOL,
    FORECAST_TIME,
    FORECAST_ACTUAL,
    FORECAST_Q10,
    FORECAST_Q50,
    FORECAST_Q90,
]


@dataclass(frozen=True)
class ForecastRecord:
    symbol: str
    origin_timestamp: str
    actual_return_h: float
    q10_return_h: float
    q50_return_h: float
    q90_return_h: float


@dataclass(frozen=True)
class SignalRecord:
    symbol: str
    origin_timestamp: str
    signal: str
    interval_width: float
    thresholds_used: dict[str, Any]


@dataclass(frozen=True)
class TradeRecord:
    symbol: str
    origin_timestamp: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    gross_return: float
    net_return: float
    signal: str


@dataclass(frozen=True)
class EvaluationReport:
    forecast_metrics: dict[str, float]
    signal_metrics: dict[str, float]
    trading_metrics: dict[str, float]
    forecast_count: int
    signal_count: int
    trade_count: int
    notes: list[str]


def _as_config(config: SplitConfig | dict[str, Any]) -> SplitConfig:
    return SplitConfig.from_obj(config)


def _as_dataframe(records: pd.DataFrame | list[dict[str, Any]] | list[ForecastRecord]) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records.copy()
    if not records:
        return pd.DataFrame()
    first = records[0]
    if isinstance(first, ForecastRecord):
        return pd.DataFrame([asdict(item) for item in records])
    if isinstance(first, dict):
        return pd.DataFrame(records)
    raise TypeError("unsupported forecast record type")


def _ensure_required_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _coerce_forecast_frame(
    forecasts: pd.DataFrame | list[dict[str, Any]] | list[ForecastRecord],
    fallback_actuals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    forecast_df = _as_dataframe(forecasts)
    if forecast_df.empty:
        raise ValueError("empty forecast set")

    if FORECAST_TIME not in forecast_df.columns:
        raise ValueError("forecast set must contain origin_timestamp")

    forecast_df = forecast_df.copy()
    forecast_df[FORECAST_TIME] = pd.to_datetime(forecast_df[FORECAST_TIME], utc=True, errors="coerce")
    if forecast_df[FORECAST_TIME].isna().any():
        raise ValueError("forecast set contains invalid origin timestamps")

    if fallback_actuals is not None and FORECAST_ACTUAL not in forecast_df.columns:
        _ensure_required_columns(fallback_actuals, ["open_time", TARGET_NAME, "symbol"], "fallback actual frame")
        actual_source = fallback_actuals.copy()
        if "is_context" in actual_source.columns:
            actual_source = actual_source.loc[~actual_source["is_context"].fillna(False)]
        actual_lookup = actual_source.loc[:, ["open_time", "symbol", TARGET_NAME]].rename(
            columns={"open_time": FORECAST_TIME, TARGET_NAME: FORECAST_ACTUAL}
        )
        actual_lookup[FORECAST_TIME] = pd.to_datetime(actual_lookup[FORECAST_TIME], utc=True, errors="coerce")
        forecast_df = forecast_df.merge(actual_lookup, on=[FORECAST_TIME, "symbol"], how="left")

    _ensure_required_columns(forecast_df, REQUIRED_FORECAST_COLUMNS, "forecast set")

    numeric_cols = [FORECAST_ACTUAL, FORECAST_Q10, FORECAST_Q50, FORECAST_Q90]
    for col in numeric_cols:
        forecast_df[col] = pd.to_numeric(forecast_df[col], errors="coerce")

    if forecast_df[numeric_cols].isna().any().any():
        raise ValueError("forecast set contains NaN values in required return columns")

    invalid_order = (forecast_df[FORECAST_Q10] > forecast_df[FORECAST_Q50]) | (
        forecast_df[FORECAST_Q50] > forecast_df[FORECAST_Q90]
    )
    if invalid_order.any():
        raise ValueError("invalid quantile ordering detected in forecast set")

    forecast_df = forecast_df.sort_values([FORECAST_SYMBOL, FORECAST_TIME]).reset_index(drop=True)
    return forecast_df


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    errors = y_true - y_pred
    return float(np.mean(np.maximum(quantile * errors, (quantile - 1.0) * errors)))


def evaluate_forecasts(
    forecasts: pd.DataFrame | list[dict[str, Any]] | list[ForecastRecord],
) -> dict[str, float]:
    """
    Evaluate quantile forecasts in return space only.
    """
    forecast_df = _coerce_forecast_frame(forecasts)
    y_true = forecast_df[FORECAST_ACTUAL].to_numpy(dtype=np.float64)
    q10 = forecast_df[FORECAST_Q10].to_numpy(dtype=np.float64)
    q50 = forecast_df[FORECAST_Q50].to_numpy(dtype=np.float64)
    q90 = forecast_df[FORECAST_Q90].to_numpy(dtype=np.float64)

    mae = float(np.mean(np.abs(q50 - y_true)))
    rmse = float(np.sqrt(np.mean((q50 - y_true) ** 2)))
    interval_width = q90 - q10
    interval_coverage = float(np.mean((y_true >= q10) & (y_true <= q90)))

    return {
        "forecast_count": float(len(forecast_df)),
        "mae_q50": mae,
        "rmse_q50": rmse,
        "pinball_q10": _pinball_loss(y_true, q10, 0.1),
        "pinball_q50": _pinball_loss(y_true, q50, 0.5),
        "pinball_q90": _pinball_loss(y_true, q90, 0.9),
        "interval_width_mean": float(np.mean(interval_width)),
        "interval_coverage_rate": interval_coverage,
    }


def forecast_to_signal(
    forecasts: pd.DataFrame | list[dict[str, Any]] | list[ForecastRecord],
    *,
    mode: str = "quantile_barrier",
    cost_threshold: float = 0.0,
    buy_threshold: float = 0.0,
    sell_threshold: float = 0.0,
    max_width: float | None = None,
) -> pd.DataFrame:
    """
    Convert forecast quantiles into offline evaluation signals.
    """
    forecast_df = _coerce_forecast_frame(forecasts)
    signal_df = forecast_df.copy()
    signal_df[INTERVAL_WIDTH_COLUMN] = signal_df[FORECAST_Q90] - signal_df[FORECAST_Q10]

    if mode == "quantile_barrier":
        signal_df[SIGNAL_COLUMN] = np.where(
            signal_df[FORECAST_Q10] > cost_threshold,
            SIGNAL_BUY,
            np.where(signal_df[FORECAST_Q90] < -cost_threshold, SIGNAL_SELL, SIGNAL_HOLD),
        )
        thresholds_used = {
            "mode": mode,
            "cost_threshold": cost_threshold,
        }
    elif mode == "median_with_width":
        if max_width is None:
            raise ValueError("max_width is required for mode='median_with_width'")
        width_ok = signal_df[INTERVAL_WIDTH_COLUMN] < max_width
        signal_df[SIGNAL_COLUMN] = np.where(
            width_ok & (signal_df[FORECAST_Q50] > buy_threshold),
            SIGNAL_BUY,
            np.where(
                width_ok & (signal_df[FORECAST_Q50] < -sell_threshold),
                SIGNAL_SELL,
                SIGNAL_HOLD,
            ),
        )
        thresholds_used = {
            "mode": mode,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            "max_width": max_width,
        }
    else:
        raise ValueError(f"unsupported signal mode: {mode}")

    signal_df["thresholds_used"] = [thresholds_used] * len(signal_df)
    return signal_df


def _compute_signal_metrics(signal_df: pd.DataFrame) -> dict[str, float]:
    if signal_df.empty:
        raise ValueError("empty signal set")

    actual = signal_df[FORECAST_ACTUAL]
    predicted_buy = signal_df[SIGNAL_COLUMN] == SIGNAL_BUY
    predicted_sell = signal_df[SIGNAL_COLUMN] == SIGNAL_SELL
    actual_buy = actual > 0
    actual_sell = actual < 0

    def _safe_ratio(num: float, den: float) -> float:
        return float(num / den) if den else 0.0

    directional_predictions = signal_df[SIGNAL_COLUMN] != SIGNAL_HOLD
    directional_accuracy = _safe_ratio(
        ((predicted_buy & actual_buy) | (predicted_sell & actual_sell)).sum(),
        directional_predictions.sum(),
    )

    buy_precision = _safe_ratio((predicted_buy & actual_buy).sum(), predicted_buy.sum())
    buy_recall = _safe_ratio((predicted_buy & actual_buy).sum(), actual_buy.sum())
    sell_precision = _safe_ratio((predicted_sell & actual_sell).sum(), predicted_sell.sum())
    sell_recall = _safe_ratio((predicted_sell & actual_sell).sum(), actual_sell.sum())

    return {
        "signal_count": float(len(signal_df)),
        "buy_signal_count": float(predicted_buy.sum()),
        "sell_signal_count": float(predicted_sell.sum()),
        "hold_signal_count": float((signal_df[SIGNAL_COLUMN] == SIGNAL_HOLD).sum()),
        "directional_accuracy": directional_accuracy,
        "buy_precision": buy_precision,
        "buy_recall": buy_recall,
        "sell_precision": sell_precision,
        "sell_recall": sell_recall,
    }


def _build_trade_from_close_to_close(row: pd.Series, transaction_cost: float, slippage: float) -> TradeRecord:
    actual_return = float(row[FORECAST_ACTUAL])
    direction = 1.0 if row[SIGNAL_COLUMN] == SIGNAL_BUY else -1.0
    gross_return = direction * actual_return
    net_return = gross_return - transaction_cost - slippage

    entry_price = 1.0
    exit_price = 1.0 + gross_return
    if exit_price <= 0:
        exit_price = 0.0

    timestamp = pd.Timestamp(row[FORECAST_TIME])
    return TradeRecord(
        symbol=str(row[FORECAST_SYMBOL]),
        origin_timestamp=timestamp.isoformat(),
        entry_time=timestamp.isoformat(),
        exit_time=timestamp.isoformat(),
        entry_price=entry_price,
        exit_price=exit_price,
        gross_return=gross_return,
        net_return=net_return,
        signal=str(row[SIGNAL_COLUMN]),
    )


def _build_trade_from_price_frame(
    row: pd.Series,
    price_frame: pd.DataFrame,
    horizon_h: int,
    transaction_cost: float,
    slippage: float,
) -> TradeRecord:
    _ensure_required_columns(price_frame, ["symbol", "open_time", "open", "close"], "price frame")
    if horizon_h <= 0:
        raise ValueError("horizon_h must be > 0")

    local_prices = price_frame.copy()
    local_prices["open_time"] = pd.to_datetime(local_prices["open_time"], utc=True, errors="coerce")
    if local_prices["open_time"].isna().any():
        raise ValueError("price frame contains invalid timestamps")

    local_prices = local_prices.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    row_symbol = str(row[FORECAST_SYMBOL])
    row_time = pd.Timestamp(row[FORECAST_TIME])
    symbol_prices = local_prices[local_prices["symbol"].astype(str) == row_symbol].reset_index(drop=True)
    if symbol_prices.empty:
        raise ValueError(f"missing price history for symbol {row_symbol}")

    origin_matches = symbol_prices.index[symbol_prices["open_time"] == row_time].tolist()
    if not origin_matches:
        raise ValueError(f"origin timestamp {row_time.isoformat()} not found in price frame")

    origin_pos = origin_matches[0]
    entry_pos = origin_pos + 1
    exit_pos = origin_pos + horizon_h

    if entry_pos >= len(symbol_prices) or exit_pos >= len(symbol_prices):
        raise ValueError("missing entry or exit price for trade simulation")

    entry_price = float(symbol_prices.iloc[entry_pos]["open"])
    exit_price = float(symbol_prices.iloc[exit_pos]["close"])
    if not np.isfinite(entry_price) or not np.isfinite(exit_price):
        raise ValueError("entry or exit price is invalid")
    if entry_price <= 0:
        raise ValueError("entry price must be > 0")

    direction = 1.0 if row[SIGNAL_COLUMN] == SIGNAL_BUY else -1.0
    raw_return = (exit_price / entry_price) - 1.0
    gross_return = direction * raw_return
    net_return = gross_return - transaction_cost - slippage

    return TradeRecord(
        symbol=row_symbol,
        origin_timestamp=row_time.isoformat(),
        entry_time=pd.Timestamp(symbol_prices.iloc[entry_pos]["open_time"]).isoformat(),
        exit_time=pd.Timestamp(symbol_prices.iloc[exit_pos]["open_time"]).isoformat(),
        entry_price=entry_price,
        exit_price=exit_price,
        gross_return=gross_return,
        net_return=net_return,
        signal=str(row[SIGNAL_COLUMN]),
    )


def _compute_trading_metrics(trade_df: pd.DataFrame, total_signals: int) -> dict[str, float]:
    if trade_df.empty:
        return {
            "trade_count": 0.0,
            "cumulative_return": 0.0,
            "average_trade_return": 0.0,
            "hit_rate": 0.0,
            "max_drawdown": 0.0,
            "turnover": 0.0,
            "profit_factor": 0.0,
            "sharpe_like": 0.0,
        }

    net_returns = trade_df["net_return"].to_numpy(dtype=np.float64)
    equity_curve = np.cumprod(1.0 + net_returns)
    cumulative_return = float(equity_curve[-1] - 1.0)
    running_peak = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve / running_peak - 1.0
    max_drawdown = float(abs(np.min(drawdowns)))

    gains = net_returns[net_returns > 0].sum()
    losses = -net_returns[net_returns < 0].sum()
    profit_factor = float(gains / losses) if losses > 0 else float("inf")

    mean_return = float(np.mean(net_returns))
    std_return = float(np.std(net_returns, ddof=0))
    sharpe_like = float(mean_return / std_return) if std_return > 0 else 0.0

    return {
        "trade_count": float(len(trade_df)),
        "cumulative_return": cumulative_return,
        "average_trade_return": mean_return,
        "hit_rate": float(np.mean(net_returns > 0)),
        "max_drawdown": max_drawdown,
        "turnover": float(len(trade_df) / total_signals) if total_signals > 0 else 0.0,
        "profit_factor": profit_factor,
        "sharpe_like": sharpe_like,
    }


def evaluate_forecast_frame(
    forecasts: pd.DataFrame | list[dict[str, Any]] | list[ForecastRecord],
    *,
    horizon_h: int,
    signal_mode: str = "quantile_barrier",
    signal_kwargs: dict[str, Any] | None = None,
    execution_rule: str = TRADE_EXECUTION_CLOSE_TO_CLOSE,
    transaction_cost: float = 0.0,
    slippage: float = 0.0,
    price_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    signal_kwargs = signal_kwargs or {}
    forecast_df = _coerce_forecast_frame(forecasts)
    signal_df = forecast_to_signal(forecast_df, mode=signal_mode, **signal_kwargs)
    trade_df, trading_metrics = simulate_trades(
        signal_df,
        horizon_h=horizon_h,
        execution_rule=execution_rule,
        transaction_cost=transaction_cost,
        slippage=slippage,
        price_frame=price_frame,
    )

    report = EvaluationReport(
        forecast_metrics=evaluate_forecasts(forecast_df),
        signal_metrics=_compute_signal_metrics(signal_df),
        trading_metrics=trading_metrics,
        forecast_count=len(forecast_df),
        signal_count=len(signal_df),
        trade_count=len(trade_df),
        notes=["forecast_frame_evaluation"],
    )
    return {
        "report": asdict(report),
        "forecasts": forecast_df,
        "signals": signal_df,
        "trades": trade_df,
    }


def simulate_trades(
    signals: pd.DataFrame,
    *,
    horizon_h: int,
    execution_rule: str = TRADE_EXECUTION_CLOSE_TO_CLOSE,
    transaction_cost: float = 0.0,
    slippage: float = 0.0,
    price_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Run a deterministic one-trade-per-origin offline trade simulation.
    """
    if horizon_h <= 0:
        raise ValueError("horizon_h must be > 0")
    if transaction_cost < 0 or slippage < 0:
        raise ValueError("transaction_cost and slippage must be >= 0")
    if signals.empty:
        raise ValueError("empty signal set")

    _ensure_required_columns(
        signals,
        REQUIRED_FORECAST_COLUMNS + [SIGNAL_COLUMN],
        "signals",
    )

    trade_rows = signals[signals[SIGNAL_COLUMN].isin([SIGNAL_BUY, SIGNAL_SELL])].copy()
    if trade_rows.empty:
        trade_df = pd.DataFrame(columns=[field.name for field in TradeRecord.__dataclass_fields__.values()])
        return trade_df, _compute_trading_metrics(trade_df, len(signals))

    records: list[TradeRecord] = []
    for _, row in trade_rows.iterrows():
        if execution_rule == TRADE_EXECUTION_CLOSE_TO_CLOSE:
            records.append(_build_trade_from_close_to_close(row, transaction_cost, slippage))
        elif execution_rule == TRADE_EXECUTION_NEXT_OPEN_TO_H_CLOSE:
            if price_frame is None:
                raise ValueError("price_frame is required for execution_rule='next_open_to_h_close'")
            records.append(
                _build_trade_from_price_frame(
                    row=row,
                    price_frame=price_frame,
                    horizon_h=horizon_h,
                    transaction_cost=transaction_cost,
                    slippage=slippage,
                )
            )
        else:
            raise ValueError(f"unsupported execution_rule: {execution_rule}")

    trade_df = pd.DataFrame([asdict(record) for record in records])
    metrics = _compute_trading_metrics(trade_df, len(signals))
    return trade_df, metrics


def run_walk_forward_evaluation(
    train_val_df: pd.DataFrame,
    config: SplitConfig | dict[str, Any],
    prediction_fn: Callable[..., pd.DataFrame | list[dict[str, Any]] | list[ForecastRecord]],
    *,
    signal_mode: str = "quantile_barrier",
    signal_kwargs: dict[str, Any] | None = None,
    execution_rule: str = TRADE_EXECUTION_CLOSE_TO_CLOSE,
    transaction_cost: float = 0.0,
    slippage: float = 0.0,
    price_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Run walk-forward validation using generate_walk_forward_folds(...) and a pluggable prediction callback.
    """
    split_config = _as_config(config)
    folds = generate_walk_forward_folds(train_val_df, split_config)
    if not folds:
        raise ValueError("no walk-forward folds generated")

    signal_kwargs = signal_kwargs or {}
    all_forecasts: list[pd.DataFrame] = []
    all_signals: list[pd.DataFrame] = []
    all_trades: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []

    for fold in folds:
        fold_train_df = fold["fold_train_df"]
        fold_val_df = fold["fold_val_df"]
        metadata = fold["metadata"]

        labeled_val = fold_val_df.loc[~fold_val_df["is_context"].fillna(False)].copy()
        if len(labeled_val) < 1:
            raise ValueError(f"fold {metadata['fold_number']} has no validation samples")

        forecasts = prediction_fn(
            fold_train_df=fold_train_df,
            fold_val_df=fold_val_df,
            metadata=metadata,
            raw_train_df=fold.get("raw_train_df"),
            raw_val_df=fold.get("raw_val_df"),
        )
        forecast_df = _coerce_forecast_frame(forecasts, fallback_actuals=fold_val_df)
        signal_df = forecast_to_signal(forecast_df, mode=signal_mode, **signal_kwargs)
        trade_df, trading_metrics = simulate_trades(
            signal_df,
            horizon_h=metadata["horizon_h"],
            execution_rule=execution_rule,
            transaction_cost=transaction_cost,
            slippage=slippage,
            price_frame=price_frame if execution_rule == TRADE_EXECUTION_NEXT_OPEN_TO_H_CLOSE else None,
        )

        forecast_metrics = evaluate_forecasts(forecast_df)
        signal_metrics = _compute_signal_metrics(signal_df)

        all_forecasts.append(forecast_df)
        all_signals.append(signal_df)
        all_trades.append(trade_df)
        fold_reports.append(
            {
                "metadata": metadata,
                "forecast_metrics": forecast_metrics,
                "signal_metrics": signal_metrics,
                "trading_metrics": trading_metrics,
            }
        )

    merged_forecasts = pd.concat(all_forecasts, ignore_index=True)
    merged_signals = pd.concat(all_signals, ignore_index=True)
    merged_trades = (
        pd.concat(all_trades, ignore_index=True)
        if any(not df.empty for df in all_trades)
        else pd.DataFrame(columns=[field.name for field in TradeRecord.__dataclass_fields__.values()])
    )

    report = EvaluationReport(
        forecast_metrics=evaluate_forecasts(merged_forecasts),
        signal_metrics=_compute_signal_metrics(merged_signals),
        trading_metrics=_compute_trading_metrics(merged_trades, len(merged_signals)),
        forecast_count=len(merged_forecasts),
        signal_count=len(merged_signals),
        trade_count=len(merged_trades),
        notes=["walk_forward_validation"],
    )

    return {
        "report": asdict(report),
        "fold_reports": fold_reports,
        "forecasts": merged_forecasts,
        "signals": merged_signals,
        "trades": merged_trades,
    }


def run_final_test_evaluation(
    full_df: pd.DataFrame,
    config: SplitConfig | dict[str, Any],
    prediction_fn: Callable[..., pd.DataFrame | list[dict[str, Any]] | list[ForecastRecord]],
    *,
    signal_mode: str = "quantile_barrier",
    signal_kwargs: dict[str, Any] | None = None,
    execution_rule: str = TRADE_EXECUTION_CLOSE_TO_CLOSE,
    transaction_cost: float = 0.0,
    slippage: float = 0.0,
    price_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Run final locked test evaluation with no tuning on the test set.
    """
    split_config = _as_config(config)
    train_df, val_df, test_df, metadata = split_dataset(full_df, split_config)
    if test_df.empty:
        raise ValueError("test split contains no evaluation samples")

    signal_kwargs = signal_kwargs or {}
    forecasts = prediction_fn(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        metadata=metadata,
    )
    forecast_df = _coerce_forecast_frame(forecasts, fallback_actuals=test_df)
    signal_df = forecast_to_signal(forecast_df, mode=signal_mode, **signal_kwargs)
    trade_df, trading_metrics = simulate_trades(
        signal_df,
        horizon_h=split_config.horizon_h,
        execution_rule=execution_rule,
        transaction_cost=transaction_cost,
        slippage=slippage,
        price_frame=price_frame if execution_rule == TRADE_EXECUTION_NEXT_OPEN_TO_H_CLOSE else None,
    )

    report = EvaluationReport(
        forecast_metrics=evaluate_forecasts(forecast_df),
        signal_metrics=_compute_signal_metrics(signal_df),
        trading_metrics=trading_metrics,
        forecast_count=len(forecast_df),
        signal_count=len(signal_df),
        trade_count=len(trade_df),
        notes=["final_test_evaluation"],
    )

    return {
        "report": asdict(report),
        "metadata": metadata,
        "forecasts": forecast_df,
        "signals": signal_df,
        "trades": trade_df,
    }


__all__ = [
    "EvaluationReport",
    "ForecastRecord",
    "SignalRecord",
    "TradeRecord",
    "evaluate_forecast_frame",
    "evaluate_forecasts",
    "forecast_to_signal",
    "run_final_test_evaluation",
    "run_walk_forward_evaluation",
    "simulate_trades",
]
