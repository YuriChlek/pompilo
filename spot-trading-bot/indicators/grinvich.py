from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from utils.config import GREENWICH_BASIS_TYPE, GREENWICH_LENGTH, GREENWICH_MULTIPLIER_1, GREENWICH_MULTIPLIER_2, GREENWICH_MULTIPLIER_3


@dataclass(frozen=True)
class GreenwichSignalSnapshot:
    basis: Decimal
    upper1: Decimal
    upper2: Decimal
    upper3: Decimal
    lower1: Decimal
    lower2: Decimal
    lower3: Decimal
    buy_signal: bool
    sell_signal: bool
    signal_price: Decimal
    close_time: object


def _rma(series: pd.Series, length: int) -> pd.Series:
    alpha = 1 / length
    return series.ewm(alpha=alpha, adjust=False).mean()


def _wma(series: pd.Series, length: int) -> pd.Series:
    weights = pd.Series(range(1, length + 1), dtype="float64")
    return series.rolling(length).apply(lambda values: (values * weights).sum() / weights.sum(), raw=True)


def _basis(series: pd.Series, ma_type: str, length: int) -> pd.Series:
    normalized = ma_type.upper()
    if normalized == "EMA":
        return series.ewm(span=length, adjust=False).mean()
    if normalized == "SMA":
        return series.rolling(length).mean()
    if normalized == "RMA":
        return _rma(series, length)
    return _wma(series, length)


def _to_decimal(value: object) -> Decimal:
    return Decimal(str(value))


def _crossover(source: pd.Series, reference: pd.Series) -> pd.Series:
    return (source.shift(1) <= reference.shift(1)) & (source > reference)


def _crossunder(source: pd.Series, reference: pd.Series) -> pd.Series:
    return (source.shift(1) >= reference.shift(1)) & (source < reference)


def build_greenwich_snapshot(df: pd.DataFrame) -> GreenwichSignalSnapshot:
    if len(df) < GREENWICH_LENGTH + 2:
        raise ValueError("Недостатньо історії для Greenwich D1")

    working = df.copy().reset_index(drop=True)
    working["high"] = working["high"].astype("float64")
    working["low"] = working["low"].astype("float64")
    working["close"] = working["close"].astype("float64")

    basis = _basis(working["close"], GREENWICH_BASIS_TYPE, GREENWICH_LENGTH)
    prev_close = working["close"].shift(1)
    true_range = pd.concat(
        [
            (working["high"] - working["low"]).abs(),
            (working["high"] - prev_close).abs(),
            (working["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = _rma(true_range.astype("float64"), GREENWICH_LENGTH)

    working["basis"] = basis
    working["upper1"] = basis + float(GREENWICH_MULTIPLIER_1) * atr
    working["upper2"] = basis + float(GREENWICH_MULTIPLIER_2) * atr
    working["upper3"] = basis + float(GREENWICH_MULTIPLIER_3) * atr
    working["lower1"] = basis - float(GREENWICH_MULTIPLIER_1) * atr
    working["lower2"] = basis - float(GREENWICH_MULTIPLIER_2) * atr
    working["lower3"] = basis - float(GREENWICH_MULTIPLIER_3) * atr

    buy_cross = _crossover(working["low"], working["lower3"])
    sell_cross = _crossunder(working["close"], working["upper2"])
    last = working.iloc[-1]

    return GreenwichSignalSnapshot(
        basis=_to_decimal(last["basis"]),
        upper1=_to_decimal(last["upper1"]),
        upper2=_to_decimal(last["upper2"]),
        upper3=_to_decimal(last["upper3"]),
        lower1=_to_decimal(last["lower1"]),
        lower2=_to_decimal(last["lower2"]),
        lower3=_to_decimal(last["lower3"]),
        buy_signal=bool(buy_cross.iloc[-1]),
        sell_signal=bool(sell_cross.iloc[-1]),
        signal_price=_to_decimal(last["close"]),
        close_time=last["close_time"],
    )
