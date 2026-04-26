from __future__ import annotations

from typing import TYPE_CHECKING

from domain.models import SpotSignal
from utils.config import BUY_SIGNAL, HOLD_SIGNAL, SELL_SIGNAL

if TYPE_CHECKING:
    from indicators import GreenwichSignalSnapshot


def build_greenwich_signal_snapshot(candles_df) -> GreenwichSignalSnapshot:
    """Return the latest Greenwich indicator snapshot for one candle history frame."""

    from indicators import build_greenwich_snapshot

    return build_greenwich_snapshot(candles_df)


def generate_spot_signal(symbol: str, candles_df) -> SpotSignal:
    """Build the trading signal for one symbol from the Greenwich strategy snapshot."""

    snapshot = build_greenwich_signal_snapshot(candles_df)
    if snapshot.buy_signal:
        return SpotSignal(
            symbol=symbol,
            signal_type=BUY_SIGNAL,
            signal_price=snapshot.signal_price,
            close_time=snapshot.close_time,
            reason="greenwich_buy_recovery",
        )
    if snapshot.sell_signal:
        return SpotSignal(
            symbol=symbol,
            signal_type=SELL_SIGNAL,
            signal_price=snapshot.signal_price,
            close_time=snapshot.close_time,
            reason="greenwich_sell_fade",
        )
    return SpotSignal(
        symbol=symbol,
        signal_type=HOLD_SIGNAL,
        signal_price=snapshot.signal_price,
        close_time=snapshot.close_time,
        reason="no_greenwich_signal",
    )


__all__ = ["build_greenwich_signal_snapshot", "generate_spot_signal"]
