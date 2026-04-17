from __future__ import annotations

from trading.domain.models import SpotSignal
from indicators import build_greenwich_snapshot
from utils.config import BUY_SIGNAL, HOLD_SIGNAL, SELL_SIGNAL


def generate_spot_signal(symbol: str, candles_df) -> SpotSignal:
    snapshot = build_greenwich_snapshot(candles_df)
    if snapshot.buy_signal:
        return SpotSignal(symbol=symbol, signal_type=BUY_SIGNAL, signal_price=snapshot.signal_price, close_time=snapshot.close_time, reason="greenwich_buy_recovery")
    if snapshot.sell_signal:
        return SpotSignal(symbol=symbol, signal_type=SELL_SIGNAL, signal_price=snapshot.signal_price, close_time=snapshot.close_time, reason="greenwich_sell_fade")
    return SpotSignal(symbol=symbol, signal_type=HOLD_SIGNAL, signal_price=snapshot.signal_price, close_time=snapshot.close_time, reason="no_greenwich_signal")
