from __future__ import annotations

import time

from trading.application.runtime_models import BookLevel, OrderBookSnapshot, TradePrint
from utils.config import ORDERFLOW_BOOK_DEPTH, ORDERFLOW_TICK_SIZES


def build_book_snapshot(
    exchange: str,
    symbol: str,
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    timestamp_ms: int | None = None,
) -> OrderBookSnapshot:
    bids = sorted(((price, size) for price, size in bids if size > 0), key=lambda item: item[0], reverse=True)[:ORDERFLOW_BOOK_DEPTH]
    asks = sorted(((price, size) for price, size in asks if size > 0), key=lambda item: item[0])[:ORDERFLOW_BOOK_DEPTH]
    if not bids or not asks:
        raise ValueError(f"Invalid book for {exchange}:{symbol}")

    tick_size = ORDERFLOW_TICK_SIZES.get(symbol, 0.01)
    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid_price = (best_bid + best_ask) / 2
    spread_ticks = max(1, int(round((best_ask - best_bid) / tick_size)))

    bid_levels = [
        BookLevel(
            price=price,
            size=size,
            notional=price * size,
            distance_ticks=max(0, int(round((best_bid - price) / tick_size))),
            distance_bps=abs(mid_price - price) / mid_price * 10000,
        )
        for price, size in bids
    ]
    ask_levels = [
        BookLevel(
            price=price,
            size=size,
            notional=price * size,
            distance_ticks=max(0, int(round((price - best_ask) / tick_size))),
            distance_bps=abs(price - mid_price) / mid_price * 10000,
        )
        for price, size in asks
    ]

    return OrderBookSnapshot(
        exchange=exchange,
        symbol=symbol,
        timestamp_ms=timestamp_ms or int(time.time() * 1000),
        bids=bid_levels,
        asks=ask_levels,
        best_bid=best_bid,
        best_ask=best_ask,
        mid_price=mid_price,
        spread_ticks=spread_ticks,
        tick_size=tick_size,
    )


def build_trade(exchange: str, symbol: str, side: str, price: float, size: float, timestamp_ms: int) -> TradePrint:
    return TradePrint(
        exchange=exchange,
        symbol=symbol,
        timestamp_ms=timestamp_ms,
        price=price,
        size=size,
        side=side,
        notional=price * size,
    )
