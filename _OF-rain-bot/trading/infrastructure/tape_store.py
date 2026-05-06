from __future__ import annotations

from collections import defaultdict, deque

from trading.application.runtime_models import TapeWindowStats, TradePrint


class TapeStore:
    def __init__(self) -> None:
        self._trades: dict[str, dict[str, deque[TradePrint]]] = defaultdict(lambda: defaultdict(deque))

    def append(self, trade: TradePrint) -> None:
        self._trades[trade.symbol][trade.exchange].append(trade)

    def prune(self, symbol: str, exchange: str, now_ms: int, max_window_ms: int) -> None:
        trades = self._trades[symbol][exchange]
        while trades and now_ms - trades[0].timestamp_ms > max_window_ms:
            trades.popleft()

    def get_window_stats(self, symbol: str, exchange: str, now_ms: int, window_ms: int) -> TapeWindowStats:
        self.prune(symbol, exchange, now_ms, window_ms)
        trades = self._trades.get(symbol, {}).get(exchange, deque())
        buy_notional = 0.0
        sell_notional = 0.0
        buy_count = 0
        sell_count = 0
        last_price = None

        for trade in trades:
            if now_ms - trade.timestamp_ms > window_ms:
                continue
            last_price = trade.price
            if trade.side == "buy":
                buy_notional += trade.notional
                buy_count += 1
            else:
                sell_notional += trade.notional
                sell_count += 1

        return TapeWindowStats(
            symbol=symbol,
            exchange=exchange,
            window_ms=window_ms,
            buy_notional=buy_notional,
            sell_notional=sell_notional,
            buy_count=buy_count,
            sell_count=sell_count,
            last_price=last_price,
            exchange_count=1,
        )

    def get_aggregated_window_stats(
        self,
        symbol: str,
        now_ms: int,
        window_ms: int,
        exchanges: tuple[str, ...] | list[str] | None = None,
    ) -> TapeWindowStats:
        symbol_trades = self._trades.get(symbol, {})
        exchange_names = tuple(symbol_trades.keys()) if exchanges is None else tuple(exchanges)
        buy_notional = 0.0
        sell_notional = 0.0
        buy_count = 0
        sell_count = 0
        last_price = None
        active_exchanges = 0

        for exchange in exchange_names:
            if exchange not in symbol_trades:
                continue
            stats = self.get_window_stats(symbol, exchange, now_ms, window_ms)
            if stats.buy_count == 0 and stats.sell_count == 0 and stats.last_price is None:
                continue
            active_exchanges += 1
            buy_notional += stats.buy_notional
            sell_notional += stats.sell_notional
            buy_count += stats.buy_count
            sell_count += stats.sell_count
            last_price = stats.last_price if stats.last_price is not None else last_price

        return TapeWindowStats(
            symbol=symbol,
            exchange=None,
            window_ms=window_ms,
            buy_notional=buy_notional,
            sell_notional=sell_notional,
            buy_count=buy_count,
            sell_count=sell_count,
            last_price=last_price,
            exchange_count=max(active_exchanges, 1 if exchange_names else 0),
        )
