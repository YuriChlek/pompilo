from __future__ import annotations

from collections import defaultdict, deque

from trading.application.runtime_models import OrderBookSnapshot
from utils.config import ORDERFLOW_BOOK_HISTORY_SIZE


class OrderBookStore:
    def __init__(self) -> None:
        self._books: dict[str, dict[str, OrderBookSnapshot]] = defaultdict(dict)
        self._history_limit = max(4, ORDERFLOW_BOOK_HISTORY_SIZE)
        self._history: dict[str, dict[str, deque[OrderBookSnapshot]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self._history_limit))
        )

    def update(self, snapshot: OrderBookSnapshot) -> None:
        self._books[snapshot.symbol][snapshot.exchange] = snapshot
        self._history[snapshot.symbol][snapshot.exchange].append(snapshot)

    def get(self, symbol: str, exchange: str) -> OrderBookSnapshot | None:
        return self._books.get(symbol, {}).get(exchange)

    def get_symbol_books(self, symbol: str) -> dict[str, OrderBookSnapshot]:
        return dict(self._books.get(symbol, {}))

    def get_history(self, symbol: str, exchange: str) -> list[OrderBookSnapshot]:
        return list(self._history.get(symbol, {}).get(exchange, ()))

    def has_fresh_book(self, symbol: str, exchange: str, now_ms: int, max_age_ms: int) -> bool:
        book = self.get(symbol, exchange)
        if book is None:
            return False
        return now_ms - book.timestamp_ms <= max_age_ms
