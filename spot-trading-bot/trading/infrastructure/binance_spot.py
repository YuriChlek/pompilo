from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from decimal import Decimal
from decimal import ROUND_DOWN
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import requests

from utils.config import BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_REST_ENDPOINT


def _sign(query: str) -> str:
    return hmac.new(BINANCE_API_SECRET.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class BinanceSymbolFilters:
    symbol: str
    min_qty: Decimal
    max_qty: Decimal
    step_size: Decimal
    min_notional: Decimal
    tick_size: Decimal


@dataclass(frozen=True)
class BinanceBalance:
    asset: str
    free: Decimal
    locked: Decimal

    @property
    def total(self) -> Decimal:
        return self.free + self.locked


def _as_decimal(value: Any, default: str = "0") -> Decimal:
    if value in (None, ""):
        return Decimal(default)
    return Decimal(str(value))


def _floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def normalize_order_quantity(quantity: Decimal, filters: BinanceSymbolFilters) -> Decimal:
    normalized = _floor_to_step(quantity, filters.step_size)
    if normalized < filters.min_qty:
        return Decimal("0")
    if filters.max_qty > 0 and normalized > filters.max_qty:
        normalized = _floor_to_step(filters.max_qty, filters.step_size)
    return normalized.normalize()


def satisfies_min_notional(quantity: Decimal, reference_price: Decimal, filters: BinanceSymbolFilters) -> bool:
    if filters.min_notional <= 0:
        return True
    return quantity * reference_price >= filters.min_notional


class BinanceSpotClient:
    def __init__(self, endpoint: str = BINANCE_REST_ENDPOINT) -> None:
        self.endpoint = endpoint
        self.session = requests.Session()
        self._symbol_filters_cache: dict[str, BinanceSymbolFilters] = {}
        if BINANCE_API_KEY:
            self.session.headers.update({"X-MBX-APIKEY": BINANCE_API_KEY})

    def _signed_request(self, method: str, path: str, params: dict[str, Any]) -> dict[str, Any]:
        if not BINANCE_API_KEY or not BINANCE_API_SECRET:
            raise RuntimeError("BINANCE_API_KEY/BINANCE_API_SECRET are required for spot execution")

        payload = {**params, "timestamp": int(time.time() * 1000)}
        query = urlencode(payload, doseq=True)
        payload["signature"] = _sign(query)
        response = self.session.request(method, f"{self.endpoint}{path}", params=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    def fetch_current_price(self, symbol: str) -> Decimal:
        response = self.session.get(f"{self.endpoint}/api/v3/ticker/price", params={"symbol": symbol}, timeout=30)
        response.raise_for_status()
        return Decimal(str(response.json()["price"]))

    def get_symbol_filters(self, symbol: str) -> BinanceSymbolFilters:
        normalized_symbol = symbol.upper()
        cached = self._symbol_filters_cache.get(normalized_symbol)
        if cached is not None:
            return cached

        response = self.session.get(
            f"{self.endpoint}/api/v3/exchangeInfo",
            params={"symbol": normalized_symbol},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        symbols = payload.get("symbols") or []
        if not symbols:
            raise RuntimeError(f"Binance exchangeInfo returned no metadata for {normalized_symbol}")

        symbol_payload = symbols[0]
        filters_by_type = {item.get("filterType"): item for item in symbol_payload.get("filters", [])}
        lot_size = filters_by_type.get("LOT_SIZE", {})
        min_notional = filters_by_type.get("MIN_NOTIONAL", {})
        notional = filters_by_type.get("NOTIONAL", {})
        price_filter = filters_by_type.get("PRICE_FILTER", {})
        resolved = BinanceSymbolFilters(
            symbol=normalized_symbol,
            min_qty=_as_decimal(lot_size.get("minQty")),
            max_qty=_as_decimal(lot_size.get("maxQty")),
            step_size=_as_decimal(lot_size.get("stepSize"), "1"),
            min_notional=_as_decimal(notional.get("minNotional") or min_notional.get("minNotional")),
            tick_size=_as_decimal(price_filter.get("tickSize"), "0"),
        )
        self._symbol_filters_cache[normalized_symbol] = resolved
        return resolved

    def place_market_order(self, symbol: str, side: str, quantity: Decimal) -> dict[str, Any]:
        return self._signed_request(
            "POST",
            "/api/v3/order",
            {
                "symbol": symbol,
                "side": side.upper(),
                "type": "MARKET",
                "quantity": format(quantity, "f"),
            },
        )

    def fetch_account_balances(self) -> dict[str, BinanceBalance]:
        payload = self._signed_request("GET", "/api/v3/account", {})
        balances = {}
        for item in payload.get("balances", []):
            balance = BinanceBalance(
                asset=str(item.get("asset", "")),
                free=_as_decimal(item.get("free")),
                locked=_as_decimal(item.get("locked")),
            )
            balances[balance.asset] = balance
        return balances

    def fetch_asset_balance(self, asset: str) -> BinanceBalance:
        balances = self.fetch_account_balances()
        return balances.get(asset.upper(), BinanceBalance(asset.upper(), Decimal("0"), Decimal("0")))

    def fetch_my_trades(self, symbol: str, limit: int = 1000) -> list[dict[str, Any]]:
        payload = self._signed_request(
            "GET",
            "/api/v3/myTrades",
            {
                "symbol": symbol.upper(),
                "limit": limit,
            },
        )
        return list(payload)

    @staticmethod
    def extract_fill_price(order_payload: dict[str, Any], fallback_price: Decimal) -> Decimal:
        fills = order_payload.get("fills") or []
        if fills:
            total_qty = Decimal("0")
            total_cost = Decimal("0")
            for fill in fills:
                qty = Decimal(str(fill.get("qty", "0")))
                price = Decimal(str(fill.get("price", "0")))
                total_qty += qty
                total_cost += qty * price
            if total_qty > 0:
                return total_cost / total_qty

        executed_qty = Decimal(str(order_payload.get("executedQty", "0")))
        cummulative_quote_qty = Decimal(str(order_payload.get("cummulativeQuoteQty", "0")))
        if executed_qty > 0 and cummulative_quote_qty > 0:
            return cummulative_quote_qty / executed_qty
        return fallback_price

    @staticmethod
    def dumps_payload(payload: Optional[Dict[str, Any]]) -> str:
        return json.dumps(payload or {}, ensure_ascii=True, sort_keys=True)


def base_asset_from_symbol(symbol: str, quote_asset: str = "USDT") -> str:
    normalized_symbol = symbol.upper()
    normalized_quote = quote_asset.upper()
    if normalized_symbol.endswith(normalized_quote):
        return normalized_symbol[: -len(normalized_quote)]
    raise ValueError(f"Unsupported symbol format for quote asset {quote_asset}: {symbol}")


def derive_avg_entry_price_from_trades(symbol: str, current_quantity: Decimal, trades: list[dict[str, Any]]) -> Decimal:
    if current_quantity <= 0:
        return Decimal("0")

    remaining_lots: list[list[Decimal]] = []
    for trade in sorted(trades, key=lambda item: (int(item.get("time", 0)), int(item.get("id", 0)))):
        qty = _as_decimal(trade.get("qty"))
        price = _as_decimal(trade.get("price"))
        is_buyer = bool(trade.get("isBuyer"))

        if qty <= 0 or price <= 0:
            continue

        if is_buyer:
            remaining_lots.append([qty, price])
            continue

        qty_to_close = qty
        while qty_to_close > 0 and remaining_lots:
            head_qty, head_price = remaining_lots[0]
            consumed = min(head_qty, qty_to_close)
            head_qty -= consumed
            qty_to_close -= consumed
            if head_qty <= 0:
                remaining_lots.pop(0)
            else:
                remaining_lots[0][0] = head_qty

    total_qty = sum((lot_qty for lot_qty, _ in remaining_lots), Decimal("0"))
    if total_qty <= 0:
        return Decimal("0")

    target_qty = min(current_quantity, total_qty)
    qty_left = target_qty
    total_cost = Decimal("0")
    for lot_qty, lot_price in reversed(remaining_lots):
        if qty_left <= 0:
            break
        taken = min(lot_qty, qty_left)
        total_cost += taken * lot_price
        qty_left -= taken

    accounted_qty = target_qty - qty_left
    if accounted_qty <= 0:
        return Decimal("0")
    return total_cost / accounted_qty
