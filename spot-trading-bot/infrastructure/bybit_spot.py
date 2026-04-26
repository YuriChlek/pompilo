from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from functools import lru_cache
from typing import Any

from utils.config import BYBIT_API_ENDPOINT, BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_RECV_WINDOW

SPOT_CATEGORY = "spot"
KNOWN_QUOTE_ASSETS = ("USDT", "USDC", "BTC", "ETH", "EUR")


@dataclass(frozen=True)
class BybitSpotFilters:
    symbol: str
    min_qty: Decimal
    max_qty: Decimal
    step_size: Decimal
    min_notional: Decimal
    tick_size: Decimal


@dataclass(frozen=True)
class BybitBalance:
    asset: str
    free: Decimal
    locked: Decimal

    @property
    def total(self) -> Decimal:
        return self.free + self.locked


def to_decimal(value: Any, default: str = "0") -> Decimal:
    if value in (None, ""):
        return Decimal(default)
    return Decimal(str(value))


def split_symbol(symbol: str) -> tuple[str, str]:
    normalized = symbol.upper()
    for quote_asset in KNOWN_QUOTE_ASSETS:
        if normalized.endswith(quote_asset) and len(normalized) > len(quote_asset):
            return normalized[: -len(quote_asset)], quote_asset
    raise ValueError(f"Unsupported spot symbol format: {symbol}")


def _floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def normalize_order_quantity(quantity: Decimal, filters: BybitSpotFilters) -> Decimal:
    normalized = _floor_to_step(quantity, filters.step_size)
    if normalized < filters.min_qty:
        return Decimal("0")
    if filters.max_qty > 0 and normalized > filters.max_qty:
        normalized = _floor_to_step(filters.max_qty, filters.step_size)
    return normalized.normalize()


def satisfies_min_notional(quantity: Decimal, reference_price: Decimal, filters: BybitSpotFilters) -> bool:
    if filters.min_notional <= 0:
        return True
    return quantity * reference_price >= filters.min_notional


def derive_avg_entry_price_from_trades(symbol: str, current_quantity: Decimal, trades: list[dict[str, Any]]) -> Decimal:
    if current_quantity <= 0:
        return Decimal("0")

    remaining_lots: list[list[Decimal]] = []
    for trade in sorted(trades, key=lambda item: (int(item.get("execTime", 0)), str(item.get("execId", "")))):
        qty = to_decimal(trade.get("execQty"))
        price = to_decimal(trade.get("execPrice"))
        side = str(trade.get("side", "")).lower()

        if qty <= 0 or price <= 0:
            continue

        if side == "buy":
            remaining_lots.append([qty, price])
            continue

        qty_to_close = qty
        while qty_to_close > 0 and remaining_lots:
            head_qty, _ = remaining_lots[0]
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


class BybitSpotClient:
    def __init__(self) -> None:
        from pybit.unified_trading import HTTP

        self.client = HTTP(
            api_key=BYBIT_API_KEY,
            api_secret=BYBIT_API_SECRET,
            recv_window=BYBIT_RECV_WINDOW,
            demo="api-demo" in BYBIT_API_ENDPOINT,
            testnet=False,
        )

    @lru_cache(maxsize=256)
    def get_symbol_filters(self, symbol: str) -> BybitSpotFilters:
        response = self.client.get_instruments_info(category=SPOT_CATEGORY, symbol=symbol.upper())
        instruments = ((response.get("result") or {}).get("list")) or []
        if not instruments:
            raise RuntimeError(f"Bybit returned no instrument metadata for {symbol}")
        instrument = instruments[0]
        price_filter = instrument.get("priceFilter") or {}
        lot_size_filter = instrument.get("lotSizeFilter") or {}
        step_size = to_decimal(lot_size_filter.get("basePrecision"), "1")
        return BybitSpotFilters(
            symbol=symbol.upper(),
            min_qty=to_decimal(lot_size_filter.get("minOrderQty")),
            max_qty=to_decimal(lot_size_filter.get("maxLimitOrderQty")),
            step_size=step_size,
            min_notional=to_decimal(lot_size_filter.get("minOrderAmt")),
            tick_size=to_decimal(price_filter.get("tickSize")),
        )

    def fetch_current_price(self, symbol: str) -> Decimal:
        response = self.client.get_tickers(category=SPOT_CATEGORY, symbol=symbol.upper())
        tickers = ((response.get("result") or {}).get("list")) or []
        if not tickers:
            raise RuntimeError(f"Bybit returned no ticker for {symbol}")
        return Decimal(str(tickers[0].get("lastPrice") or "0"))

    def fetch_asset_balance(self, asset: str) -> BybitBalance:
        response = self.client.get_wallet_balance(accountType="UNIFIED")
        accounts = ((response.get("result") or {}).get("list")) or []
        if not accounts:
            return BybitBalance(asset.upper(), Decimal("0"), Decimal("0"))
        coins = {str(item.get("coin", "")).upper(): item for item in (accounts[0].get("coin") or [])}
        coin = coins.get(asset.upper(), {})
        return BybitBalance(
            asset.upper(),
            to_decimal(coin.get("walletBalance")),
            to_decimal(coin.get("locked")),
        )

    def fetch_my_trades(self, symbol: str, limit: int = 100) -> list[dict[str, Any]]:
        response = self.client.get_executions(category=SPOT_CATEGORY, symbol=symbol.upper(), limit=limit)
        return list(((response.get("result") or {}).get("list")) or [])

    def place_market_order(self, symbol: str, side: str, quantity: Decimal) -> dict[str, Any]:
        response = self.client.place_order(
            category=SPOT_CATEGORY,
            symbol=symbol.upper(),
            side="Buy" if side.upper() == "BUY" else "Sell",
            orderType="Market",
            qty=format(quantity, "f"),
            marketUnit="baseCoin",
        )
        if response.get("retCode") != 0:
            raise RuntimeError(f"Bybit order failed: retCode={response.get('retCode')} retMsg={response.get('retMsg')}")
        return response

    @staticmethod
    def extract_fill_price(order_payload: dict[str, Any], fallback_price: Decimal) -> Decimal:
        result = order_payload.get("result") or {}
        avg_price = result.get("avgPrice")
        if avg_price not in (None, "", "0"):
            return Decimal(str(avg_price))
        return fallback_price
