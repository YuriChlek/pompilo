from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
import logging
import re
import time
from uuid import uuid4

from pybit.unified_trading import HTTP
from pybit.exceptions import InvalidRequestError

from domain.models import InventorySnapshot, LiveOrder, OrderSide, TargetOrder
from domain.strategy_config import StrategyConfig
from infrastructure.bybit_account_client import BybitSpotAccountClient
from infrastructure.bybit_market_client import BybitSpotMarketClient
from infrastructure.bybit_spot_types import (
    SpotInstrumentFilters,
    UnsupportedSpotSymbolError,
    quantize_down,
    split_symbol,
)
from infrastructure.cost_basis_resolver import CostBasisResolver, calculate_cost_basis_from_executions, to_decimal
from infrastructure.execution_guardrails import apply_execution_guardrails, order_key
from utils.config import BYBIT_API_ENDPOINT, BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_RECV_WINDOW

logger = logging.getLogger(__name__)
SPOT_CATEGORY = "spot"
BOT_ORDER_LINK_ID_RE = re.compile(r"^[a-z0-9-]{1,16}-[0-9a-f]{16,32}$")
DUPLICATE_ORDER_LOOKUP_RETRIES = 4
DUPLICATE_ORDER_LOOKUP_DELAY_SECONDS = 0.35


@dataclass(slots=True)
class PaperSpotExchange:
    """In-memory exchange adapter used for paper trading and backtesting flows."""

    inventory: InventorySnapshot
    open_orders: dict[str, list[LiveOrder]] = field(default_factory=dict)
    order_sequence: int = 0

    def get_balances(self, symbol: str) -> InventorySnapshot:
        """Return the current paper inventory snapshot."""
        return self.inventory

    def get_instrument_filters(self, symbol: str) -> "SpotInstrumentFilters":
        """Return permissive instrument filters for paper-trading flows."""
        return SpotInstrumentFilters(
            tick_size=Decimal("0"),
            qty_step=Decimal("0"),
            min_order_qty=Decimal("0"),
            min_order_amt=Decimal("0"),
            max_limit_order_qty=Decimal("0"),
            max_market_order_qty=Decimal("0"),
        )

    def get_open_orders(self, symbol: str) -> list[LiveOrder]:
        """Return open paper orders for one symbol."""
        return list(self.open_orders.get(symbol.upper(), []))

    def sync_orders(self, symbol: str, target_orders: list[TargetOrder]) -> list[LiveOrder]:
        """Replace paper open orders with the requested target set for one symbol."""
        current_orders = self.open_orders.get(symbol.upper(), [])
        current_keys = {(order.side.value, round(order.price, 8), round(order.size, 8)) for order in current_orders}
        target_keys = {(order.side.value, round(order.price, 8), round(order.size, 8)) for order in target_orders if order.size > 0}

        synced_orders = [
            order
            for order in current_orders
            if (order.side.value, round(order.price, 8), round(order.size, 8)) in target_keys
        ]
        for order in target_orders:
            key = (order.side.value, round(order.price, 8), round(order.size, 8))
            if order.size > 0 and key not in current_keys:
                self.order_sequence += 1
                synced_orders.append(
                    LiveOrder(
                        order_id=f"paper-{self.order_sequence}",
                        symbol=order.symbol,
                        side=order.side,
                        price=order.price,
                        size=order.size,
                        filled_size=0.0,
                        status="OPEN",
                        client_order_id=order.client_order_id,
                    )
                )
        self.open_orders[symbol.upper()] = synced_orders
        return list(synced_orders)


class PaperExecutionService:
    """Execution adapter that applies target orders to the in-memory paper exchange."""

    def __init__(self, strategy_config: StrategyConfig) -> None:
        """Store runtime config and initialize the paper exchange inventory snapshot."""
        self.strategy_config = strategy_config
        self.exchange = PaperSpotExchange(
            inventory=InventorySnapshot(
                base_balance=strategy_config.portfolio.starting_base_balance,
                quote_balance=strategy_config.portfolio.starting_quote_balance,
                reserved_quote=0.0,
                mark_price=0.0,
            )
        )

    async def reconcile_state(self, symbols) -> None:
        """Skip reconciliation because paper execution has no external state to load."""
        return None

    async def sync_orders(self, symbol: str, target_orders: list[TargetOrder]) -> bool:
        """Apply guarded target orders to the paper exchange for one symbol."""
        guarded_orders = _apply_execution_guardrails(
            symbol,
            self.exchange.get_open_orders(symbol),
            target_orders,
            self.exchange.inventory,
            self.strategy_config,
        )
        logger.info(
            "paper_execution_sync symbol=%s requested=%s guarded=%s",
            symbol,
            len(target_orders),
            len(guarded_orders),
        )
        self.exchange.sync_orders(symbol, guarded_orders)
        return True

class BybitSpotExchange:
    """Infrastructure adapter for live Bybit spot balances, orders, and executions."""

    def __init__(self, strategy_config: StrategyConfig) -> None:
        """Initialize the Bybit HTTP client and short-lived cost-basis cache."""
        self.strategy_config = strategy_config
        self.client = HTTP(
            api_key=BYBIT_API_KEY,
            api_secret=BYBIT_API_SECRET,
            recv_window=BYBIT_RECV_WINDOW,
            demo="api-demo" in BYBIT_API_ENDPOINT,
            testnet=False,
        )
        self._cost_basis_resolver = CostBasisResolver(self._fetch_executions)
        self._unsupported_symbols: set[str] = set()
        self._market_client = BybitSpotMarketClient(self.client)
        self._account_client = BybitSpotAccountClient(
            self.client,
            fetch_current_price=self.fetch_current_price,
            resolve_cost_basis_price=self.resolve_cost_basis_price,
        )

    def get_instrument_filters(self, symbol: str) -> SpotInstrumentFilters:
        """Fetch and cache Bybit spot instrument filters for one symbol."""
        self._ensure_symbol_supported(symbol)
        return self._market_client.get_instrument_filters(symbol)

    def get_balances(self, symbol: str) -> InventorySnapshot:
        """Return current spot balances, live mark price, and derived cost basis for one symbol."""
        self._ensure_symbol_supported(symbol)
        return self._account_client.get_balances(symbol)

    def get_open_orders(self, symbol: str) -> list[LiveOrder]:
        """Return active Bybit spot orders for one symbol."""
        self._ensure_symbol_supported(symbol)
        response = self.client.get_open_orders(
            category=SPOT_CATEGORY,
            symbol=symbol.upper(),
            openOnly=0,
            orderFilter="Order",
            limit=50,
        )
        orders = ((response.get("result") or {}).get("list")) or []
        return [
            LiveOrder(
                order_id=str(order.get("orderId") or ""),
                symbol=str(order.get("symbol") or symbol.upper()),
                side=OrderSide.BUY if str(order.get("side")).lower() == "buy" else OrderSide.SELL,
                price=float(order.get("price") or 0.0),
                size=float(order.get("qty") or 0.0),
                filled_size=float(order.get("cumExecQty") or 0.0),
                status=str(order.get("orderStatus") or ""),
                client_order_id=str(order.get("orderLinkId") or ""),
            )
            for order in orders
            if str(order.get("orderId") or "")
        ]

    def fetch_current_price(self, symbol: str) -> float:
        """Return the latest Bybit spot price for one symbol."""
        self._ensure_symbol_supported(symbol)
        return self._market_client.fetch_current_price(symbol)

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel one Bybit spot order and return whether the exchange confirmed it."""
        response = self.client.cancel_order(
            category=SPOT_CATEGORY,
            symbol=symbol.upper(),
            orderId=order_id,
            orderFilter="Order",
        )
        return response.get("retCode") == 0

    def place_order(self, order: TargetOrder) -> str | None:
        """Create one normalized Bybit spot limit order and return the resulting order id."""
        normalized_qty = self._normalize_qty(order.symbol, order.size)
        normalized_price = self._normalize_price(order.symbol, order.price)
        if normalized_qty is None or normalized_price is None:
            logger.error("bybit_spot_order_rejected symbol=%s reason=invalid_qty_or_price", order.symbol)
            return None
        if not self._validate_order(order.symbol, normalized_qty, normalized_price):
            logger.error("bybit_spot_order_rejected symbol=%s reason=exchange_filters", order.symbol)
            return None
        order_link_id = _build_exchange_order_link_id(order)
        try:
            response = self.client.place_order(
                category=SPOT_CATEGORY,
                symbol=order.symbol,
                side="Buy" if order.side == OrderSide.BUY else "Sell",
                orderType="Limit",
                qty=str(normalized_qty),
                price=str(normalized_price),
                timeInForce="GTC",
                orderFilter="Order",
                orderLinkId=order_link_id,
            )
        except InvalidRequestError as exc:
            if "Duplicate clientOrderId" in str(exc):
                existing_order_id = self._wait_for_existing_order_id_by_link_id(order.symbol, order_link_id)
                if existing_order_id:
                    logger.info(
                        "bybit_spot_duplicate_client_order_recovered symbol=%s orderLinkId=%s orderId=%s",
                        order.symbol,
                        order_link_id,
                        existing_order_id,
                    )
                    return existing_order_id
            logger.error("bybit_spot_order_invalid_request symbol=%s error=%s", order.symbol, exc)
            return None
        if response.get("retCode") != 0:
            logger.error(
                "bybit_spot_order_create_failed symbol=%s retCode=%s retMsg=%s",
                order.symbol,
                response.get("retCode"),
                response.get("retMsg"),
            )
            return None
        return ((response.get("result") or {}).get("orderId")) or None

    def sync_orders(self, symbol: str, target_orders: list[TargetOrder]) -> list[LiveOrder]:
        """Synchronize live Bybit spot orders to match the requested target set."""
        current_orders = self.get_open_orders(symbol)
        normalized_targets = [order for order in (self.normalize_target_order(order) for order in target_orders) if order is not None]
        current_keys = {
            order_key(order.side, order.price, order.size): order
            for order in current_orders
            if _is_bot_managed_live_order(order)
        }
        target_keys = {
            order_key(order.side, order.price, order.size): order
            for order in normalized_targets
            if order.size > 0
        }

        for key, live_order in current_keys.items():
            if key not in target_keys:
                self.cancel_order(symbol, live_order.order_id)

        refreshed_orders = self.get_open_orders(symbol)
        refreshed_keys = {
            order_key(order.side, order.price, order.size)
            for order in refreshed_orders
            if _is_bot_managed_live_order(order)
        }
        for key, target_order in target_keys.items():
            if key not in refreshed_keys:
                self.place_order(target_order)
        return self.get_open_orders(symbol)

    def normalize_target_order(self, order: TargetOrder) -> TargetOrder | None:
        """Return a venue-normalized target order or ``None`` when normalization fails."""
        normalized_qty = self._normalize_qty(order.symbol, order.size)
        normalized_price = self._normalize_price(order.symbol, order.price)
        if normalized_qty is None or normalized_price is None:
            return None
        if not self._meets_execution_min_order_notional(order.symbol, normalized_qty, normalized_price):
            return None
        return TargetOrder(
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            price=float(normalized_price),
            size=float(normalized_qty),
            reduce_only=order.reduce_only,
            tag=order.tag,
        )

    def _normalize_price(self, symbol: str, price: float) -> Decimal | None:
        """Normalize a price to the exchange tick size for the requested symbol."""
        raw_price = to_decimal(price)
        if raw_price is None:
            return None
        tick = self.get_instrument_filters(symbol).tick_size
        return quantize_down(raw_price, tick) if tick > 0 else raw_price

    def _normalize_qty(self, symbol: str, qty: float) -> Decimal | None:
        """Normalize a quantity to the exchange quantity step and minimum size."""
        raw_qty = to_decimal(qty)
        if raw_qty is None or raw_qty <= 0:
            return None
        filters = self.get_instrument_filters(symbol)
        normalized = quantize_down(raw_qty, filters.qty_step) if filters.qty_step > 0 else raw_qty
        if normalized <= 0:
            return None
        if filters.min_order_qty > 0 and normalized < filters.min_order_qty:
            return None
        return normalized

    def _validate_order(self, symbol: str, qty: Decimal, price: Decimal) -> bool:
        """Check exchange-level notional and maximum size constraints for one order."""
        filters = self.get_instrument_filters(symbol)
        notional = qty * price
        min_order_notional = Decimal(str(self.strategy_config.execution.min_order_notional_usdt))
        if filters.min_order_amt > 0:
            min_order_notional = max(filters.min_order_amt, min_order_notional)
        if min_order_notional > 0 and notional < min_order_notional:
            return False
        if filters.max_limit_order_qty > 0 and qty > filters.max_limit_order_qty:
            return False
        return True

    def _meets_execution_min_order_notional(self, symbol: str, qty: Decimal, price: Decimal) -> bool:
        """Return whether the normalized order value is at least the effective venue-aware execution floor."""
        notional = (qty * price).quantize(Decimal("0.01"))
        filters = self.get_instrument_filters(symbol)
        minimum = Decimal(str(self.strategy_config.execution.min_order_notional_usdt))
        if filters.min_order_amt > 0:
            minimum = max(filters.min_order_amt, minimum)
        minimum = minimum.quantize(Decimal("0.01"))
        return notional >= minimum

    def _split_symbol(self, symbol: str) -> tuple[str, str]:
        """Split a spot symbol into base and quote assets using known quote suffixes."""
        return split_symbol(symbol)

    def _find_existing_order_id_by_link_id(self, symbol: str, order_link_id: str) -> str | None:
        """Return a matching live order id for a previously used ``orderLinkId``."""
        for order in self.get_open_orders(symbol):
            if order.client_order_id == order_link_id:
                return order.order_id
        return None

    def _wait_for_existing_order_id_by_link_id(self, symbol: str, order_link_id: str) -> str | None:
        """Poll open orders briefly to recover an order that became visible after a duplicate error."""
        for attempt in range(DUPLICATE_ORDER_LOOKUP_RETRIES):
            existing_order_id = self._find_existing_order_id_by_link_id(symbol, order_link_id)
            if existing_order_id:
                return existing_order_id
            if attempt < DUPLICATE_ORDER_LOOKUP_RETRIES - 1:
                time.sleep(DUPLICATE_ORDER_LOOKUP_DELAY_SECONDS)
        return None

    def mark_symbol_unsupported(self, symbol: str) -> None:
        """Remember that a symbol is not tradable on the connected Bybit spot venue."""
        self._unsupported_symbols.add(symbol.upper())

    def is_symbol_supported(self, symbol: str) -> bool:
        """Return whether the symbol is considered tradable on the connected venue."""
        return symbol.upper() not in self._unsupported_symbols

    def _ensure_symbol_supported(self, symbol: str) -> None:
        """Fail fast when a symbol was previously marked unsupported on the venue."""
        if symbol.upper() in self._unsupported_symbols:
            raise UnsupportedSpotSymbolError(f"Unsupported Bybit spot symbol: {symbol.upper()}")

    def resolve_cost_basis_price(self, symbol: str, base_balance: float, ttl_seconds: int = 30) -> float | None:
        """Derive spot cost basis from recent executions and cache it briefly."""
        return self._cost_basis_resolver.resolve(symbol, base_balance, ttl_seconds=ttl_seconds)

    def _fetch_executions(self, symbol: str) -> list[dict]:
        """Fetch recent spot executions for one symbol."""
        response = self.client.get_executions(
            category=SPOT_CATEGORY,
            symbol=symbol.upper(),
            limit=100,
        )
        return ((response.get("result") or {}).get("list")) or []


class BybitSpotExecutionService:
    """Execution adapter that applies guarded target orders to the live Bybit spot venue."""

    def __init__(self, strategy_config: StrategyConfig) -> None:
        """Store runtime configuration and initialize the live Bybit exchange adapter."""
        self.strategy_config = strategy_config
        self.exchange = BybitSpotExchange(strategy_config)

    async def reconcile_state(self, symbols) -> None:
        """Warm exchange caches and verify balances and orders for configured symbols."""
        for symbol in symbols:
            normalized_symbol = str(symbol).upper()
            try:
                self.exchange.get_open_orders(normalized_symbol)
                self.exchange.get_balances(normalized_symbol)
            except InvalidRequestError as exc:
                if "Not supported symbols" in str(exc):
                    self.exchange.mark_symbol_unsupported(normalized_symbol)
                    logger.warning("bybit_spot_symbol_unsupported symbol=%s", normalized_symbol)
                    continue
                raise

    def is_symbol_supported(self, symbol: str) -> bool:
        """Return whether the symbol passed Bybit spot support checks."""
        return self.exchange.is_symbol_supported(symbol)

    async def sync_orders(self, symbol: str, target_orders: list[TargetOrder]) -> bool:
        """Apply guarded target orders to Bybit spot for one symbol."""
        inventory = self.exchange.get_balances(symbol)
        current_orders = self.exchange.get_open_orders(symbol)
        guarded_orders = apply_execution_guardrails(
            symbol,
            current_orders,
            target_orders,
            inventory,
            self.strategy_config,
        )
        logger.info(
            "bybit_spot_execution_sync symbol=%s current=%s requested=%s guarded=%s",
            symbol,
            len(current_orders),
            len(target_orders),
            len(guarded_orders),
        )
        normalized_guarded = [
            order
            for order in (self.exchange.normalize_target_order(order) for order in guarded_orders)
            if order is not None
        ]
        logger.info(
            "bybit_spot_execution_sync_normalized symbol=%s guarded=%s normalized=%s",
            symbol,
            len(guarded_orders),
            len(normalized_guarded),
        )
        if target_orders and guarded_orders and not normalized_guarded:
            logger.warning(
                "bybit_spot_execution_sync_blocked_by_venue_filters symbol=%s requested=%s guarded=%s normalized=%s",
                symbol,
                len(target_orders),
                len(guarded_orders),
                len(normalized_guarded),
            )
            return False
        final_orders = self.exchange.sync_orders(symbol, guarded_orders)
        expected_keys = {order_key(order.side, order.price, order.size) for order in normalized_guarded}
        actual_keys = {
            order_key(order.side, order.price, order.size)
            for order in final_orders
            if _is_bot_managed_live_order(order)
        }
        sync_succeeded = expected_keys == actual_keys
        if not sync_succeeded:
            logger.warning(
                "bybit_spot_execution_sync_incomplete symbol=%s expected=%s actual=%s",
                symbol,
                len(expected_keys),
                len(actual_keys),
            )
        else:
            logger.info(
                "bybit_spot_execution_sync_complete symbol=%s expected=%s actual=%s",
                symbol,
                len(expected_keys),
                len(actual_keys),
            )
        return sync_succeeded


def _apply_execution_guardrails(
    symbol: str,
    current_orders: list[LiveOrder],
    target_orders: list[TargetOrder],
    inventory: InventorySnapshot,
    strategy_config: StrategyConfig,
) -> list[TargetOrder]:
    """Backward-compatible wrapper around the extracted execution guardrails module."""
    return apply_execution_guardrails(symbol, current_orders, target_orders, inventory, strategy_config)


def _is_bot_managed_order_link_id(client_order_id: str) -> bool:
    """Return whether a live ``orderLinkId`` looks like one generated by this bot."""
    normalized = (client_order_id or "").strip().lower()
    return bool(normalized) and BOT_ORDER_LINK_ID_RE.match(normalized) is not None


def _is_bot_managed_live_order(order: LiveOrder) -> bool:
    """Return whether a live order appears to be managed by this bot."""
    return _is_bot_managed_order_link_id(order.client_order_id)


def _build_exchange_order_link_id(order: TargetOrder) -> str:
    """Build a unique ``orderLinkId`` for Bybit order creation."""
    base = order.client_order_id.replace("_", "-")[:16]
    unique_suffix = uuid4().hex[:20]
    return f"{base}-{unique_suffix}"[:36]


def _calculate_cost_basis_from_executions(executions: list[dict], base_balance: float) -> float | None:
    """Backward-compatible wrapper around the extracted cost-basis resolver helper."""
    return calculate_cost_basis_from_executions(executions, base_balance)
