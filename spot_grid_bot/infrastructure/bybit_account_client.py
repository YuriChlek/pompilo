from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging

from infrastructure.cost_basis_resolver import calculate_cost_basis_from_executions
from domain.models import InventorySnapshot
from infrastructure.bybit_spot_types import split_symbol

SPOT_CATEGORY = "spot"
EXECUTION_HISTORY_WINDOW_DAYS = 7
EXECUTION_HISTORY_LOOKBACK_DAYS = 180
EXECUTION_HISTORY_PAGE_LIMIT = 100
logger = logging.getLogger(__name__)


class BybitSpotAccountClient:
    """Balance and inventory snapshot access for Bybit spot."""

    def __init__(self, http_client, fetch_current_price) -> None:
        self.client = http_client
        self._fetch_current_price = fetch_current_price

    def get_balances(self, symbol: str, *, persisted_cost_basis: float | None = None) -> InventorySnapshot:
        """Return current spot balances, live mark price, and derived cost basis for one symbol."""
        base_asset, quote_asset = split_symbol(symbol)
        response = self.client.get_wallet_balance(accountType="UNIFIED")
        accounts = ((response.get("result") or {}).get("list")) or []
        if not accounts:
            raise ValueError("Missing unified wallet balance payload")
        coins = {item.get("coin"): item for item in (accounts[0].get("coin") or [])}
        base_coin = coins.get(base_asset, {})
        quote_coin = coins.get(quote_asset, {})
        mark_price = self._fetch_current_price(symbol)
        base_balance = float(base_coin.get("walletBalance") or 0.0)
        cost_basis_price: float | None = None
        avg_price_raw = base_coin.get("avgPrice")
        if avg_price_raw is not None:
            avg_price = float(avg_price_raw)
            if avg_price > 0 and base_balance > 0:
                cost_basis_price = avg_price

        if cost_basis_price is None and base_balance > 0:
            cost_basis_price = self._resolve_cost_basis_from_execution_history(symbol, base_balance)
            if cost_basis_price is not None:
                logger.info(
                    "cost_basis_restored_from_executions symbol=%s cost_basis=%.4f",
                    symbol.upper(),
                    cost_basis_price,
                )

        if cost_basis_price is None and persisted_cost_basis is not None and persisted_cost_basis > 0 and base_balance > 0:
            cost_basis_price = persisted_cost_basis
            logger.debug(
                "cost_basis_fallback_to_persisted symbol=%s persisted=%.4f",
                symbol.upper(),
                persisted_cost_basis,
            )

        if base_balance <= 0:
            cost_basis_price = None
        elif cost_basis_price is None:
            logger.warning(
                "cost_basis_unavailable symbol=%s base_balance=%.6f sells_blocked=true",
                symbol.upper(),
                base_balance,
            )

        return InventorySnapshot(
            base_balance=base_balance,
            quote_balance=float(quote_coin.get("walletBalance") or 0.0),
            reserved_quote=float(quote_coin.get("locked") or 0.0),
            mark_price=mark_price,
            cost_basis_price=cost_basis_price,
        )

    def _resolve_cost_basis_from_execution_history(self, symbol: str, base_balance: float) -> float | None:
        """Rebuild remaining spot inventory average cost from execution history when available."""
        if base_balance <= 0:
            return None
        executions = self._fetch_execution_history(symbol)
        if not executions:
            return None
        return calculate_cost_basis_from_executions(executions, base_balance)

    def _fetch_execution_history(self, symbol: str) -> list[dict]:
        """Fetch recent execution history across Bybit's seven-day query windows."""
        submit_request = getattr(self.client, "_submit_request", None)
        endpoint = getattr(self.client, "endpoint", None)
        if submit_request is None or endpoint is None:
            return []

        all_executions: dict[str, dict] = {}
        now = datetime.now(timezone.utc)
        total_windows = max(EXECUTION_HISTORY_LOOKBACK_DAYS // EXECUTION_HISTORY_WINDOW_DAYS, 1)
        for window_index in range(total_windows):
            end = now - timedelta(days=EXECUTION_HISTORY_WINDOW_DAYS * window_index)
            start = end - timedelta(days=EXECUTION_HISTORY_WINDOW_DAYS)
            cursor: str | None = None

            while True:
                response = submit_request(
                    method="GET",
                    path=f"{endpoint}/v5/execution/list",
                    query={
                        "category": SPOT_CATEGORY,
                        "symbol": symbol.upper(),
                        "startTime": int(start.timestamp() * 1000),
                        "endTime": int(end.timestamp() * 1000),
                        "limit": EXECUTION_HISTORY_PAGE_LIMIT,
                        "cursor": cursor,
                    },
                    auth=True,
                )
                result = response.get("result") or {}
                items = result.get("list") or []
                for item in items:
                    execution_id = str(item.get("execId") or "")
                    fallback_id = f"{item.get('orderId')}:{item.get('execTime')}:{item.get('execQty')}"
                    all_executions[execution_id or fallback_id] = item

                cursor = result.get("nextPageCursor") or None
                if not cursor or not items:
                    break

        return sorted(
            all_executions.values(),
            key=lambda item: int(item.get("execTime") or 0),
            reverse=True,
        )
