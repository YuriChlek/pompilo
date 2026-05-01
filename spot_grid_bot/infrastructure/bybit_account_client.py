from __future__ import annotations

import logging

from domain.models import InventorySnapshot
from infrastructure.bybit_spot_types import split_symbol

SPOT_CATEGORY = "spot"
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
