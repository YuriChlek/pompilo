from __future__ import annotations

from domain.models import InventorySnapshot
from infrastructure.bybit_spot_types import split_symbol

SPOT_CATEGORY = "spot"


class BybitSpotAccountClient:
    """Balance and inventory snapshot access for Bybit spot."""

    def __init__(self, http_client, fetch_current_price, resolve_cost_basis_price) -> None:
        self.client = http_client
        self._fetch_current_price = fetch_current_price
        self._resolve_cost_basis_price = resolve_cost_basis_price

    def get_balances(self, symbol: str) -> InventorySnapshot:
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
        return InventorySnapshot(
            base_balance=base_balance,
            quote_balance=float(quote_coin.get("walletBalance") or 0.0),
            reserved_quote=float(quote_coin.get("locked") or 0.0),
            mark_price=mark_price,
            cost_basis_price=self._resolve_cost_basis_price(symbol, base_balance),
        )
