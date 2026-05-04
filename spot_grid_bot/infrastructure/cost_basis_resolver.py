from __future__ import annotations

from decimal import Decimal
from typing import Any


def to_decimal(value: Any) -> Decimal | None:
    """Convert raw numeric-like values to ``Decimal`` when possible."""
    try:
        if value is None or value == "":
            return None
        return Decimal(str(value))
    except Exception:
        return None


def calculate_cost_basis_from_executions(executions: list[dict[str, Any]], base_balance: float) -> float | None:
    """Calculate average remaining spot inventory cost from recent execution history."""
    if base_balance <= 0 or not executions:
        return None

    running_qty = Decimal("0")
    running_cost = Decimal("0")

    for execution in reversed(executions):
        side = str(execution.get("side") or "").upper()
        qty = to_decimal(execution.get("execQty") or execution.get("leavesQty") or execution.get("orderQty"))
        price = to_decimal(execution.get("execPrice") or execution.get("avgPrice") or execution.get("price"))
        if side not in {"BUY", "SELL"} or qty is None or qty <= 0 or price is None or price <= 0:
            continue

        fee = to_decimal(execution.get("execFee")) or Decimal("0")
        fee_currency = str(execution.get("feeCurrency") or execution.get("feeCcy") or "").upper()

        if side == "BUY":
            running_qty += qty
            running_cost += qty * price
            if fee_currency.endswith("USDT") or fee_currency.endswith("USDC"):
                running_cost += fee
        else:
            if running_qty <= 0:
                continue
            avg_cost = running_cost / running_qty if running_qty > 0 else Decimal("0")
            qty_to_remove = min(qty, running_qty)
            running_cost = max(running_cost - avg_cost * qty_to_remove, Decimal("0"))
            running_qty -= qty_to_remove
            if running_qty <= 0:
                running_qty = Decimal("0")
                running_cost = Decimal("0")

    actual_balance = Decimal(str(base_balance))
    if running_qty <= 0 or actual_balance <= 0:
        return None

    if running_qty > actual_balance:
        average_cost = running_cost / running_qty
        running_qty = actual_balance
        running_cost = average_cost * running_qty

    if running_cost <= 0:
        return None
    return float(running_cost / running_qty)
