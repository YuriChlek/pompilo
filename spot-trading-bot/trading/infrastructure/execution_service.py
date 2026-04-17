from __future__ import annotations

import json
from decimal import Decimal

from trading.domain.models import ExecutionDecision, ExecutionResult, PositionState
from trading.infrastructure.binance_spot import BinanceSpotClient, base_asset_from_symbol, derive_avg_entry_price_from_trades, normalize_order_quantity, satisfies_min_notional
from utils.config import SPOT_BOT_SCHEMA
from utils.db_actions import create_connection


class BinanceSpotExecutor:
    def __init__(self, client: BinanceSpotClient | None = None) -> None:
        self.client = client or BinanceSpotClient()

    async def get_position_state(self, symbol: str) -> PositionState:
        conn = await create_connection()
        try:
            row = await conn.fetchrow(
                f"""
                SELECT symbol, quantity, avg_entry_price, total_cost
                FROM {SPOT_BOT_SCHEMA}.position_state
                WHERE symbol = $1
                """,
                symbol,
            )
            local_state = PositionState(symbol=symbol, quantity=Decimal("0"), avg_entry_price=Decimal("0"), total_cost=Decimal("0"))
            if row is not None:
                local_state = PositionState(
                    symbol=row["symbol"],
                    quantity=Decimal(str(row["quantity"])),
                    avg_entry_price=Decimal(str(row["avg_entry_price"])),
                    total_cost=Decimal(str(row["total_cost"])),
                )
            return await self._reconcile_position_state(conn, symbol, local_state)
        finally:
            await conn.close()

    async def _reconcile_position_state(
        self,
        conn,
        symbol: str,
        local_state: PositionState,
    ) -> PositionState:
        try:
            asset = base_asset_from_symbol(symbol)
            balance = self.client.fetch_asset_balance(asset)
            exchange_quantity = balance.total
            if exchange_quantity <= 0:
                reconciled_state = PositionState(symbol=symbol, quantity=Decimal("0"), avg_entry_price=Decimal("0"), total_cost=Decimal("0"))
            else:
                trades = self.client.fetch_my_trades(symbol)
                avg_entry_price = derive_avg_entry_price_from_trades(symbol, exchange_quantity, trades)
                if avg_entry_price <= 0 and local_state.quantity > 0:
                    avg_entry_price = local_state.avg_entry_price
                total_cost = exchange_quantity * avg_entry_price
                reconciled_state = PositionState(
                    symbol=symbol,
                    quantity=exchange_quantity,
                    avg_entry_price=avg_entry_price,
                    total_cost=total_cost,
                )

            if reconciled_state != local_state:
                await conn.execute(
                    f"""
                    INSERT INTO {SPOT_BOT_SCHEMA}.position_state (symbol, quantity, avg_entry_price, total_cost, updated_at)
                    VALUES ($1, $2, $3, $4, NOW())
                    ON CONFLICT (symbol) DO UPDATE
                    SET quantity = EXCLUDED.quantity,
                        avg_entry_price = EXCLUDED.avg_entry_price,
                        total_cost = EXCLUDED.total_cost,
                        updated_at = NOW()
                    """,
                    reconciled_state.symbol,
                    reconciled_state.quantity,
                    reconciled_state.avg_entry_price,
                    reconciled_state.total_cost,
                )
                print(
                    "🔁 Reconciled position state "
                    f"symbol={symbol} local_qty={local_state.quantity} exchange_qty={reconciled_state.quantity} "
                    f"avg_entry={reconciled_state.avg_entry_price}"
                )
            return reconciled_state
        except Exception as exc:
            print(f"⚠️ Reconciliation skipped for {symbol}: {exc}")
            return local_state

    async def execute(
        self,
        decision: ExecutionDecision,
        position_state: PositionState,
        *,
        dry_run: bool = False,
    ) -> ExecutionResult:
        if decision.action == "skip":
            await self._record_ledger(decision, position_state, None, None, "skipped")
            return ExecutionResult(False, decision.symbol, decision.action, decision.reason, decision.signal_price, dry_run=dry_run)

        if dry_run:
            await self._record_ledger(decision, position_state, decision.signal_price, None, "dry_run")
            return ExecutionResult(
                False,
                decision.symbol,
                decision.action,
                f"dry_run:{decision.reason}",
                decision.signal_price,
                executed_price=decision.signal_price,
                quantity=decision.quantity,
                exchange_order_id=None,
                dry_run=True,
            )

        return await self._execute_trade(decision, position_state, dry_run=False)

    async def _execute_trade(
        self,
        decision: ExecutionDecision,
        position_state: PositionState,
        *,
        dry_run: bool,
    ) -> ExecutionResult:
        filters = self.client.get_symbol_filters(decision.symbol)
        normalized_quantity = normalize_order_quantity(decision.quantity, filters)
        if normalized_quantity <= 0:
            await self._record_ledger(decision, position_state, None, None, "skipped")
            return ExecutionResult(
                False,
                decision.symbol,
                "skip",
                "quantity_below_binance_min_qty",
                decision.signal_price,
                dry_run=dry_run,
            )
        if not satisfies_min_notional(normalized_quantity, decision.signal_price, filters):
            await self._record_ledger(decision, position_state, None, None, "skipped")
            return ExecutionResult(
                False,
                decision.symbol,
                "skip",
                "quantity_below_binance_min_notional",
                decision.signal_price,
                quantity=normalized_quantity,
                dry_run=dry_run,
            )

        side = "BUY" if decision.action == "buy" else "SELL"
        order_payload = self.client.place_market_order(decision.symbol, side, normalized_quantity)
        executed_price = self.client.extract_fill_price(order_payload, decision.signal_price)
        exchange_order_id = str(order_payload.get("orderId"))
        normalized_decision = ExecutionDecision(
            decision.action,
            decision.symbol,
            decision.signal_price,
            normalized_quantity,
            normalized_quantity * decision.signal_price,
            decision.reason,
        )
        await self._apply_position_update(normalized_decision, position_state, executed_price, exchange_order_id, order_payload)
        return ExecutionResult(
            True,
            decision.symbol,
            decision.action,
            decision.reason,
            decision.signal_price,
            executed_price=executed_price,
            quantity=normalized_quantity,
            exchange_order_id=exchange_order_id,
            dry_run=dry_run,
        )

    async def _apply_position_update(
        self,
        decision: ExecutionDecision,
        position_state: PositionState,
        executed_price: Decimal,
        exchange_order_id: str,
        order_payload: dict,
    ) -> None:
        conn = await create_connection()
        try:
            if decision.action == "buy":
                new_quantity = position_state.quantity + decision.quantity
                new_total_cost = position_state.total_cost + (decision.quantity * executed_price)
                new_avg_entry = new_total_cost / new_quantity
            else:
                new_quantity = Decimal("0")
                new_total_cost = Decimal("0")
                new_avg_entry = Decimal("0")

            await conn.execute(
                f"""
                INSERT INTO {SPOT_BOT_SCHEMA}.position_state (symbol, quantity, avg_entry_price, total_cost, updated_at)
                VALUES ($1, $2, $3, $4, NOW())
                ON CONFLICT (symbol) DO UPDATE
                SET quantity = EXCLUDED.quantity,
                    avg_entry_price = EXCLUDED.avg_entry_price,
                    total_cost = EXCLUDED.total_cost,
                    updated_at = NOW()
                """,
                decision.symbol,
                new_quantity,
                new_avg_entry,
                new_total_cost,
            )
            await self._record_ledger(decision, position_state, executed_price, exchange_order_id, "executed", conn=conn, payload=order_payload, avg_entry_after=new_avg_entry)
        finally:
            await conn.close()

    async def _record_ledger(
        self,
        decision: ExecutionDecision,
        position_state: PositionState,
        executed_price: Decimal | None,
        exchange_order_id: str | None,
        status: str,
        *,
        conn=None,
        payload: dict | None = None,
        avg_entry_after: Decimal | None = None,
    ) -> None:
        owns_connection = conn is None
        conn = conn or await create_connection()
        try:
            await conn.execute(
                f"""
                INSERT INTO {SPOT_BOT_SCHEMA}.order_ledger (
                    symbol, side, status, signal_price, executed_price, quantity, quote_amount,
                    avg_entry_price_before, avg_entry_price_after, exchange_order_id, exchange_payload
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::jsonb)
                """,
                decision.symbol,
                decision.action,
                status,
                decision.signal_price,
                executed_price,
                decision.quantity,
                decision.quote_amount,
                position_state.avg_entry_price,
                avg_entry_after if avg_entry_after is not None else position_state.avg_entry_price,
                exchange_order_id,
                json.dumps(payload or {}),
            )
        finally:
            if owns_connection:
                await conn.close()
