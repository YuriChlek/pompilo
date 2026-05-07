from __future__ import annotations

import json
import logging
from decimal import Decimal

from domain.models import ExecutionDecision, ExecutionResult, PositionState
from infrastructure.bybit_spot import BybitSpotClient, derive_avg_entry_price_from_trades, normalize_order_quantity, satisfies_min_notional, split_symbol
from utils.config import MIN_PROFIT_RATIO, NO_LOSS_GUARD_ENABLED, SPOT_BOT_SCHEMA

logger = logging.getLogger(__name__)

DECIMAL_ZERO = Decimal("0")


def _create_connection():
    from utils.db_actions import create_connection

    return create_connection()


class BybitSpotExecutor:
    """Execute spot market decisions on Bybit and persist position/ledger updates."""

    def __init__(self, client: BybitSpotClient | None = None) -> None:
        self.client = client or BybitSpotClient()

    def _minimum_sell_price(self, position_state: PositionState) -> Decimal:
        return position_state.avg_entry_price * (Decimal("1") + MIN_PROFIT_RATIO)

    async def get_position_state(self, symbol: str) -> PositionState:
        """Return the local position state reconciled against Bybit balances and trades."""

        conn = await _create_connection()
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

    async def get_quote_balance(self, symbol: str) -> Decimal:
        """Return the free quote-asset balance available for a buy order."""

        _, quote_asset = split_symbol(symbol)
        balance = self.client.fetch_asset_balance(quote_asset)
        return balance.free

    async def _reconcile_position_state(
        self,
        conn,
        symbol: str,
        local_state: PositionState,
    ) -> PositionState:
        try:
            asset, _ = split_symbol(symbol)
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
                logger.info(
                    "position_reconciled symbol=%s local_qty=%s exchange_qty=%s avg_entry=%s",
                    symbol,
                    local_state.quantity,
                    reconciled_state.quantity,
                    reconciled_state.avg_entry_price,
                )
            return reconciled_state
        except Exception as exc:
            logger.exception("position_reconciliation_skipped symbol=%s error=%s", symbol, exc)
            return local_state

    async def execute(
        self,
        decision: ExecutionDecision,
        position_state: PositionState,
        *,
        dry_run: bool = False,
    ) -> ExecutionResult:
        """Execute one decision or record a skipped/dry-run outcome."""

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
                "quantity_below_bybit_min_qty",
                decision.signal_price,
                dry_run=dry_run,
            )
        if not satisfies_min_notional(normalized_quantity, decision.signal_price, filters):
            await self._record_ledger(decision, position_state, None, None, "skipped")
            return ExecutionResult(
                False,
                decision.symbol,
                "skip",
                "quantity_below_bybit_min_notional",
                decision.signal_price,
                quantity=normalized_quantity,
                dry_run=dry_run,
            )

        if await self._has_executed_signal(decision):
            await self._record_ledger(decision, position_state, None, None, "skipped")
            return ExecutionResult(
                False,
                decision.symbol,
                "skip",
                "duplicate_signal_order",
                decision.signal_price,
                quantity=normalized_quantity,
                dry_run=dry_run,
            )

        no_loss_check_price: Decimal | None = None
        if decision.action == "sell" and position_state.has_position and NO_LOSS_GUARD_ENABLED:
            current_price = self.client.fetch_current_price(decision.symbol)
            no_loss_check_price = current_price
            min_sell_price = self._minimum_sell_price(position_state)
            if current_price < min_sell_price:
                logger.warning(
                    "no_loss_guard_triggered symbol=%s current_price=%s min_sell=%s avg_entry=%s",
                    decision.symbol,
                    current_price,
                    min_sell_price,
                    position_state.avg_entry_price,
                )
                await self._record_ledger(
                    decision,
                    position_state,
                    None,
                    None,
                    "blocked_no_loss",
                    no_loss_check_price=current_price,
                )
                return ExecutionResult(
                    False,
                    decision.symbol,
                    "skip",
                    "no_loss_guard_infrastructure",
                    decision.signal_price,
                    quantity=normalized_quantity,
                    dry_run=dry_run,
                )

        side = "BUY" if decision.action == "buy" else "SELL"
        order_payload = self.client.place_market_order(decision.symbol, side, normalized_quantity)
        executed_price = self.client.extract_fill_price(order_payload, decision.signal_price)
        exchange_order_id = str((order_payload.get("result") or {}).get("orderId", ""))
        if exchange_order_id:
            confirmed_order = self.client.get_order_status(decision.symbol, exchange_order_id)
            if confirmed_order.get("status") != "Filled":
                raise RuntimeError(f"Order {exchange_order_id} not filled: {confirmed_order}")
        normalized_decision = ExecutionDecision(
            decision.action,
            decision.symbol,
            decision.signal_price,
            normalized_quantity,
            normalized_quantity * decision.signal_price,
            decision.reason,
            decision.signal_timeframe,
            decision.signal_candle_id,
        )
        await self._apply_position_update(
            normalized_decision,
            position_state,
            executed_price,
            exchange_order_id,
            order_payload,
            no_loss_check_price=no_loss_check_price,
        )
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

    async def _has_executed_signal(self, decision: ExecutionDecision) -> bool:
        if not decision.signal_timeframe or not decision.signal_candle_id or decision.action == "skip":
            return False

        conn = await _create_connection()
        try:
            count = await conn.fetchval(
                f"""
                SELECT COUNT(*)
                FROM {SPOT_BOT_SCHEMA}.order_ledger
                WHERE symbol = $1
                  AND side = $2
                  AND status = 'executed'
                  AND signal_timeframe = $3
                  AND signal_candle_id = $4
                """,
                decision.symbol,
                decision.action,
                decision.signal_timeframe,
                decision.signal_candle_id,
            )
            return int(count or 0) > 0
        finally:
            await conn.close()

    async def _apply_position_update(
        self,
        decision: ExecutionDecision,
        position_state: PositionState,
        executed_price: Decimal,
        exchange_order_id: str,
        order_payload: dict,
        *,
        no_loss_check_price: Decimal | None = None,
    ) -> None:
        conn = await _create_connection()
        try:
            async with conn.transaction():
                ledger_status = "executed"
                realized_pnl_usdt: Decimal | None = None
                realized_pnl_pct: Decimal | None = None
                if decision.action == "buy":
                    new_quantity = position_state.quantity + decision.quantity
                    new_total_cost = position_state.total_cost + (decision.quantity * executed_price)
                    new_avg_entry = new_total_cost / new_quantity
                else:
                    realized_pnl_usdt = (executed_price - position_state.avg_entry_price) * decision.quantity
                    if position_state.total_cost > 0:
                        realized_pnl_pct = (realized_pnl_usdt / position_state.total_cost) * Decimal("100")
                    min_sell_price = self._minimum_sell_price(position_state)
                    if executed_price < min_sell_price:
                        ledger_status = "no_loss_violation"
                        logger.error(
                            "no_loss_violation symbol=%s executed_price=%s min_sell=%s avg_entry=%s pnl=%s",
                            decision.symbol,
                            executed_price,
                            min_sell_price,
                            position_state.avg_entry_price,
                            realized_pnl_usdt,
                        )
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
                await self._record_ledger(
                    decision,
                    position_state,
                    executed_price,
                    exchange_order_id,
                    ledger_status,
                    conn=conn,
                    payload=order_payload,
                    avg_entry_after=new_avg_entry,
                    realized_pnl_usdt=realized_pnl_usdt,
                    realized_pnl_pct=realized_pnl_pct,
                    no_loss_check_price=no_loss_check_price,
                )
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
        realized_pnl_usdt: Decimal | None = None,
        realized_pnl_pct: Decimal | None = None,
        no_loss_check_price: Decimal | None = None,
    ) -> None:
        owns_connection = conn is None
        conn = conn or await _create_connection()
        try:
            await conn.execute(
                f"""
                INSERT INTO {SPOT_BOT_SCHEMA}.order_ledger (
                    symbol, side, status, signal_price, executed_price, quantity, quote_amount,
                    avg_entry_price_before, avg_entry_price_after, exchange_order_id, exchange_payload,
                    realized_pnl_usdt, realized_pnl_pct, no_loss_check_price, signal_timeframe, signal_candle_id
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::jsonb,$12,$13,$14,$15,$16)
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
                realized_pnl_usdt,
                realized_pnl_pct,
                no_loss_check_price,
                decision.signal_timeframe,
                decision.signal_candle_id,
            )
        finally:
            if owns_connection:
                await conn.close()
