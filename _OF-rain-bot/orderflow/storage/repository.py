from __future__ import annotations

import json
import logging

from utils.config import ORDERFLOW_DB_SCHEMA
from utils.db import get_db_pool

logger = logging.getLogger("orderflow.repository")


class OrderFlowRepository:
    async def _execute(self, query: str, *args) -> None:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(query, *args)

    async def insert_order_event(self, symbol: str, event_type: str, payload: dict) -> None:
        try:
            await self._execute(
                f"""
                INSERT INTO {ORDERFLOW_DB_SCHEMA}.order_events (symbol, event_type, payload)
                VALUES ($1, $2, $3::jsonb);
                """,
                symbol,
                event_type,
                json.dumps(payload),
            )
        except Exception as exc:
            logger.warning("order event skipped: %s", exc)


    async def insert_position_event(
        self,
        symbol: str,
        event_type: str,
        side: str | None = None,
        entry_price: float | None = None,
        mark_price: float | None = None,
        stop_price: float | None = None,
        take_profit_price: float | None = None,
        size: float | None = None,
        hold_time_ms: int | None = None,
        reason: str | None = None,
        payload: dict | None = None,
    ) -> None:
        try:
            await self._execute(
                f"""
                INSERT INTO {ORDERFLOW_DB_SCHEMA}.position_events (
                    symbol, event_type, side, entry_price, mark_price,
                    stop_price, take_profit_price, size, hold_time_ms,
                    reason, payload
                ) VALUES (
                    $1, $2, $3, $4, $5,
                    $6, $7, $8, $9,
                    $10, $11::jsonb
                );
                """,
                symbol,
                event_type,
                side,
                entry_price,
                mark_price,
                stop_price,
                take_profit_price,
                size,
                hold_time_ms,
                reason,
                json.dumps(payload or {}),
            )
        except Exception as exc:
            logger.warning("position event skipped: %s", exc)

    async def insert_runtime_transition(
        self,
        symbol: str,
        from_state: str | None,
        to_state: str,
        reason: str | None = None,
        payload: dict | None = None,
    ) -> None:
        try:
            await self._execute(
                f"""
                INSERT INTO {ORDERFLOW_DB_SCHEMA}.runtime_transitions (
                    symbol, from_state, to_state, reason, payload
                ) VALUES ($1, $2, $3, $4, $5::jsonb);
                """,
                symbol,
                from_state,
                to_state,
                reason,
                json.dumps(payload or {}),
            )
        except Exception as exc:
            logger.warning("runtime transition skipped: %s", exc)
