from __future__ import annotations

import logging
import time

from orderflow.market_data.models import ScalpSignal
from orderflow.execution.bybit_client import AsyncBybitTradingClient
from orderflow.storage.repository import OrderFlowRepository
from telegram_bot import send_breakeven_message, send_limit_order_message

logger = logging.getLogger("orderflow.executor")


class BybitExecutionService:
    def __init__(self, repository: OrderFlowRepository, trading_client: AsyncBybitTradingClient | None = None) -> None:
        self.repository = repository
        self.trading_client = trading_client or AsyncBybitTradingClient()

    async def execute(self, signal: ScalpSignal, order_data: dict, dry_run: bool = True) -> dict:
        if dry_run:
            result = {"status": "dry_run", "symbol": signal.symbol, "payload": order_data}
            logger.info("dry-run entry submitted symbol=%s side=%s price=%s size=%s", signal.symbol, order_data["direction"], order_data["price"], order_data["size"])
            await self.repository.insert_order_event(signal.symbol, "dry_run_entry", result)
            await send_limit_order_message(
                signal.symbol,
                order_data["direction"],
                float(order_data["price"]),
                float(order_data["take_profit"]),
                float(order_data["stop_loss"]),
                signal.reason,
                "DRY_RUN",
            )
            return result

        order_id = None
        order_id = await self.trading_client.place_entry_order(order_data)
        result = {
            "status": "submitted" if order_id else "rejected",
            "order_id": order_id,
            "symbol": signal.symbol,
            "payload": order_data,
        }
        logger.info("live entry %s symbol=%s order_id=%s side=%s price=%s size=%s", result["status"], signal.symbol, order_id, order_data["direction"], order_data["price"], order_data["size"])
        await self.repository.insert_order_event(signal.symbol, "live_entry_submission", result)
        if result["status"] == "submitted":
            await send_limit_order_message(
                signal.symbol,
                order_data["direction"],
                float(order_data["price"]),
                float(order_data["take_profit"]),
                float(order_data["stop_loss"]),
                signal.reason,
                "LIVE",
            )
        return result

    async def poll_entry(self, symbol: str, order_id: str, dry_run: bool = True) -> dict:
        if dry_run:
            return {"status": "submitted", "order_id": order_id, "symbol": symbol}

        status = await self.trading_client.get_order_status(symbol, order_id)
        if status is None:
            return {"status": "unknown", "order_id": order_id, "symbol": symbol}

        normalized = {
            "New": "submitted",
            "PartiallyFilled": "partially_filled",
            "Filled": "filled",
            "Cancelled": "cancelled",
            "Rejected": "rejected",
        }.get(status["status"], status["status"].lower())
        payload = {"status": normalized, **status}
        logger.info("entry status symbol=%s order_id=%s status=%s", symbol, order_id, normalized)
        await self.repository.insert_order_event(symbol, "entry_status", payload)
        return payload

    async def cancel_entry(self, symbol: str, order_id: str, dry_run: bool = True, reason: str = "cancelled") -> dict:
        if dry_run:
            payload = {"status": "cancelled", "order_id": order_id, "symbol": symbol, "reason": reason}
            logger.info("dry-run entry cancel symbol=%s order_id=%s reason=%s", symbol, order_id, reason)
            await self.repository.insert_order_event(symbol, "dry_run_cancel", payload)
            return payload

        cancelled = await self.trading_client.cancel_order(symbol, order_id)
        payload = {
            "status": "cancelled" if cancelled else "cancel_failed",
            "order_id": order_id,
            "symbol": symbol,
            "reason": reason,
        }
        logger.info("live entry cancel symbol=%s order_id=%s status=%s reason=%s", symbol, order_id, payload["status"], reason)
        await self.repository.insert_order_event(symbol, "entry_cancel", payload)
        return payload

    async def detect_live_position(self, symbol: str) -> dict | None:
        positions = await self.trading_client.get_open_positions(symbol)
        active_positions = [item for item in positions if float(item.get("size") or 0) > 0]
        if not active_positions:
            return None
        return active_positions[0]

    async def exit_position(self, symbol: str, side: str, size: float, dry_run: bool = True, reason: str = "exit") -> dict:
        if dry_run:
            payload = {
                "status": "closed",
                "symbol": symbol,
                "side": side,
                "size": size,
                "reason": reason,
                "closed_at_ms": int(time.time() * 1000),
            }
            logger.info("dry-run exit symbol=%s side=%s size=%s reason=%s", symbol, side, size, reason)
            await self.repository.insert_order_event(symbol, "dry_run_exit", payload)
            return payload

        close_order_id = await self.trading_client.close_position_market(symbol, side, size)
        payload = {
            "status": "submitted" if close_order_id else "rejected",
            "symbol": symbol,
            "side": side,
            "size": size,
            "reason": reason,
            "order_id": close_order_id,
        }
        logger.info("live exit %s symbol=%s side=%s size=%s reason=%s", payload["status"], symbol, side, size, reason)
        await self.repository.insert_order_event(symbol, "live_exit_submission", payload)
        return payload

    async def move_stop_to_breakeven(
        self,
        symbol: str,
        side: str,
        stop_price: float,
        current_price: float,
        dry_run: bool = True,
        reason: str = "breakeven",
    ) -> dict:
        if dry_run:
            payload = {
                "status": "updated",
                "symbol": symbol,
                "stop_price": stop_price,
                "reason": reason,
            }
            logger.info("dry-run stop move symbol=%s stop_price=%s reason=%s", symbol, stop_price, reason)
            await self.repository.insert_order_event(symbol, "dry_run_stop_move", payload)
            await send_breakeven_message(symbol, side, stop_price, current_price, "DRY_RUN")
            return payload

        await self.trading_client.move_stop_loss(symbol, stop_price)
        payload = {
            "status": "updated",
            "symbol": symbol,
            "stop_price": stop_price,
            "reason": reason,
        }
        logger.info("live stop moved symbol=%s stop_price=%s reason=%s", symbol, stop_price, reason)
        await self.repository.insert_order_event(symbol, "live_stop_move", payload)
        await send_breakeven_message(symbol, side, stop_price, current_price, "LIVE")
        return payload
