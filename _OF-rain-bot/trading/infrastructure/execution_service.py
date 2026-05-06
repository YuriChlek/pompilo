from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
import logging
import time
from typing import Any

from trading.application.ports import (
    ExecutionOrder,
    ExecutionStreamEvent,
    FuturesTickerSnapshot,
    LivePositionSnapshot,
    OrderStatusSnapshot,
    OrderStreamEvent,
    OrderSubmissionResult,
    PositionStreamEvent,
    RuntimeRepositoryPort,
    StopMoveResult,
)
from trading.domain.models import ScalpSignal
from trading.infrastructure.bybit import (
    AsyncBybitTradingClient,
    BybitAccountError,
    BybitMarketDataError,
    BybitOrderError,
    BybitPositionError,
    BybitStreamError,
)
from trading.infrastructure.notifications import TelegramSignalNotifier

logger = logging.getLogger("trading.infrastructure.execution_service")

__all__ = [
    "BybitExecutionService",
    "ExecutionAccountError",
    "ExecutionMarketDataError",
    "ExecutionOrderError",
    "ExecutionPositionError",
    "ExecutionServiceError",
    "ExecutionStreamError",
]


class ExecutionServiceError(RuntimeError):
    pass


class ExecutionOrderError(ExecutionServiceError):
    pass


class ExecutionPositionError(ExecutionServiceError):
    pass


class ExecutionAccountError(ExecutionServiceError):
    pass


class ExecutionMarketDataError(ExecutionServiceError):
    pass


class ExecutionStreamError(ExecutionServiceError):
    pass


class BybitExecutionService:
    """Canonical ExecutionPort implementation backed by Bybit adapters."""

    def __init__(
        self,
        repository: RuntimeRepositoryPort,
        trading_client: AsyncBybitTradingClient | None = None,
        notifier: TelegramSignalNotifier | None = None,
    ) -> None:
        self.repository = repository
        self.trading_client = trading_client or AsyncBybitTradingClient()
        self.notifier = notifier or TelegramSignalNotifier()

    async def execute(self, signal: ScalpSignal, order_data: ExecutionOrder, dry_run: bool = True) -> OrderSubmissionResult:
        if dry_run:
            result = OrderSubmissionResult(status="dry_run", symbol=signal.symbol, payload=asdict(order_data))
            logger.info(
                "dry-run entry submitted symbol=%s side=%s price=%s size=%s",
                signal.symbol,
                order_data.direction,
                order_data.price,
                order_data.size,
            )
            await self.repository.insert_order_event(signal.symbol, "dry_run_entry", _serialize_dataclass(result))
            await self.notifier.notify_entry_submitted(
                signal.symbol,
                {
                    "direction": order_data.direction,
                    "price": float(order_data.price or 0),
                    "take_profit": float(order_data.take_profit or 0),
                    "stop_loss": float(order_data.stop_loss or 0),
                    "strategy_mode": signal.reason,
                    "run_mode": "DRY_RUN",
                },
            )
            return result

        try:
            order_id = await self.trading_client.place_entry_order(
                {
                    "symbol": order_data.symbol,
                    "direction": order_data.direction,
                    "order_type": order_data.order_type,
                    "size": float(order_data.size),
                    "price": None if order_data.price is None else float(order_data.price),
                    "stop_loss": None if order_data.stop_loss is None else float(order_data.stop_loss),
                    "take_profit": None if order_data.take_profit is None else float(order_data.take_profit),
                }
            )
        except BybitOrderError as exc:
            raise ExecutionOrderError(f"entry submission failed for {signal.symbol}") from exc

        result = OrderSubmissionResult(
            status="submitted" if order_id else "rejected",
            order_id=order_id,
            symbol=signal.symbol,
            side=order_data.direction,
            size=order_data.size,
            payload=asdict(order_data),
        )
        logger.info(
            "live entry %s symbol=%s order_id=%s side=%s price=%s size=%s",
            result.status,
            signal.symbol,
            order_id,
            order_data.direction,
            order_data.price,
            order_data.size,
        )
        await self.repository.insert_order_event(signal.symbol, "live_entry_submission", _serialize_dataclass(result))
        if result.status == "submitted":
            await self.notifier.notify_entry_submitted(
                signal.symbol,
                {
                    "direction": order_data.direction,
                    "price": float(order_data.price or 0),
                    "take_profit": float(order_data.take_profit or 0),
                    "stop_loss": float(order_data.stop_loss or 0),
                    "strategy_mode": signal.reason,
                    "run_mode": "LIVE",
                },
            )
        return result

    async def poll_entry(self, symbol: str, order_id: str, dry_run: bool = True) -> OrderStatusSnapshot:
        if dry_run:
            return OrderStatusSnapshot(status="submitted", order_id=order_id, symbol=symbol)
        try:
            status = await self.trading_client.get_order_status(symbol, order_id)
        except BybitOrderError as exc:
            raise ExecutionOrderError(f"entry status fetch failed for {symbol}") from exc
        if status is None:
            return OrderStatusSnapshot(status="unknown", order_id=order_id, symbol=symbol)

        normalized = {
            "New": "submitted",
            "PartiallyFilled": "partially_filled",
            "Filled": "filled",
            "Cancelled": "cancelled",
            "Rejected": "rejected",
        }.get(status["status"], str(status["status"]).lower())
        payload = OrderStatusSnapshot(
            status=normalized,
            order_id=str(status["order_id"]),
            symbol=str(status["symbol"]),
            side=status.get("side"),
            price=_to_optional_decimal(status.get("price")),
            qty=_to_optional_decimal(status.get("qty")),
            cum_exec_qty=_to_optional_decimal(status.get("cum_exec_qty")),
            payload=status,
        )
        logger.info("entry status symbol=%s order_id=%s status=%s", symbol, order_id, normalized)
        await self.repository.insert_order_event(symbol, "entry_status", _serialize_dataclass(payload))
        return payload

    async def cancel_entry(self, symbol: str, order_id: str, dry_run: bool = True, reason: str = "cancelled") -> OrderSubmissionResult:
        if dry_run:
            payload = OrderSubmissionResult(status="cancelled", order_id=order_id, symbol=symbol, reason=reason)
            logger.info("dry-run entry cancel symbol=%s order_id=%s reason=%s", symbol, order_id, reason)
            await self.repository.insert_order_event(symbol, "dry_run_cancel", _serialize_dataclass(payload))
            return payload
        try:
            cancelled = await self.trading_client.cancel_order(symbol, order_id)
        except BybitOrderError as exc:
            raise ExecutionOrderError(f"entry cancel failed for {symbol}") from exc
        payload = OrderSubmissionResult(
            status="cancelled" if cancelled else "cancel_failed",
            order_id=order_id,
            symbol=symbol,
            reason=reason,
        )
        logger.info("live entry cancel symbol=%s order_id=%s status=%s reason=%s", symbol, order_id, payload.status, reason)
        await self.repository.insert_order_event(symbol, "entry_cancel", _serialize_dataclass(payload))
        return payload

    async def detect_live_position(self, symbol: str) -> LivePositionSnapshot | None:
        try:
            positions = await self.trading_client.get_open_positions(symbol)
        except BybitPositionError as exc:
            raise ExecutionPositionError(f"position fetch failed for {symbol}") from exc
        active_positions = [item for item in positions if float(item.get("size") or 0) > 0]
        if not active_positions:
            return None
        item = active_positions[0]
        return LivePositionSnapshot(
            symbol=str(item["symbol"]),
            direction=str(item["direction"]),
            size=Decimal(str(item["size"])),
            avg_price=Decimal(str(item["avgPrice"])),
            take_profit=_to_optional_decimal(item.get("takeProfit")),
            stop_loss=_to_optional_decimal(item.get("stopLoss")),
        )

    async def fetch_futures_tick_size(self, symbol: str) -> float | None:
        try:
            instrument = await self.trading_client.get_linear_instrument_info(symbol)
        except BybitMarketDataError as exc:
            raise ExecutionMarketDataError(f"futures tick size fetch failed for {symbol}") from exc
        if instrument is None:
            return None
        price_filter = instrument.get("priceFilter") or {}
        tick_size = float(price_filter.get("tickSize") or 0.0)
        return tick_size if tick_size > 0 else None

    async def fetch_account_equity(self) -> float:
        try:
            return await self.trading_client.get_wallet_balance()
        except BybitAccountError as exc:
            raise ExecutionAccountError("account equity fetch failed") from exc

    async def fetch_futures_ticker(self, symbol: str) -> FuturesTickerSnapshot | None:
        try:
            ticker = await self.trading_client.get_linear_ticker(symbol)
        except BybitMarketDataError as exc:
            raise ExecutionMarketDataError(f"futures ticker fetch failed for {symbol}") from exc
        if ticker is None:
            return None
        return FuturesTickerSnapshot(
            bid=Decimal(str(ticker["bid"])),
            ask=Decimal(str(ticker["ask"])),
            mark=Decimal(str(ticker["mark"])),
            last=Decimal(str(ticker["last"])),
            mid=Decimal(str(ticker["mid"])),
        )

    async def exit_position(self, symbol: str, side: str, size: float, dry_run: bool = True, reason: str = "exit") -> OrderSubmissionResult:
        if dry_run:
            payload = OrderSubmissionResult(
                status="closed",
                symbol=symbol,
                side=side,
                size=Decimal(str(size)),
                reason=reason,
                payload={"closed_at_ms": int(time.time() * 1000)},
            )
            logger.info("dry-run exit symbol=%s side=%s size=%s reason=%s", symbol, side, size, reason)
            await self.repository.insert_order_event(symbol, "dry_run_exit", _serialize_dataclass(payload))
            return payload
        try:
            close_order_id = await self.trading_client.close_position_market(symbol, side, size)
        except BybitOrderError as exc:
            raise ExecutionOrderError(f"exit submission failed for {symbol}") from exc
        payload = OrderSubmissionResult(
            status="submitted" if close_order_id else "rejected",
            symbol=symbol,
            side=side,
            size=Decimal(str(size)),
            reason=reason,
            order_id=close_order_id,
        )
        logger.info("live exit %s symbol=%s side=%s size=%s reason=%s", payload.status, symbol, side, size, reason)
        await self.repository.insert_order_event(symbol, "live_exit_submission", _serialize_dataclass(payload))
        return payload

    async def move_stop_to_breakeven(
        self,
        symbol: str,
        side: str,
        stop_price: float,
        current_price: float,
        dry_run: bool = True,
        reason: str = "breakeven",
    ) -> StopMoveResult:
        if dry_run:
            payload = StopMoveResult(status="updated", symbol=symbol, stop_price=Decimal(str(stop_price)), reason=reason)
            logger.info("dry-run stop move symbol=%s stop_price=%s reason=%s", symbol, stop_price, reason)
            await self.repository.insert_order_event(symbol, "dry_run_stop_move", _serialize_dataclass(payload))
            await self.notifier.notify_stop_moved(
                symbol,
                {
                    "side": side,
                    "stop_price": stop_price,
                    "current_price": current_price,
                    "run_mode": "DRY_RUN",
                },
            )
            return payload
        try:
            await self.trading_client.move_stop_loss(symbol, stop_price)
        except BybitOrderError as exc:
            raise ExecutionOrderError(f"stop move failed for {symbol}") from exc
        payload = StopMoveResult(status="updated", symbol=symbol, stop_price=Decimal(str(stop_price)), reason=reason)
        logger.info("live stop moved symbol=%s stop_price=%s reason=%s", symbol, stop_price, reason)
        await self.repository.insert_order_event(symbol, "live_stop_move", _serialize_dataclass(payload))
        await self.notifier.notify_stop_moved(
            symbol,
            {
                "side": side,
                "stop_price": stop_price,
                "current_price": current_price,
                "run_mode": "LIVE",
            },
        )
        return payload

    def supports_private_execution_stream(self) -> bool:
        return bool(self.trading_client.transport.api_key and self.trading_client.transport.api_secret)

    async def run_private_stream(self, on_order_update, on_execution_update, on_position_update) -> None:
        await self.stream_private_execution_events(on_order_update, on_execution_update, on_position_update)

    def private_stream_enabled(self) -> bool:
        return self.supports_private_execution_stream()

    async def stream_private_execution_events(self, on_order_update, on_execution_update, on_position_update) -> None:
        async def _handle_order_update(payload: dict[str, Any]) -> None:
            await on_order_update(
                OrderStreamEvent(
                    order_id=str(payload.get("order_id") or ""),
                    order_link_id=payload.get("order_link_id"),
                    symbol=str(payload.get("symbol") or ""),
                    side=payload.get("side"),
                    status=payload.get("status"),
                    price=_to_optional_decimal(payload.get("price")),
                    qty=_to_optional_decimal(payload.get("qty")),
                    cum_exec_qty=_to_optional_decimal(payload.get("cum_exec_qty")),
                    avg_price=_to_optional_decimal(payload.get("avg_price")),
                    take_profit=_to_optional_decimal(payload.get("take_profit")),
                    stop_loss=_to_optional_decimal(payload.get("stop_loss")),
                    updated_time=_to_optional_int(payload.get("updated_time")),
                    raw=dict(payload.get("raw") or payload),
                )
            )

        async def _handle_execution_update(payload: dict[str, Any]) -> None:
            await on_execution_update(
                ExecutionStreamEvent(
                    order_id=str(payload.get("order_id") or ""),
                    order_link_id=payload.get("order_link_id"),
                    symbol=str(payload.get("symbol") or ""),
                    side=payload.get("side"),
                    exec_price=_to_optional_decimal(payload.get("exec_price")),
                    exec_qty=_to_optional_decimal(payload.get("exec_qty")),
                    leaves_qty=_to_optional_decimal(payload.get("leaves_qty")),
                    exec_time=_to_optional_int(payload.get("exec_time")),
                    raw=dict(payload.get("raw") or payload),
                )
            )

        async def _handle_position_update(payload: dict[str, Any]) -> None:
            await on_position_update(
                PositionStreamEvent(
                    symbol=str(payload.get("symbol") or ""),
                    side=payload.get("side"),
                    size=Decimal(str(payload.get("size") or 0)),
                    avg_price=_to_optional_decimal(payload.get("avg_price")),
                    take_profit=_to_optional_decimal(payload.get("take_profit")),
                    stop_loss=_to_optional_decimal(payload.get("stop_loss")),
                    updated_time=_to_optional_int(payload.get("updated_time")),
                    raw=dict(payload.get("raw") or payload),
                )
            )

        try:
            await self.trading_client.run_private_stream(
                on_order_update=_handle_order_update,
                on_execution_update=_handle_execution_update,
                on_position_update=_handle_position_update,
            )
        except (BybitStreamError, BybitOrderError) as exc:
            raise ExecutionStreamError("private execution stream failed") from exc

    async def close(self) -> None:
        await self.trading_client.close_private_stream()


def _to_optional_decimal(value: Any) -> Decimal | None:
    if value in {None, ""}:
        return None
    return Decimal(str(value))


def _to_optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def _serialize_dataclass(value: Any) -> dict[str, Any]:
    return _serialize_value(asdict(value))


def _serialize_value(value: Any):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    return value
