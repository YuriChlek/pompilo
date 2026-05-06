from __future__ import annotations

import logging
from datetime import datetime
from functools import lru_cache
import zoneinfo
from typing import Any

from telegram import Bot

from utils.config import APP_TIMEZONE, TELEGRAM_CHAT_ID, TELEGRAM_TOKEN

LOGGER = logging.getLogger("trading.infrastructure.notifications")

__all__ = ["TelegramSignalNotifier", "get_position_icon", "send_breakeven_message", "send_limit_order_message"]


def get_position_icon(direction: str) -> str:
    return "🟢 📈 LONG" if str(direction).lower() == "buy" else "🔴 📉 SHORT"


def _current_time() -> datetime:
    return datetime.now(zoneinfo.ZoneInfo(APP_TIMEZONE)).replace(microsecond=0)


@lru_cache(maxsize=1)
def _get_bot() -> Bot:
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Missing required environment variable: TOKEN")
    return Bot(token=TELEGRAM_TOKEN)


async def _send_html_message(chat_id: str, message: str) -> None:
    if not chat_id:
        LOGGER.warning("telegram_message_skipped reason=missing_chat_id")
        return
    try:
        await _get_bot().send_message(chat_id=chat_id, text=message, parse_mode="HTML")
    except Exception as exc:
        LOGGER.error("telegram_message_error chat_id=%s error=%s", chat_id, exc)


async def _send_position_message(
    *,
    title: str,
    symbol: str,
    direction: str,
    run_mode: str,
    fields: list[tuple[str, object]],
) -> None:
    message = (
        f"{title}\n\n"
        f"<b>{get_position_icon(direction)}</b>\n"
        f"<b><i>{symbol}</i></b>\n\n"
        f"<b>Time:</b> {_current_time()}\n"
        f"<b>Run mode:</b> {run_mode}\n"
        f"{chr(10).join(f'<b>{label}:</b> {value}' for label, value in fields)}\n"
    )
    await _send_html_message(TELEGRAM_CHAT_ID, message)


class TelegramSignalNotifier:
    """Canonical Telegram notifier used by infrastructure adapters."""

    async def notify_entry_submitted(self, symbol: str, payload: dict[str, Any]) -> None:
        direction = str(payload.get("direction") or payload.get("side") or "")
        await send_limit_order_message(
            symbol=symbol,
            direction=direction,
            price=float(payload.get("price") or 0.0),
            take_profit=float(payload.get("take_profit") or 0.0),
            stop_loss=float(payload.get("stop_loss") or 0.0),
            strategy_mode=str(payload.get("strategy_mode") or payload.get("reason") or ""),
            run_mode=str(payload.get("run_mode") or "LIVE"),
        )

    async def notify_stop_moved(self, symbol: str, payload: dict[str, Any]) -> None:
        direction = str(payload.get("direction") or payload.get("side") or "")
        await send_breakeven_message(
            symbol=symbol,
            direction=direction,
            entry_price=float(payload.get("stop_price") or 0.0),
            current_price=float(payload.get("current_price") or 0.0),
            run_mode=str(payload.get("run_mode") or "LIVE"),
        )


async def send_limit_order_message(
    symbol: str,
    direction: str,
    price: float,
    take_profit: float,
    stop_loss: float,
    strategy_mode: str,
    run_mode: str,
) -> None:
    await _send_position_message(
        title="📌 LIMIT ORDER PLACED",
        symbol=symbol,
        direction=direction,
        run_mode=run_mode,
        fields=[
            ("Strategy", strategy_mode),
            ("Entry price", price),
            ("Take profit", take_profit),
            ("Stop loss", stop_loss),
        ],
    )


async def send_breakeven_message(symbol: str, direction: str, entry_price: float, current_price: float, run_mode: str) -> None:
    await _send_position_message(
        title="🛡️ POSITION MOVED TO BREAKEVEN",
        symbol=symbol,
        direction=direction,
        run_mode=run_mode,
        fields=[
            ("Breakeven price", entry_price),
            ("Current price", current_price),
        ],
    )
