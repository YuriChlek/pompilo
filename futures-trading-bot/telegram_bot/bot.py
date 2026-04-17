from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from functools import lru_cache
import zoneinfo

from telegram import Bot
from utils.env import load_project_env

LOGGER = logging.getLogger(__name__)

load_project_env()

DEFAULT_TIMEZONE = os.getenv("APP_TIMEZONE", "Europe/Kyiv")


def get_position_icon(direction: str) -> str:
    """Return a text label with emoji for the requested trade direction."""
    return "🟢 📈 LONG" if str(direction).lower() == "buy" else "🔴 📉 SHORT"


def _current_time() -> datetime:
    """Return current localized time for Telegram notifications."""
    timezone_name = os.getenv("APP_TIMEZONE", DEFAULT_TIMEZONE)
    return datetime.now(zoneinfo.ZoneInfo(timezone_name)).replace(microsecond=0)


def _get_required_env(name: str) -> str:
    """Read a required environment variable and fail fast when it is missing."""
    value = str(os.getenv(name, "")).strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@lru_cache(maxsize=1)
def _get_bot() -> Bot:
    """Create and cache a Telegram bot client configured from environment variables."""
    return Bot(token=_get_required_env("TOKEN"))


async def _send_html_message(chat_id: str, message: str) -> None:
    """Send an HTML-formatted Telegram message without breaking the trading flow on errors."""
    if not chat_id:
        LOGGER.warning("telegram_message_skipped reason=missing_chat_id")
        return

    try:
        await _get_bot().send_message(chat_id=chat_id, text=message, parse_mode="HTML")
    except Exception as exc:
        LOGGER.error("telegram_message_error chat_id=%s error=%s", chat_id, exc)


def _format_strategy_name(strategy_mode: str) -> str:
    """Return a human-readable strategy name for Telegram notifications."""
    return str(strategy_mode or "unknown").replace("_", " ").title()


async def send_message(symbol, direction, price, take_profit, stop_loss, strategy_mode):
    """Send a new-position notification to the primary Telegram chat."""
    chat_id = os.getenv("CHAT_ID", "").strip()
    position = get_position_icon(str(direction).lower())
    current_time = _current_time()
    strategy_name = _format_strategy_name(strategy_mode)

    message = f"""🚀 NEW TRADE

<b>{position}</b>
<b><i>{symbol}</i></b>

<b>Time:</b> {current_time}
<b>Strategy:</b> {strategy_name}
<b>Price:</b> {price}
<b>Take profit:</b> {take_profit}
<b>Stop loss:</b> {stop_loss}
"""
    await _send_html_message(chat_id, message)


async def send_pompilo_order_message(symbol, price, take_profit, stop_loss, direction, signal="Test message"):
    """Send an expanded order message with signal details to the primary Telegram chat."""
    chat_id = os.getenv("CHAT_ID", "").strip()
    position = get_position_icon(str(direction).lower())

    message = f"""
<b>{position}: </b>

<b><i>{symbol}</i></b>
<b>Entry price:</b> {price}
<b>Take Profit:</b> {take_profit}
<b>Stop Loss:</b> {stop_loss}

<i>{signal}</i>
"""
    await _send_html_message(chat_id, message)


async def send_breakeven_message(symbol, direction, entry_price, current_price, partial_close_qty=None):
    """Send a notification that a position was moved to breakeven."""
    chat_id = os.getenv("CHAT_ID", "").strip()
    position = get_position_icon(str(direction).lower())
    current_time = _current_time()

    partial_close_line = ""
    if partial_close_qty is not None:
        partial_close_line = f"<b>Partial close:</b> {partial_close_qty}\n"

    message = f"""🛡️ POSITION MOVED TO BREAKEVEN

<b>{position}</b>
<b><i>{symbol}</i></b>

<b>Time:</b> {current_time}
<b>Breakeven price:</b> {entry_price}
<b>Current price:</b> {current_price}
{partial_close_line}"""
    await _send_html_message(chat_id, message)


async def test_run():
    """Run a local smoke test for Telegram message delivery."""
    await send_pompilo_order_message("SOLUSDT", 1999, 1999, 1999, "sell")


__all__ = [
    "get_position_icon",
    "send_message",
    "send_breakeven_message",
    "send_pompilo_order_message",
    "test_run",
]


if __name__ == "__main__":
    asyncio.run(test_run())
