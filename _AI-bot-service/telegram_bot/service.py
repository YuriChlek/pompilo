from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from telegram import Bot

from telegram_bot.config import TelegramBotSettings, get_telegram_bot_settings


@dataclass(frozen=True)
class TelegramMessageContext:
    symbol: str
    order_side: str | None
    order_grid: Any
    price: float | None
    time: str


def get_position_icon(direction: str) -> str:
    return "🟢 📈 LONG" if direction.lower() == "buy" else "🔴 📉 SHORT"


def _grid_lines(order_grid: Any) -> Iterable[str]:
    if not order_grid:
        return []
    lines: list[str] = []
    for index, level in enumerate(reversed(order_grid), start=1):
        order_type = getattr(level, "order_type", None)
        price = getattr(level, "price", None)
        if order_type is None and isinstance(level, dict):
            order_type = level.get("order_type")
            price = level.get("price")
        if order_type is None or price is None:
            continue
        lines.append(f"<b>Level {index}</b>: {str(order_type).upper()} at {float(price):.4f}")
    return lines


def build_order_message(signal_data: dict[str, Any]) -> str:
    context = TelegramMessageContext(
        symbol=str(signal_data["symbol"]),
        order_side=str(signal_data.get("order_side") or ""),
        order_grid=signal_data.get("order_grid"),
        price=signal_data.get("price"),
        time=str(signal_data["time"]),
    )
    position = get_position_icon(context.order_side.lower())
    lines = [
        f"<b>{position}: </b>",
        "",
        f"<b><i>{context.symbol}</i></b>",
    ]
    grid_lines = list(_grid_lines(context.order_grid))
    if grid_lines:
        lines.extend(grid_lines)
    elif context.order_side.lower() == "sell":
        lines.append(f"<b>Price: <i>{context.price}</i></b>")
    lines.append(f"Time: {context.time}")
    return "\n".join(lines)


def build_telegram_bot(settings: TelegramBotSettings | None = None) -> Bot:
    resolved_settings = settings or get_telegram_bot_settings()
    return Bot(token=resolved_settings.token)


async def send_order_message(
    signal_data: dict[str, Any],
    *,
    settings: TelegramBotSettings | None = None,
) -> None:
    resolved_settings = settings or get_telegram_bot_settings()
    bot = build_telegram_bot(resolved_settings)
    message = build_order_message(signal_data)
    await bot.send_message(chat_id=resolved_settings.chat_id, text=message, parse_mode="HTML")


__all__ = [
    "TelegramMessageContext",
    "build_order_message",
    "build_telegram_bot",
    "get_position_icon",
    "send_order_message",
]
