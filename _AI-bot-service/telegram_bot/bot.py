from __future__ import annotations

import asyncio
from typing import Any

from telegram_bot.service import get_position_icon, send_order_message


async def send_pompilo_order_message(signal_data: dict[str, Any]) -> None:
    try:
        await send_order_message(signal_data)
    except Exception as exc:
        print(f"Telegram message error {exc}")


async def test_run() -> None:
    test_data = {
        "symbol": "SOLUSDT",
        "order_side": "sell",
        "price": 199.45,
        "time": "test-run",
        "order_grid": [
            {"order_type": "limit", "price": 199.45},
            {"order_type": "limit", "price": 198.95},
        ],
    }
    await send_pompilo_order_message(test_data)


__all__ = [
    "get_position_icon",
    "send_pompilo_order_message",
    "test_run",
]


if __name__ == "__main__":
    asyncio.run(test_run())
