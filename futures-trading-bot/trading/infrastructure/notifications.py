from __future__ import annotations

from telegram_bot import send_breakeven_message, send_message


class TelegramSignalNotifier:
    """Infrastructure adapter for Telegram trade notifications."""

    async def notify_new_position(
        self,
        symbol: str,
        direction: str,
        price,
        take_profit,
        stop_loss,
        strategy_mode: str,
    ) -> None:
        """Send a Telegram notification about a successfully executed trade signal."""
        await send_message(symbol, direction, price, take_profit, stop_loss, strategy_mode)

    async def notify_position_moved_to_breakeven(
        self,
        symbol: str,
        direction: str,
        entry_price,
        current_price,
        partial_close_qty=None,
    ) -> None:
        """Send a Telegram notification when a live position is moved to breakeven."""
        await send_breakeven_message(symbol, direction, entry_price, current_price, partial_close_qty=partial_close_qty)
