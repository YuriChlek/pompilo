import argparse
import asyncio
import logging
import sys

from orderflow import OrderFlowScalpBot
from utils.db import close_db_pool


class _ColorFormatter(logging.Formatter):
    RESET = "\033[0m"
    GREEN = "\033[32m"
    RED = "\033[31m"

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        color = self._color_for_record(record)
        if not color:
            return message
        return f"{color}{message}{self.RESET}"

    def _color_for_record(self, record: logging.LogRecord) -> str:
        if record.levelno >= logging.WARNING:
            return self.RED
        if record.levelno == logging.INFO:
            message = record.getMessage().lower()
            if "status=connected" in message or "status=transport_connected" in message:
                return self.GREEN
            if "status=disconnected" in message or "error=" in message:
                return self.RED
            return self.GREEN
        return ""


def _configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ColorFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Order-flow scalp bot entrypoint.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the bot in dry-run mode without sending real orders to Bybit.",
    )
    return parser


async def start() -> None:
    args = _build_parser().parse_args()
    bot = OrderFlowScalpBot(dry_run=args.dry_run)
    try:
        await bot.start()
    finally:
        await bot.close()
        await close_db_pool()


if __name__ == "__main__":
    _configure_logging()
    try:
        asyncio.run(start())
    except KeyboardInterrupt:
        print("Bot stopped by user")
    except Exception as e:
        print(f"💥 Неочікувана помилка: {e}")
        import traceback

        traceback.print_exc()
