import asyncio
from typing import Sequence

from utils.config import TRADING_SYMBOLS
from utils.db_actions import create_tables
from utils.logging import setup_logging

DEFAULT_SUFFIXES: Sequence[str] = (
    '_1h',
    '_4h',
)


async def main() -> None:
    """Create the standard candle tables for all configured trading symbols."""
    await create_tables(symbols=TRADING_SYMBOLS, suffixes=DEFAULT_SUFFIXES)


if __name__ == "__main__":
    setup_logging()
    asyncio.run(main())
