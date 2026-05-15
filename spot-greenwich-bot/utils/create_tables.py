from __future__ import annotations

import asyncio

from utils.db_actions import create_tables


async def main() -> None:
    await create_tables()


if __name__ == "__main__":
    asyncio.run(main())
