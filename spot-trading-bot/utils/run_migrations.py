from __future__ import annotations

import asyncio
from pathlib import Path

from utils.db_actions import create_connection

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


async def main() -> None:
    conn = await create_connection()
    try:
        for migration_path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            sql = migration_path.read_text(encoding="utf-8")
            await conn.execute(sql)
            print(f"Applied migration {migration_path.name}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
