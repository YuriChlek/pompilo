from __future__ import annotations

import logging

from utils.config import ORDERFLOW_DB_SCHEMA
from utils.db import get_db_pool

from .schema import get_index_statements, get_schema_statement, get_table_statements

logger = logging.getLogger("orderflow.migrations")


class OrderFlowMigrationRunner:
    async def run(self) -> None:
        try:
            pool = await get_db_pool()
            async with pool.acquire() as conn:
                await conn.execute(get_schema_statement())
                existing_tables = await self._get_existing_tables(conn)
                for table_name, statement in get_table_statements().items():
                    if table_name in existing_tables:
                        logger.info("migration skip table=%s schema=%s", table_name, ORDERFLOW_DB_SCHEMA)
                        continue
                    await conn.execute(statement)
                    logger.info("migration create table=%s schema=%s", table_name, ORDERFLOW_DB_SCHEMA)
                for index_name, statement in get_index_statements().items():
                    await conn.execute(statement)
                    logger.info("migration ensure index=%s schema=%s", index_name, ORDERFLOW_DB_SCHEMA)
        except Exception as exc:
            logger.warning("migration run skipped: %s", exc)

    @staticmethod
    async def _get_existing_tables(conn) -> set[str]:
        rows = await conn.fetch(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = $1
            """,
            ORDERFLOW_DB_SCHEMA,
        )
        return {row["table_name"] for row in rows}
