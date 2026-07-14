from __future__ import annotations

import os
from urllib.parse import quote_plus


def get_database_url() -> str:
    explicit_url = os.getenv("MARKET_DATA_DATABASE_URL") or os.getenv("DATABASE_URL")
    if explicit_url:
        return explicit_url

    user = os.getenv("DB_USER", "admin")
    password = os.getenv("DB_PASS", os.getenv("DB_PASSWORD", "admin_pass"))
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    database = os.getenv("DATABASE", "pompilo_db")
    return (
        "postgresql+asyncpg://"
        f"{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{quote_plus(database)}"
    )
