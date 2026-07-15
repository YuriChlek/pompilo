from __future__ import annotations

import os


def get_database_url() -> str:
    """Return the PostgreSQL async database URL for Alembic/runtime setup."""
    explicit_url = os.getenv("BOT_PLATFORM_DATABASE_URL")
    if explicit_url:
        return explicit_url

    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    user = os.getenv("DB_USER", "admin")
    password = os.getenv("DB_PASSWORD", os.getenv("DB_PASS", "admin_pass"))
    database = os.getenv("DB_NAME", os.getenv("DATABASE", "pampilo_db"))
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
