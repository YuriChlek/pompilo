from __future__ import annotations

from sqlalchemy import MetaData

BOT_PLATFORM_SCHEMA = "_bot_platform"

metadata = MetaData(
    schema=BOT_PLATFORM_SCHEMA,
    naming_convention={
        "ix": "%(table_name)s_%(column_0_name)s_idx",
        "uq": "%(table_name)s_%(column_0_N_name)s_uq",
        "ck": "%(table_name)s_%(constraint_name)s_ck",
        "fk": "%(table_name)s_%(column_0_N_name)s_fk",
        "pk": "%(table_name)s_pk",
    },
)
