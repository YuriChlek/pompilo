from __future__ import annotations

from sqlalchemy import MetaData

MARKET_DATA_SCHEMA = "_market_data"

metadata = MetaData(
    schema=MARKET_DATA_SCHEMA,
    naming_convention={
        "ix": "%(table_name)s_%(column_0_name)s_idx",
        "uq": "%(table_name)s_%(column_0_N_name)s_uq",
        "ck": "%(table_name)s_%(constraint_name)s_ck",
        "fk": "%(table_name)s_%(column_0_N_name)s_fk",
        "pk": "%(table_name)s_pk",
    },
)
