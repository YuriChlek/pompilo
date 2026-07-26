from __future__ import annotations

from market_data_service.config.settings import load_database_settings


def get_database_url() -> str:
    return load_database_settings().database_url
