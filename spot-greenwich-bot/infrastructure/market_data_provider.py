from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text

from utils.config import ANALYSIS_WINDOW, CANDLES_DATA_SCHEMA, DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER
from utils.db_actions import d1_table_name


class DatabaseMarketDataProvider:
    """Load candle history for one symbol from PostgreSQL."""

    def __init__(self) -> None:
        self.engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

    def get_symbol_history(self, symbol: str) -> pd.DataFrame:
        """Return the analysis window of candles for one symbol ordered from oldest to newest."""

        table_name = d1_table_name(symbol)
        query = text(
            f"""
            SELECT open_time, close_time, symbol, open, high, low, close, volume, cvd
            FROM {CANDLES_DATA_SCHEMA}.{table_name}
            WHERE close_time < (NOW() AT TIME ZONE 'UTC')
            ORDER BY open_time DESC
            LIMIT :limit
            """
        )
        with self.engine.begin() as conn:
            df = pd.read_sql(query, conn, params={"limit": ANALYSIS_WINDOW})
        return df.sort_values(by="open_time").reset_index(drop=True)
