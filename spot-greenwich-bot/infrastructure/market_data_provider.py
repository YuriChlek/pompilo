from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text

from utils.config import ANALYSIS_WINDOW, CANDLES_DATA_SCHEMA, DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER, H4_ANALYSIS_WINDOW
from utils.db_actions import d1_table_name, h4_table_name


class DatabaseMarketDataProvider:
    """Load candle history for one symbol from PostgreSQL."""

    def __init__(self) -> None:
        self.engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

    def _load_history(self, table_name: str, limit: int) -> pd.DataFrame:
        query = text(
            f"""
            SELECT open_time, close_time, symbol, open, high, low, close, volume
            FROM {CANDLES_DATA_SCHEMA}.{table_name}
            WHERE close_time < (NOW() AT TIME ZONE 'UTC')
            ORDER BY open_time DESC
            LIMIT :limit
            """
        )
        with self.engine.begin() as conn:
            df = pd.read_sql(query, conn, params={"limit": limit})
        return df.sort_values(by="open_time").reset_index(drop=True)

    def get_symbol_history(self, symbol: str) -> pd.DataFrame:
        """Return the D1 analysis window of candles for one symbol ordered from oldest to newest."""

        return self._load_history(d1_table_name(symbol), ANALYSIS_WINDOW)


class MultiTimeframeMarketDataProvider(DatabaseMarketDataProvider):
    """Load D1 and H4 candle history for one symbol from PostgreSQL."""

    def get_d1_history(self, symbol: str) -> pd.DataFrame:
        return self._load_history(d1_table_name(symbol), ANALYSIS_WINDOW)

    def get_h4_history(self, symbol: str) -> pd.DataFrame:
        return self._load_history(h4_table_name(symbol), H4_ANALYSIS_WINDOW)

    def get_symbol_history(self, symbol: str) -> dict[str, pd.DataFrame]:
        """Return D1 regime and H4 execution histories for one symbol."""

        return {
            "d1": self.get_d1_history(symbol),
            "h4": self.get_h4_history(symbol),
        }
