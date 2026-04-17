from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


@dataclass
class DataFetcher:
    """Load candle data from PostgreSQL tables used by the indicator pipeline."""

    db_user: str
    db_pass: str
    db_host: str
    db_port: str
    db_name: str
    _engine: Optional[Engine] = None

    REQUIRED_COLUMNS = ('open', 'high', 'low', 'close', 'volume', 'cvd')

    @property
    def engine(self) -> Engine:
        """Lazily create the SQLAlchemy engine used to access candle tables."""
        if self._engine is None:
            self._engine = create_engine(
                f"postgresql://{self.db_user}:{self.db_pass}@{self.db_host}:{self.db_port}/{self.db_name}"
            )
        return self._engine

    def fetch_candle_data(self, table: str, limit: int = 500) -> pd.DataFrame:
        """Load the latest ``limit`` candles from a table and sort them by time."""
        query = text(
            f"""
            SELECT
                open_time, close_time, symbol, open, close, high, low, cvd, volume
            FROM {table}
            WHERE close_time < (NOW() AT TIME ZONE 'UTC')
            ORDER BY open_time DESC
            LIMIT :limit
            """
        )
        with self.engine.begin() as conn:
            df = pd.read_sql(query, conn, params={'limit': limit})
        df = df.sort_values(by='open_time').reset_index(drop=True)
        self._validate_columns(df)
        return df

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Verify that the loaded DataFrame contains all required columns."""
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Відсутні обов'язкові стовпці: {', '.join(missing)}")
