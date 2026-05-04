from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from utils.config import DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER


@dataclass
class BacktestDataLoader:
    """Load historical candle series from PostgreSQL for backtest runs."""

    db_user: str = DB_USER
    db_pass: str = DB_PASS
    db_host: str = DB_HOST
    db_port: str = str(DB_PORT)
    db_name: str = DB_NAME
    _engine: Optional[Engine] = None

    @property
    def engine(self) -> Engine:
        """Lazily create the SQLAlchemy engine used for historical PostgreSQL queries."""
        if self._engine is None:
            self._engine = create_engine(
                f"postgresql://{self.db_user}:{self.db_pass}@{self.db_host}:{self.db_port}/{self.db_name}"
            )
        return self._engine

    def load_symbol_history(
        self,
        symbol: str,
        *,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Load historical candles for a symbol within the requested date range."""
        table_name = f"_candles_trading_data.{symbol.lower()}_1h"
        conditions = ["close_time < (NOW() AT TIME ZONE 'UTC')"]
        params = {}

        if date_from is not None:
            conditions.append("open_time >= :date_from")
            params["date_from"] = date_from
        if date_to is not None:
            conditions.append("open_time <= :date_to")
            params["date_to"] = date_to

        where_clause = " AND ".join(conditions)
        query = text(
            f"""
            SELECT
                open_time, close_time, symbol, open, close, high, low, cvd, volume
            FROM {table_name}
            WHERE {where_clause}
            ORDER BY open_time ASC
            """
        )

        with self.engine.begin() as conn:
            df = pd.read_sql(query, conn, params=params)

        if df.empty:
            raise ValueError(f"Не знайдено історичних даних для {symbol}")

        return df.reset_index(drop=True)
