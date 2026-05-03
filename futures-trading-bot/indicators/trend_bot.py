from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
from trading.domain.strategy_config import DEFAULT_STRATEGY_CONFIG, StrategyConfig

from utils.config import ANALYSIS_CANDLE_LIMIT, DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER

from .analyzers import (
    CVDAnalyzer,
    FractalAnalyzer,
    GMMAAnalyzer,
    MFIAnalyzer,
    SuperTrendAnalyzer,
    VolumeAnalyzer,
)
from .data_fetcher import DataFetcher
from .models import FractalResult, RangeMarketResult, TrendResult
from .range import RangeMarketDetector


class TrendBot:
    """Compose indicator analyzers and expose prepared trend snapshots for one symbol."""

    def __init__(
        self,
        data_fetcher: Optional[DataFetcher] = None,
        strategy_config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
    ):
        """Initialize indicator analyzers and an optional historical candle fetcher."""
        self.strategy_config = strategy_config
        self.super_trend_analyzer = SuperTrendAnalyzer()
        self.mfi_analyzer = MFIAnalyzer()
        self.gmma_analyzer = GMMAAnalyzer()
        self.volume_analyzer = VolumeAnalyzer()
        self.cvd_analyzer = CVDAnalyzer()
        self.fractal_analyzer = FractalAnalyzer()
        self.range_detector = RangeMarketDetector()
        self.data_fetcher = data_fetcher

    def initialize_data_fetcher(self, db_user: str, db_pass: str, db_host: str, db_port: str, db_name: str) -> None:
        """Create the data fetcher used to read historical candles from PostgreSQL."""
        self.data_fetcher = DataFetcher(db_user, db_pass, db_host, db_port, db_name)

    def calculate_trend_for_symbol(self, symbol: str, is_test: bool = False) -> TrendResult:
        """Return the full current indicator snapshot for a symbol."""
        raw_h1 = self._fetch_symbol_data(symbol, limit=ANALYSIS_CANDLE_LIMIT)
        return self.calculate_trend_from_dataframe(raw_h1)

    def calculate_history_for_symbol(self, symbol: str, period: int = 100, is_test: bool = False) -> pd.DataFrame:
        """Return prepared H1 indicator history for the latest ``period`` candles."""
        raw_h1 = self._fetch_symbol_data(symbol, limit=period + 100)
        return self.calculate_history_from_dataframe(raw_h1, period=period)

    def calculate_trend_and_history(
        self, symbol: str, period: int = 5, is_test: bool = False
    ) -> Tuple[TrendResult, pd.DataFrame]:
        """Return both the current ``TrendResult`` and a short indicator history window."""
        limit = max(ANALYSIS_CANDLE_LIMIT, period + 100)
        raw_h1 = self._fetch_symbol_data(symbol, limit=limit)
        return self.calculate_trend_and_history_from_dataframe(raw_h1, period=period)

    def calculate_trend_from_dataframe(self, raw_h1: pd.DataFrame) -> TrendResult:
        """Return the full current indicator snapshot for an already loaded H1 dataset."""
        prepared_h1 = self._prepare_h1_indicators(raw_h1.copy())
        prepared_h4 = self._prepare_h4_indicators(raw_h1)
        prepared_d1 = self._prepare_d1_indicators(raw_h1)
        return self._assemble_trend_result(prepared_h1, prepared_h4, prepared_d1)

    def calculate_history_from_dataframe(self, raw_h1: pd.DataFrame, period: int = 100) -> pd.DataFrame:
        """Return H1 indicator history for an already loaded candle dataset."""
        prepared_h1 = self._prepare_h1_indicators(raw_h1.copy())
        if len(prepared_h1) == 0:
            raise ValueError("Не вдалося розрахувати історію індикаторів.")
        return prepared_h1.tail(period)

    def calculate_trend_and_history_from_dataframe(
        self, raw_h1: pd.DataFrame, period: int = 5
    ) -> Tuple[TrendResult, pd.DataFrame]:
        """Return a ``TrendResult`` and indicator history for an already loaded H1 dataset."""
        prepared_h1 = self._prepare_h1_indicators(raw_h1.copy())
        prepared_h4 = self._prepare_h4_indicators(raw_h1)
        prepared_d1 = self._prepare_d1_indicators(raw_h1)
        trend_result = self._assemble_trend_result(prepared_h1, prepared_h4, prepared_d1)
        history = prepared_h1.tail(period)
        return trend_result, history

    def _fetch_symbol_data(self, symbol: str, limit: int) -> pd.DataFrame:
        """Load raw symbol candles from the primary storage table."""
        if self.data_fetcher is None:
            self.initialize_data_fetcher(DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME)
        table_name = f"_candles_trading_data.{symbol.lower()}_1h"
        data = self.data_fetcher.fetch_candle_data(table_name, limit=limit)
        if len(data) < 100:
            raise ValueError(
                f"Недостатньо даних для аналізу. Отримано {len(data)} свічок, потрібно мінімум 100."
            )
        return data

    def _prepare_h1_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the full H1 indicator set for the provided DataFrame."""
        df = self.fractal_analyzer.calculate_fractals(df)
        df = self.gmma_analyzer.calculate_gmma(df)
        df = self.volume_analyzer.calculate_volume_analysis(df)
        df = self.mfi_analyzer.calculate_mfi(df)
        df = self.super_trend_analyzer.calculate_super_trend(df)
        return df

    def _prepare_h4_indicators(self, df_h1: pd.DataFrame) -> pd.DataFrame:
        """Resample H1 candles to H4 and add the required higher-timeframe indicators."""
        df_h4 = self._resample_h1_to_h4(df_h1)
        if df_h4.empty:
            return df_h4
        df_h4 = self.fractal_analyzer.calculate_fractals(df_h4)
        df_h4 = self.super_trend_analyzer.calculate_super_trend(df_h4)
        return df_h4

    def _prepare_d1_indicators(self, df_h1: pd.DataFrame) -> pd.DataFrame:
        """Resample H1 candles to D1 and add the required regime indicators."""
        df_d1 = self._resample_h1_to_d1(df_h1)
        if df_d1.empty:
            return df_d1
        df_d1 = self.fractal_analyzer.calculate_fractals(df_d1)
        df_d1 = self.super_trend_analyzer.calculate_super_trend(df_d1)
        return df_d1

    def _resample_h1_to_h4(self, df_h1: pd.DataFrame) -> pd.DataFrame:
        """Convert H1 candles into H4 series for multi-timeframe analysis."""
        if len(df_h1) < 4 or 'open_time' not in df_h1.columns:
            return pd.DataFrame()
        df = df_h1.copy()
        df['open_time'] = pd.to_datetime(df['open_time'])
        df.set_index('open_time', inplace=True)
        df_h4 = df.resample('4h', label='right', closed='right').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'cvd': 'last'
        }).dropna()
        if df_h4.empty:
            return df_h4
        if 'symbol' in df.columns:
            df_h4['symbol'] = df['symbol'].iloc[-1]
        df_h4 = df_h4.reset_index()
        return df_h4

    def _resample_h1_to_d1(self, df_h1: pd.DataFrame) -> pd.DataFrame:
        """Convert H1 candles into D1 series for market-regime analysis."""
        if len(df_h1) < 24 or 'open_time' not in df_h1.columns:
            return pd.DataFrame()
        df = df_h1.copy()
        df['open_time'] = pd.to_datetime(df['open_time'])
        df.set_index('open_time', inplace=True)
        df_d1 = df.resample('1d', label='right', closed='right').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'cvd': 'last'
        }).dropna()
        if df_d1.empty:
            return df_d1
        if 'symbol' in df.columns:
            df_d1['symbol'] = df['symbol'].iloc[-1]
        df_d1 = df_d1.reset_index()
        return df_d1

    def _assemble_trend_result(self, df_h1: pd.DataFrame, df_h4: pd.DataFrame, df_d1: pd.DataFrame) -> TrendResult:
        """Assemble the final ``TrendResult`` from H1/H4 analysis and derived indicators."""
        if len(df_h1) == 0:
            raise ValueError("DataFrame H1 порожній")

        latest_h1 = df_h1.iloc[-1]
        latest_h4 = df_h4.iloc[-1] if len(df_h4) else None
        latest_d1 = df_d1.iloc[-1] if len(df_d1) else None

        super_trend_analysis = self.super_trend_analyzer.get_super_trend_analysis(df_h1)
        super_trend_h4_value = (
            latest_h4.get('super_trend_value', latest_h1['close']) if latest_h4 is not None else latest_h1['close']
        )
        super_trend_h4_signal = (
            latest_h4.get('super_trend_signal', 'neutral') if latest_h4 is not None else 'neutral'
        )
        super_trend_d1_value = (
            latest_d1.get('super_trend_value', latest_h1['close']) if latest_d1 is not None else latest_h1['close']
        )
        super_trend_d1_signal = (
            latest_d1.get('super_trend_signal', 'neutral') if latest_d1 is not None else 'neutral'
        )

        mfi_analysis = self.mfi_analyzer.get_mfi_analysis(df_h1)
        volume_analysis = self.volume_analyzer.get_volume_analysis(df_h1)
        gmma_analysis = self.gmma_analyzer.get_gmma_analysis(df_h1)
        cvd_analysis = self.cvd_analyzer.analyze(latest_h1, df_h1)

        fractal_analysis_h1 = self.fractal_analyzer.get_fractal_analysis(df_h1)
        if len(df_h4):
            fractal_analysis_h4 = self.fractal_analyzer.get_fractal_analysis(df_h4)
            range_analysis_h4 = self.range_detector.detect_range_market(df_h4, fractal_analysis_h4)
        else:
            fractal_analysis_h4 = FractalResult([], [], [], [])
            range_analysis_h4 = RangeMarketResult.empty()

        if len(df_d1):
            fractal_analysis_d1 = self.fractal_analyzer.get_fractal_analysis(df_d1)
            range_analysis_d1 = self.range_detector.detect_range_market(df_d1, fractal_analysis_d1)
        else:
            fractal_analysis_d1 = FractalResult([], [], [], [])
            range_analysis_d1 = RangeMarketResult.empty()

        atr_series = df_h1['atr'] if 'atr' in df_h1.columns else None
        range_analysis_h1 = self.range_detector.detect_range_market(
            df_h1, fractal_analysis_h1, atr_series=atr_series
        )
        candle_data = {
            'open_time': latest_h1.get('open_time'),
            'close_time': latest_h1.get('close_time'),
            'open': latest_h1['open'],
            'close': latest_h1['close'],
            'high': latest_h1['high'],
            'low': latest_h1['low'],
            'volume': latest_h1.get('volume', 0),
            'symbol': latest_h1.get('symbol', 'UNKNOWN'),
            'cvd': latest_h1.get('cvd', 0)
        }

        indicators_dict = {
            'close': latest_h1['close'],
            'high': latest_h1['high'],
            'low': latest_h1['low'],
            'open': latest_h1['open'],
            'volume': latest_h1.get('volume', 0),
            'atr': latest_h1.get('atr', 0),
            'super_trend': super_trend_analysis['super_trend'],
            'super_trend_signal': super_trend_analysis['signal'],
            'super_trend_value': super_trend_analysis['value'],
            'super_trend_h4': super_trend_h4_value,
            'super_trend_h4_signal': super_trend_h4_signal,
            'super_trend_d1': super_trend_d1_value,
            'super_trend_d1_signal': super_trend_d1_signal,
            'mfi': mfi_analysis['mfi'],
            'mfi_signal': mfi_analysis['signal'],
            'mfi_strength': mfi_analysis['strength'],
            'mfi_trend': mfi_analysis['trend'],
            'cvd': latest_h1.get('cvd', 0),
            'cvd_analysis': cvd_analysis,
            'volume_analysis': volume_analysis,
            'gmma_analysis': gmma_analysis,
            'fractal_analysis_h1': {
                'last_3_upper_fractals': fractal_analysis_h1.last_3_upper_fractals,
                'last_3_lower_fractals': fractal_analysis_h1.last_3_lower_fractals,
                'total_upper_fractals': len(fractal_analysis_h1.upper_fractals),
                'total_lower_fractals': len(fractal_analysis_h1.lower_fractals)
            },
            'fractal_analysis_h4': {
                'last_3_upper_fractals': fractal_analysis_h4.last_3_upper_fractals,
                'last_3_lower_fractals': fractal_analysis_h4.last_3_lower_fractals,
                'total_upper_fractals': len(fractal_analysis_h4.upper_fractals),
                'total_lower_fractals': len(fractal_analysis_h4.lower_fractals)
            },
            'fractal_analysis_d1': {
                'last_3_upper_fractals': fractal_analysis_d1.last_3_upper_fractals,
                'last_3_lower_fractals': fractal_analysis_d1.last_3_lower_fractals,
                'total_upper_fractals': len(fractal_analysis_d1.upper_fractals),
                'total_lower_fractals': len(fractal_analysis_d1.lower_fractals)
            },
            'range_analysis_h1': {
                'is_range': range_analysis_h1.is_range,
                'confidence': range_analysis_h1.confidence,
                'reasons': range_analysis_h1.reasons,
                'atr_pct': range_analysis_h1.atr_pct,
                'fractal_pattern': range_analysis_h1.fractal_pattern,
                'avg_upper': range_analysis_h1.avg_upper,
                'avg_lower': range_analysis_h1.avg_lower,
                'range_type': range_analysis_h1.range_type,
                'cluster_strength': range_analysis_h1.cluster_strength,
                'price_levels': range_analysis_h1.price_levels,
                'upper_fractals_analyzed': range_analysis_h1.upper_fractals_analyzed,
                'lower_fractals_analyzed': range_analysis_h1.lower_fractals_analyzed
            },
            'range_analysis_h4': {
                'is_range': range_analysis_h4.is_range,
                'confidence': range_analysis_h4.confidence,
                'reasons': range_analysis_h4.reasons,
                'atr_pct': range_analysis_h4.atr_pct,
                'fractal_pattern': range_analysis_h4.fractal_pattern,
                'avg_upper': range_analysis_h4.avg_upper,
                'avg_lower': range_analysis_h4.avg_lower,
                'range_type': range_analysis_h4.range_type,
                'cluster_strength': range_analysis_h4.cluster_strength,
                'price_levels': range_analysis_h4.price_levels,
                'upper_fractals_analyzed': range_analysis_h4.upper_fractals_analyzed,
                'lower_fractals_analyzed': range_analysis_h4.lower_fractals_analyzed
            },
            'range_analysis_d1': {
                'is_range': range_analysis_d1.is_range,
                'confidence': range_analysis_d1.confidence,
                'reasons': range_analysis_d1.reasons,
                'atr_pct': range_analysis_d1.atr_pct,
                'fractal_pattern': range_analysis_d1.fractal_pattern,
                'avg_upper': range_analysis_d1.avg_upper,
                'avg_lower': range_analysis_d1.avg_lower,
                'range_type': range_analysis_d1.range_type,
                'cluster_strength': range_analysis_d1.cluster_strength,
                'price_levels': range_analysis_d1.price_levels,
                'upper_fractals_analyzed': range_analysis_d1.upper_fractals_analyzed,
                'lower_fractals_analyzed': range_analysis_d1.lower_fractals_analyzed
            }
        }

        return TrendResult(
            atr=latest_h1.get('atr', 0),
            super_trend=super_trend_analysis['super_trend'],
            super_trend_signal=super_trend_analysis['signal'],
            super_trend_h4=super_trend_h4_value,
            super_trend_h4_signal=super_trend_h4_signal,
            super_trend_d1=super_trend_d1_value,
            super_trend_d1_signal=super_trend_d1_signal,
            mfi=mfi_analysis['mfi'],
            mfi_signal=mfi_analysis['signal'],
            candle=candle_data,
            indicators=indicators_dict,
            gmma_analysis=gmma_analysis,
            volume_analysis=volume_analysis,
            cvd_analysis=cvd_analysis,
            fractal_analysis_h1=fractal_analysis_h1,
            fractal_analysis_h4=fractal_analysis_h4,
            fractal_analysis_d1=fractal_analysis_d1,
            range_analysis_h1=range_analysis_h1,
            range_analysis_h4=range_analysis_h4,
            range_analysis_d1=range_analysis_d1,
            timestamp=latest_h1.get('open_time') or latest_h1.get('close_time'),
            funding_rate=latest_h1.get('funding_rate'),
        )
