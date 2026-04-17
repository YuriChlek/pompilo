from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

from .models import FractalResult


class FractalAnalyzer:
    """Calculate price fractals and summarize recent swing structure."""

    def __init__(self, fractal_period: int = 5):
        """Initialize the fractal analyzer with the configured lookback period."""
        self.fractal_period = fractal_period

    def calculate_fractals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate upper and lower fractals for the provided OHLC dataframe."""
        if 'fractal_upper' not in df.columns:
            df['fractal_upper'] = np.nan
        if 'fractal_lower' not in df.columns:
            df['fractal_lower'] = np.nan

        for i in range(self.fractal_period, len(df) - self.fractal_period):
            if self._is_upper_fractal(df, i):
                df.loc[df.index[i], 'fractal_upper'] = df['high'].iloc[i]
            if self._is_lower_fractal(df, i):
                df.loc[df.index[i], 'fractal_lower'] = df['low'].iloc[i]
        return df

    def _is_upper_fractal(self, df: pd.DataFrame, idx: int) -> bool:
        """Check whether the candle at ``idx`` forms an upper fractal."""
        current_high = df['high'].iloc[idx]
        for i in range(1, self.fractal_period + 1):
            if df['high'].iloc[idx - i] >= current_high:
                return False
        for i in range(1, self.fractal_period + 1):
            if df['high'].iloc[idx + i] >= current_high:
                return False
        return True

    def _is_lower_fractal(self, df: pd.DataFrame, idx: int) -> bool:
        """Check whether the candle at ``idx`` forms a lower fractal."""
        current_low = df['low'].iloc[idx]
        for i in range(1, self.fractal_period + 1):
            if df['low'].iloc[idx - i] <= current_low:
                return False
        for i in range(1, self.fractal_period + 1):
            if df['low'].iloc[idx + i] <= current_low:
                return False
        return True

    def get_fractal_analysis(self, df: pd.DataFrame) -> FractalResult:
        """Collect recent fractal statistics from a prepared dataframe."""
        if len(df) == 0:
            return FractalResult([], [], [], [])

        all_upper: List[Dict[str, Any]] = []
        all_lower: List[Dict[str, Any]] = []

        for idx, row in df.iterrows():
            if not pd.isna(row.get('fractal_upper')):
                all_upper.append({
                    'timestamp': idx if hasattr(idx, 'strftime') else str(idx),
                    'price': float(row['fractal_upper']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'index': len(all_upper)
                })
            if not pd.isna(row.get('fractal_lower')):
                all_lower.append({
                    'timestamp': idx if hasattr(idx, 'strftime') else str(idx),
                    'price': float(row['fractal_lower']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'index': len(all_lower)
                })

        last_3_upper = [dict(fr) for fr in reversed(all_upper[-3:])]
        last_3_lower = [dict(fr) for fr in reversed(all_lower[-3:])]

        for i, fractal in enumerate(last_3_upper):
            fractal['position'] = f"{i + 1}/{len(last_3_upper)}"
            fractal['is_newest'] = (i == 0)
        for i, fractal in enumerate(last_3_lower):
            fractal['position'] = f"{i + 1}/{len(last_3_lower)}"
            fractal['is_newest'] = (i == 0)

        return FractalResult(
            upper_fractals=all_upper,
            lower_fractals=all_lower,
            last_3_upper_fractals=last_3_upper,
            last_3_lower_fractals=last_3_lower
        )


class GMMAAnalyzer:
    """Calculate Guppy Multiple Moving Average signals and trend state."""

    def __init__(self):
        """Prepare GMMA EMA period groups used in signal calculations."""
        self.short_periods = [3, 5, 8, 10, 12, 15]
        self.long_periods = [30, 35, 40, 45, 50, 60]

    def calculate_gmma(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate GMMA EMA lines and derived GMMA signals for the dataframe."""
        for period in self.short_periods:
            df[f'gmma_short_{period}'] = ta.ema(df['close'], length=period)
        for period in self.long_periods:
            df[f'gmma_long_{period}'] = ta.ema(df['close'], length=period)
        return self._calculate_gmma_signals(df)

    def _calculate_gmma_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend, compression, expansion, and signal labels from GMMA lines."""
        df['gmma_signal'] = 'neutral'
        df['gmma_trend'] = 'neutral'
        df['gmma_compression'] = False
        df['gmma_expansion'] = False
        df['gmma_trend_strength'] = 'neutral'

        for i in range(max(self.long_periods), len(df)):
            short_emas = [df[f'gmma_short_{period}'].iloc[i] for period in self.short_periods]
            long_emas = [df[f'gmma_long_{period}'].iloc[i] for period in self.long_periods]
            if any(pd.isna(ema) for ema in short_emas + long_emas):
                continue

            trend = self._check_gmma_trend(short_emas, long_emas)
            compression = self._check_compression(short_emas, long_emas)
            expansion = self._check_expansion(short_emas, long_emas)
            trend_strength = self._check_trend_strength(short_emas, long_emas, trend)
            signal = self._determine_gmma_signal(trend, compression, expansion, trend_strength, df['close'].iloc[i], short_emas)

            idx = df.index[i]
            df.loc[idx, 'gmma_signal'] = signal
            df.loc[idx, 'gmma_trend'] = trend
            df.loc[idx, 'gmma_compression'] = compression
            df.loc[idx, 'gmma_expansion'] = expansion
            df.loc[idx, 'gmma_trend_strength'] = trend_strength
        return df

    def _check_gmma_trend(self, short_emas: List[float], long_emas: List[float]) -> str:
        """Determine bullish, bearish, or neutral GMMA trend alignment."""
        avg_short = np.mean(short_emas)
        avg_long = np.mean(long_emas)
        if avg_short > avg_long:
            return 'bullish'
        if avg_short < avg_long:
            return 'bearish'
        return 'neutral'

    def _check_compression(self, short_emas: List[float], long_emas: List[float]) -> bool:
        """Check whether short and long GMMA bands are compressed."""
        distance = abs(np.mean(short_emas) - np.mean(long_emas))
        avg_long = np.mean(long_emas)
        return avg_long != 0 and distance < (avg_long * 0.02)

    def _check_expansion(self, short_emas: List[float], long_emas: List[float]) -> bool:
        """Check whether GMMA bands are expanding away from each other."""
        distance = abs(np.mean(short_emas) - np.mean(long_emas))
        avg_long = np.mean(long_emas)
        return avg_long != 0 and distance > (avg_long * 0.05)

    def _check_trend_strength(self, short_emas: List[float], long_emas: List[float], trend: str) -> str:
        """Estimate GMMA trend strength from EMA ordering."""
        if trend == 'bullish':
            short_ordered = all(short_emas[i] >= short_emas[i + 1] for i in range(len(short_emas) - 1))
            long_ordered = all(long_emas[i] >= long_emas[i + 1] for i in range(len(long_emas) - 1))
        elif trend == 'bearish':
            short_ordered = all(short_emas[i] <= short_emas[i + 1] for i in range(len(short_emas) - 1))
            long_ordered = all(long_emas[i] <= long_emas[i + 1] for i in range(len(long_emas) - 1))
        else:
            return 'neutral'

        if short_ordered and long_ordered:
            return 'strong'
        if short_ordered or long_ordered:
            return 'medium'
        return 'weak'

    def _determine_gmma_signal(
        self,
        trend: str,
        compression: bool,
        expansion: bool,
        trend_strength: str,
        price: float,
        short_emas: List[float],
    ) -> str:
        """Resolve the final GMMA signal label from trend state and band dynamics."""
        if trend == 'bullish':
            if expansion and trend_strength == 'strong' and price > max(short_emas):
                return 'strong_buy'
            if compression and trend_strength in ['medium', 'strong']:
                return 'buy_compression'
            if trend_strength in ['medium', 'strong']:
                return 'buy'
            return 'hold_bullish'
        if trend == 'bearish':
            if expansion and trend_strength == 'strong' and price < min(short_emas):
                return 'strong_sell'
            if compression and trend_strength in ['medium', 'strong']:
                return 'sell_compression'
            if trend_strength in ['medium', 'strong']:
                return 'sell'
            return 'hold_bearish'
        return 'hold'

    def get_gmma_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return the latest GMMA analysis snapshot from a prepared dataframe."""
        if len(df) == 0:
            return {
                'short_emas': {},
                'long_emas': {},
                'signal': 'neutral',
                'trend': 'neutral',
                'compression': False,
                'expansion': False,
                'trend_strength': 'neutral',
                'avg_short': 0,
                'avg_long': 0
            }
        latest = df.iloc[-1]
        current_close = latest['close']
        short_emas = {f'ema_{period}': latest.get(f'gmma_short_{period}', current_close) for period in self.short_periods}
        long_emas = {f'ema_{period}': latest.get(f'gmma_long_{period}', current_close) for period in self.long_periods}
        avg_short = np.mean(list(short_emas.values())) if short_emas else current_close
        avg_long = np.mean(list(long_emas.values())) if long_emas else current_close
        return {
            'short_emas': short_emas,
            'long_emas': long_emas,
            'signal': latest.get('gmma_signal', 'neutral'),
            'trend': latest.get('gmma_trend', 'neutral'),
            'compression': latest.get('gmma_compression', False),
            'expansion': latest.get('gmma_expansion', False),
            'trend_strength': latest.get('gmma_trend_strength', 'neutral'),
            'avg_short': avg_short,
            'avg_long': avg_long
        }


class VolumeAnalyzer:
    """Calculate rolling volume context and simple spike labels."""

    def calculate_volume_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume moving average, spike labels, and ATR for the dataframe."""
        df['volume_ma_24'] = df['volume'].rolling(window=24, min_periods=1).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma_24'] * 3)
        df['volume_signal'] = 'normal'
        df.loc[df['volume_spike'], 'volume_signal'] = 'spike'
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        return df

    def get_volume_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return the latest volume-analysis snapshot from a prepared dataframe."""
        if len(df) == 0:
            return {
                'current_volume': 0,
                'volume_ma_24': 0,
                'volume_spike': False,
                'volume_signal': 'normal',
                'atr': 0,
                'spike_ratio': 0
            }
        latest = df.iloc[-1]
        volume_ma_24 = latest.get('volume_ma_24', 0)
        current_volume = latest.get('volume', 0)
        spike_ratio = current_volume / volume_ma_24 if volume_ma_24 else 0
        return {
            'current_volume': current_volume,
            'volume_ma_24': volume_ma_24,
            'volume_spike': latest.get('volume_spike', False),
            'volume_signal': latest.get('volume_signal', 'normal'),
            'atr': latest.get('atr', 0),
            'spike_ratio': round(spike_ratio, 2)
        }


class CVDAnalyzer:
    """Analyze cumulative volume delta direction, strength, and confidence."""

    def __init__(self, cvd_threshold: float = 0.7):
        """Initialize the CVD analyzer with a confidence threshold."""
        self.cvd_threshold = cvd_threshold

    def analyze(self, candle: pd.Series, df: pd.DataFrame) -> Dict[str, Any]:
        """Build a CVD trend snapshot for the latest candle and recent history."""
        cvd_trend = self._determine_cvd_trend(candle, df)
        cvd_strength = self._calculate_cvd_strength(candle, df)
        confidence = self._calculate_cvd_confidence(df)
        signal_quality = self._assess_signal_quality(cvd_trend, cvd_strength, confidence)
        return {
            'value': candle['cvd'],
            'trend': cvd_trend,
            'strength': cvd_strength,
            'confidence': round(confidence, 2),
            'signal_quality': signal_quality,
            'timestamp': candle.name if hasattr(candle, 'name') else None
        }

    def _assess_signal_quality(self, trend: str, strength: str, confidence: float) -> str:
        """Map trend strength and confidence to a qualitative signal label."""
        quality_score = 0
        if trend != 'neutral':
            quality_score += 1
        strength_scores = {'weak': 0, 'medium': 1, 'strong': 2}
        quality_score += strength_scores.get(strength, 0)
        quality_score *= confidence
        if quality_score >= 3:
            return 'high'
        if quality_score >= 1.5:
            return 'medium'
        return 'low'

    def _calculate_cvd_confidence(self, df: pd.DataFrame) -> float:
        """Calculate confidence in the current CVD direction from recent context."""
        if len(df) < 20:
            return 0.5
        volume_trend = float(df['volume'].tail(5).mean() > df['volume'].tail(20).mean())
        cvd_trend_consistency = self._calculate_trend_consistency(df)
        market_volatility = df['close'].pct_change().std()
        if pd.isna(market_volatility):
            market_volatility = 0.1
        confidence = (
            cvd_trend_consistency * 0.5 +
            volume_trend * 0.3 +
            (1 - min(market_volatility, 0.1)) * 0.2
        )
        return max(0, min(1, confidence))

    def _calculate_trend_consistency(self, df: pd.DataFrame, period: int = 10) -> float:
        """Measure how consistently CVD moved in one direction over recent candles."""
        if len(df) < period:
            return 0.5
        cvd_changes = df['cvd'].diff().tail(period)
        if len(cvd_changes) == 0:
            return 0.5
        if cvd_changes.mean() > 0:
            consistent_moves = (cvd_changes > 0).sum()
        else:
            consistent_moves = (cvd_changes < 0).sum()
        return consistent_moves / period

    def _determine_cvd_trend(self, candle: pd.Series, df: pd.DataFrame) -> str:
        """Determine bullish, bearish, or neutral CVD trend from recent values."""
        if len(df) < 5:
            return 'neutral'
        current_cvd = candle['cvd']
        prev_cvd = df['cvd'].iloc[-2]
        cvd_ma_5 = df['cvd'].tail(5).mean()
        cvd_ma_10 = df['cvd'].tail(min(10, len(df))).mean()
        short_trend = 'bullish' if current_cvd > prev_cvd else 'bearish'
        if current_cvd > cvd_ma_5 > cvd_ma_10:
            return 'bullish'
        if current_cvd < cvd_ma_5 < cvd_ma_10:
            return 'bearish'
        cvd_slope = self._calculate_cvd_slope(df)
        if abs(cvd_slope) > 0.1:
            if cvd_slope > 0 and short_trend == 'bullish':
                return 'bullish'
            if cvd_slope < 0 and short_trend == 'bearish':
                return 'bearish'
        return 'neutral'

    def _calculate_cvd_slope(self, df: pd.DataFrame, period: int = 5) -> float:
        """Calculate a simple linear slope for recent CVD values."""
        if len(df) < period:
            return 0.0
        recent_cvd = df['cvd'].tail(period).values
        x = np.arange(len(recent_cvd))
        slope = np.polyfit(x, recent_cvd, 1)[0]
        return float(slope)

    def _calculate_cvd_strength(self, candle: pd.Series, df: pd.DataFrame) -> str:
        """Estimate qualitative CVD strength from relative recent changes."""
        if len(df) < 10:
            return 'weak'
        current_cvd = candle['cvd']
        prev_cvd = df['cvd'].iloc[-2]
        cvd_change = current_cvd - prev_cvd
        relative_change = abs(cvd_change / abs(prev_cvd)) if prev_cvd != 0 else 0
        cvd_std = df['cvd'].tail(20).std() or 0
        avg_cvd_change = df['cvd'].diff().abs().tail(20).mean() or 0
        z_score = abs(cvd_change) / cvd_std if cvd_std else 0
        normalized_change = abs(cvd_change) / avg_cvd_change if avg_cvd_change else 0
        strength_score = z_score * 0.4 + normalized_change * 0.4 + relative_change * 0.2
        if strength_score > 2.0:
            return 'strong'
        if strength_score > 1.0:
            return 'medium'
        return 'weak'


class MFIAnalyzer:
    """Calculate Money Flow Index values and derive directional labels."""

    def __init__(self, mfi_period: int = 14):
        """Initialize the MFI analyzer with the configured lookback period."""
        self.mfi_period = mfi_period

    def calculate_mfi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Money Flow Index values and derived labels for the dataframe."""
        df['mfi'] = ta.mfi(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], length=self.mfi_period)
        return self._calculate_mfi_signals(df)

    def _calculate_mfi_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend, strength, and signal labels from the MFI series."""
        df['mfi_signal'] = 'neutral'
        df['mfi_strength'] = 'neutral'
        df['mfi_trend'] = 'neutral'
        for i in range(len(df)):
            mfi = df['mfi'].iloc[i] if not pd.isna(df['mfi'].iloc[i]) else 50
            if mfi > 70:
                signal = 'overbought'
            elif mfi < 30:
                signal = 'oversold'
            else:
                signal = 'neutral'
            if mfi > 80 or mfi < 20:
                strength = 'strong'
            elif mfi > 70 or mfi < 30:
                strength = 'medium'
            else:
                strength = 'weak'
            if i > 0 and not pd.isna(df['mfi'].iloc[i - 1]):
                prev_mfi = df['mfi'].iloc[i - 1]
                if mfi > prev_mfi and mfi > 30:
                    trend = 'bullish'
                elif mfi < prev_mfi and mfi < 70:
                    trend = 'bearish'
                else:
                    trend = 'neutral'
            else:
                trend = 'neutral'
            df.loc[df.index[i], 'mfi_signal'] = signal
            df.loc[df.index[i], 'mfi_strength'] = strength
            df.loc[df.index[i], 'mfi_trend'] = trend
        return df

    def get_mfi_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return the latest MFI analysis snapshot from a prepared dataframe."""
        if len(df) == 0:
            return {'mfi': 50, 'signal': 'neutral', 'strength': 'neutral', 'trend': 'neutral'}
        latest = df.iloc[-1]
        return {
            'mfi': latest['mfi'] if not pd.isna(latest['mfi']) else 50,
            'signal': latest.get('mfi_signal', 'neutral'),
            'strength': latest.get('mfi_strength', 'neutral'),
            'trend': latest.get('mfi_trend', 'neutral')
        }


class SuperTrendAnalyzer:
    """Calculate SuperTrend values and expose the latest directional snapshot."""

    def __init__(self, period: int = 10, multiplier: float = 4):
        """Initialize SuperTrend parameters used in trend calculations."""
        self.period = period
        self.multiplier = multiplier

    def calculate_super_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate SuperTrend values and signal labels for the dataframe."""
        super_trend_result = ta.supertrend(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            period=int(self.period),
            multiplier=self.multiplier
        )
        if super_trend_result is not None and len(super_trend_result) > 0:
            for col in super_trend_result.columns:
                df[f'st_{col}'] = super_trend_result[col]
        return self._calculate_super_trend_signals(df)

    def _calculate_super_trend_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive trend labels from calculated SuperTrend values."""
        trend_columns = [col for col in df.columns if col.startswith('st_SUPERTd')]
        if trend_columns:
            trend_col = trend_columns[0]
            df['super_trend_signal'] = df[trend_col].map({1: 'bullish', -1: 'bearish'}).fillna('neutral')
        else:
            df['super_trend_signal'] = 'neutral'
        value_columns = [col for col in df.columns if col.startswith('st_SUPERT_') and not col.endswith('d')]
        if value_columns:
            value_col = value_columns[0]
            df['super_trend_value'] = df[value_col]
        else:
            df['super_trend_value'] = df['close']
        return df

    def get_super_trend_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return the latest SuperTrend analysis snapshot from a prepared dataframe."""
        if len(df) == 0:
            return {'super_trend': 0, 'signal': 'neutral', 'value': 0}
        latest = df.iloc[-1]
        return {
            'super_trend': latest.get('super_trend_value', latest['close']),
            'signal': latest.get('super_trend_signal', 'neutral'),
            'value': latest.get('super_trend_value', latest['close'])
        }
