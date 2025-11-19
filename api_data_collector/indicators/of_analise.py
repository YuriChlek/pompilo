import numpy as np
import pandas as pd
import pandas_ta as ta
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from utils import DB_NAME, DB_HOST, DB_PASS, DB_PORT, DB_USER
from sqlalchemy import create_engine, text


@dataclass
class TrendResult:
    """Результат розрахунку технічних індикаторів"""
    atr: float
    rsi: float
    rsi_signal: str
    super_trend: float
    super_trend_signal: str
    mfi: float
    mfi_signal: str
    candle: Dict
    indicators: Dict
    ema: float
    ema_signal: str
    gmma_analysis: Optional[Dict] = None  # Змінено з alligator_analysis на gmma_analysis
    volume_analysis: Optional[Dict] = None
    cvd_analysis: Optional[Dict] = None
    timestamp: Optional[pd.Timestamp] = None


class GMMAAnalyzer:
    """Клас для розрахунку індикатора GMMA (Guppy Multiple Moving Average)"""

    def __init__(self):
        # Короткострокові EMA (3, 5, 8, 10, 12, 15)
        self.short_periods = [3, 5, 8, 10, 12, 15]
        # Довгострокові EMA (30, 35, 40, 45, 50, 60)
        self.long_periods = [30, 35, 40, 45, 50, 60]

    def calculate_gmma(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Розраховує індикатор GMMA.
        """
        df = df.copy()

        # Розрахунок короткострокових EMA
        for period in self.short_periods:
            df[f'gmma_short_{period}'] = ta.ema(df['close'], length=period)

        # Розрахунок довгострокових EMA
        for period in self.long_periods:
            df[f'gmma_long_{period}'] = ta.ema(df['close'], length=period)

        # Розрахунок сигналів GMMA
        df = self._calculate_gmma_signals(df)

        return df

    def _calculate_gmma_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Розраховує сигнали на основі GMMA"""
        df['gmma_signal'] = 'neutral'
        df['gmma_trend'] = 'neutral'
        df['gmma_compression'] = False
        df['gmma_expansion'] = False
        df['gmma_trend_strength'] = 'neutral'

        for i in range(max(self.long_periods), len(df)):
            current_close = df['close'].iloc[i]

            # Отримуємо всі EMA значення
            short_emas = [df[f'gmma_short_{period}'].iloc[i] for period in self.short_periods]
            long_emas = [df[f'gmma_long_{period}'].iloc[i] for period in self.long_periods]

            # Перевірка на наявність значень
            if any(pd.isna(ema) for ema in short_emas + long_emas):
                continue

            # Визначення стану GMMA
            trend = self._check_gmma_trend(short_emas, long_emas)
            compression = self._check_compression(short_emas, long_emas)
            expansion = self._check_expansion(short_emas, long_emas)
            trend_strength = self._check_trend_strength(short_emas, long_emas, trend)

            # Визначення сигналу
            signal = self._determine_gmma_signal(trend, compression, expansion, trend_strength, current_close,
                                                 short_emas)

            df.loc[df.index[i], 'gmma_signal'] = signal
            df.loc[df.index[i], 'gmma_trend'] = trend
            df.loc[df.index[i], 'gmma_compression'] = compression
            df.loc[df.index[i], 'gmma_expansion'] = expansion
            df.loc[df.index[i], 'gmma_trend_strength'] = trend_strength

        return df

    def _check_gmma_trend(self, short_emas: List[float], long_emas: List[float]) -> str:
        """Перевіряє тренд GMMA"""
        avg_short = np.mean(short_emas)
        avg_long = np.mean(long_emas)

        if avg_short > avg_long:
            return 'bullish'
        elif avg_short < avg_long:
            return 'bearish'
        else:
            return 'neutral'

    def _check_compression(self, short_emas: List[float], long_emas: List[float]) -> bool:
        """Перевіряє компресію (зближення) груп EMA"""
        short_range = max(short_emas) - min(short_emas)
        long_range = max(long_emas) - min(long_emas)
        avg_short = np.mean(short_emas)
        avg_long = np.mean(long_emas)

        # Компресія - коли групи зближуються
        distance = abs(avg_short - avg_long)
        return distance < (avg_long * 0.02)  # 2% відстань

    def _check_expansion(self, short_emas: List[float], long_emas: List[float]) -> bool:
        """Перевіряє експансію (розширення) груп EMA"""
        short_range = max(short_emas) - min(short_emas)
        long_range = max(long_emas) - min(long_emas)
        avg_short = np.mean(short_emas)
        avg_long = np.mean(long_emas)

        # Експансія - коли групи розходяться
        distance = abs(avg_short - avg_long)
        return distance > (avg_long * 0.05)  # 5% відстань

    def _check_trend_strength(self, short_emas: List[float], long_emas: List[float], trend: str) -> str:
        """Перевіряє силу тренду"""
        # Перевірка впорядкованості EMA всередині груп
        short_ordered = all(
            short_emas[i] >= short_emas[i + 1] for i in range(len(short_emas) - 1)) if trend == 'bullish' else \
            all(short_emas[i] <= short_emas[i + 1] for i in range(len(short_emas) - 1))

        long_ordered = all(
            long_emas[i] >= long_emas[i + 1] for i in range(len(long_emas) - 1)) if trend == 'bullish' else \
            all(long_emas[i] <= long_emas[i + 1] for i in range(len(long_emas) - 1))

        if short_ordered and long_ordered:
            return 'strong'
        elif short_ordered or long_ordered:
            return 'medium'
        else:
            return 'weak'

    def _determine_gmma_signal(self, trend: str, compression: bool, expansion: bool,
                               trend_strength: str, price: float, short_emas: List[float]) -> str:
        """Визначає сигнал GMMA"""
        if trend == 'bullish':
            if expansion and trend_strength == 'strong' and price > max(short_emas):
                return 'strong_buy'
            elif compression and trend_strength in ['medium', 'strong']:
                return 'buy_compression'
            elif trend_strength in ['medium', 'strong']:
                return 'buy'
            else:
                return 'hold_bullish'

        elif trend == 'bearish':
            if expansion and trend_strength == 'strong' and price < min(short_emas):
                return 'strong_sell'
            elif compression and trend_strength in ['medium', 'strong']:
                return 'sell_compression'
            elif trend_strength in ['medium', 'strong']:
                return 'sell'
            else:
                return 'hold_bearish'

        else:
            return 'hold'

    def get_gmma_analysis(self, df: pd.DataFrame) -> Dict:
        """Повертає аналіз GMMA для поточного стану"""
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

        # Збираємо всі EMA значення
        short_emas = {}
        long_emas = {}

        for period in self.short_periods:
            ema_value = latest.get(f'gmma_short_{period}', current_close)
            short_emas[f'ema_{period}'] = ema_value

        for period in self.long_periods:
            ema_value = latest.get(f'gmma_long_{period}', current_close)
            long_emas[f'ema_{period}'] = ema_value

        # Обчислюємо середні значення
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
    """Аналіз об'ємів та ATR для виявлення акумуляції/дистрибуції"""

    def __init__(self, atr_period: int = 14):
        self.atr_period = atr_period

    def calculate_volume_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Розраховує аналіз об'ємів та ATR.
        """
        df = df.copy()

        # Розрахунок ATR
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)

        # Розрахунок ковзних середніх об'ємів
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()

        # Аналіз об'ємів та ATR
        df = self._calculate_volume_atr_signals(df)

        return df

    def _calculate_volume_atr_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Розраховує сигнали на основі об'ємів та ATR"""
        df['volume_signal'] = 'neutral'
        df['accumulation_signal'] = 'neutral'
        df['volume_trend'] = 'neutral'
        df['atr_trend'] = 'neutral'

        for i in range(20, len(df)):  # Починаємо з 20 для маючи достатньо даних для MA
            current_volume = df['volume'].iloc[i]
            current_atr = df['atr'].iloc[i]
            current_close = df['close'].iloc[i]
            prev_close = df['close'].iloc[i - 1]

            # Поточні ковзні середні об'ємів
            volume_ma_5 = df['volume_ma_5'].iloc[i]
            volume_ma_10 = df['volume_ma_10'].iloc[i]
            volume_ma_20 = df['volume_ma_20'].iloc[i]

            # ATR за попередній період для порівняння
            prev_atr = df['atr'].iloc[i - 1] if i > 0 else current_atr

            # Визначення тренду об'ємів
            if current_volume > volume_ma_10 > volume_ma_20:
                volume_trend = 'rising'
            elif current_volume < volume_ma_10 < volume_ma_20:
                volume_trend = 'falling'
            else:
                volume_trend = 'neutral'

            # Визначення тренду ATR
            if current_atr > prev_atr:
                atr_trend = 'rising'
            elif current_atr < prev_atr:
                atr_trend = 'falling'
            else:
                atr_trend = 'stable'

            # Аналіз акумуляції/дистрибуції
            accumulation_signal = self._analyze_accumulation_distribution(
                current_volume, volume_ma_10, current_atr, prev_atr, current_close, prev_close
            )

            # Сигнал об'ємів
            volume_signal = self._determine_volume_signal(
                current_volume, volume_ma_5, volume_ma_10, volume_ma_20
            )

            df.loc[df.index[i], 'volume_signal'] = volume_signal
            df.loc[df.index[i], 'accumulation_signal'] = accumulation_signal
            df.loc[df.index[i], 'volume_trend'] = volume_trend
            df.loc[df.index[i], 'atr_trend'] = atr_trend

        return df

    def _analyze_accumulation_distribution(self, current_volume: float, volume_ma_10: float,
                                           current_atr: float, prev_atr: float,
                                           current_close: float, prev_close: float) -> str:
        """
        Аналізує акумуляцію/дистрибуцію на основі об'ємів та ATR.
        """
        # Перевірка на акумуляцію (volume росте, ATR падає)
        volume_growing = current_volume > volume_ma_10 * 1.1  # Об'єм вище середнього на 10%
        atr_falling = current_atr < prev_atr * 0.9  # ATR впав на 10%
        price_consolidating = abs(current_close - prev_close) / prev_close < 0.02  # Ціна в консолідації (±2%)

        if volume_growing and atr_falling and price_consolidating:
            return 'accumulation'

        # Перевірка на дистрибуцію (volume росте, ATR росте, ціна падає)
        volume_high = current_volume > volume_ma_10 * 1.2  # Дуже високий об'єм
        atr_rising = current_atr > prev_atr * 1.1  # ATR зріс на 10%
        price_falling = current_close < prev_close * 0.98  # Ціна впала на 2%

        if volume_high and atr_rising and price_falling:
            return 'distribution'

        # Перевірка на прорив (volume росте, ATR росте, ціна росте)
        price_rising = current_close > prev_close * 1.02  # Ціна зросла на 2%
        if volume_growing and atr_rising and price_rising:
            return 'breakout'

        return 'neutral'

    def _determine_volume_signal(self, current_volume: float, volume_ma_5: float,
                                 volume_ma_10: float, volume_ma_20: float) -> str:
        """Визначає сигнал на основі об'ємів"""
        if pd.isna(volume_ma_5) or pd.isna(volume_ma_10) or pd.isna(volume_ma_20):
            return 'neutral'

        # Сильний об'ємний сигнал
        if current_volume > volume_ma_5 * 1.5 and current_volume > volume_ma_10 * 1.3:
            return 'very_high'
        elif current_volume > volume_ma_5 * 1.3 and current_volume > volume_ma_10 * 1.2:
            return 'high'
        elif current_volume > volume_ma_5 * 1.1:
            return 'above_average'
        elif current_volume < volume_ma_5 * 0.9 and current_volume < volume_ma_10 * 0.8:
            return 'low'
        else:
            return 'normal'

    def get_volume_analysis(self, df: pd.DataFrame) -> Dict:
        """Повертає аналіз об'ємів та ATR для поточного стану"""
        if len(df) == 0:
            return {
                'current_volume': 0,
                'volume_ma_5': 0,
                'volume_ma_10': 0,
                'volume_ma_20': 0,
                'atr': 0,
                'volume_signal': 'neutral',
                'accumulation_signal': 'neutral',
                'volume_trend': 'neutral',
                'atr_trend': 'neutral'
            }

        latest = df.iloc[-1]

        return {
            'current_volume': latest.get('volume', 0),
            'volume_ma_5': latest.get('volume_ma_5', 0),
            'volume_ma_10': latest.get('volume_ma_10', 0),
            'volume_ma_20': latest.get('volume_ma_20', 0),
            'atr': latest.get('atr', 0),
            'volume_signal': latest.get('volume_signal', 'neutral'),
            'accumulation_signal': latest.get('accumulation_signal', 'neutral'),
            'volume_trend': latest.get('volume_trend', 'neutral'),
            'atr_trend': latest.get('atr_trend', 'neutral')
        }


class CVDAnalyzer:
    """Аналіз Cumulative Volume Delta"""

    def __init__(self, cvd_threshold: float = 0.7):
        self.cvd_threshold = cvd_threshold

    def analyze(self, candle: pd.Series, df: pd.DataFrame) -> Dict:
        """
        Виконує аналіз CVD поточної свічки.

        Args:
            candle: Поточна свічка.
            df: Історичні дані.

        Returns:
            Словник зі значенням, трендом, силою та дивергенцією.
        """
        cvd_trend = self._determine_cvd_trend(candle, df)
        cvd_strength = self._calculate_cvd_strength(candle, df)
        confidence = self._calculate_cvd_confidence(df)

        # Комбінована оцінка
        signal_quality = self._assess_signal_quality(
            cvd_trend, cvd_strength, confidence
        )

        return {
            'value': candle['cvd'],
            'trend': cvd_trend,
            'strength': cvd_strength,
            'confidence': round(confidence, 2),
            'signal_quality': signal_quality,
            'timestamp': candle.name if hasattr(candle, 'name') else None
        }

    def _assess_signal_quality(self, trend: str, strength: str, confidence: float) -> str:
        """Оцінює загальну якість сигналу."""
        quality_score = 0

        # Бали за тренд
        if trend != "neutral":
            quality_score += 1

        # Бали за силу
        strength_scores = {'weak': 0, 'medium': 1, 'strong': 2}
        quality_score += strength_scores.get(strength, 0)

        # Множник довіри
        quality_score *= confidence

        if quality_score >= 3:
            return "high"
        elif quality_score >= 1.5:
            return "medium"
        else:
            return "low"

    def _calculate_cvd_confidence(self, df: pd.DataFrame) -> float:
        """
        Розраховує рівень довіри до сигналів CVD.
        """
        if len(df) < 20:
            return 0.5

        # Обсяг підтвердження (перетворюємо boolean в float)
        volume_trend = float(df['volume'].tail(5).mean() > df['volume'].tail(20).mean())

        # Консистентність тренду
        cvd_trend_consistency = self._calculate_trend_consistency(df)

        # Волатильність ринку
        market_volatility = df['close'].pct_change().std()
        if pd.isna(market_volatility):
            market_volatility = 0.1

        # Комбінована довіра
        confidence = (cvd_trend_consistency * 0.5 +
                      volume_trend * 0.3 +
                      (1 - min(market_volatility, 0.1)) * 0.2)

        return max(0, min(1, confidence))

    def _calculate_trend_consistency(self, df: pd.DataFrame, period: int = 10) -> float:
        """Розраховує консистентність тренду CVD."""
        if len(df) < period:
            return 0.5

        cvd_changes = df['cvd'].diff().tail(period)
        if len(cvd_changes) == 0:
            return 0.5

        consistent_moves = (cvd_changes > 0).sum() if cvd_changes.mean() > 0 else (cvd_changes < 0).sum()

        return consistent_moves / period

    def _determine_cvd_trend(self, candle: pd.Series, df: pd.DataFrame) -> str:
        """
        Визначає тренд CVD з використанням ковзних середніх та підтвердження.
        """
        if len(df) < 5:
            return "neutral"

        current_cvd = candle['cvd']
        prev_cvd = df['cvd'].iloc[-2]

        # Ковзна середня для згладжування
        cvd_ma_5 = df['cvd'].tail(5).mean()
        cvd_ma_10 = df['cvd'].tail(min(10, len(df))).mean()

        # Мульти-таймфрейм аналіз
        short_trend = "bullish" if current_cvd > prev_cvd else "bearish"

        # Підтвердження ковзними середніми
        if current_cvd > cvd_ma_5 > cvd_ma_10:
            return "bullish"
        elif current_cvd < cvd_ma_5 < cvd_ma_10:
            return "bearish"

        # Додаткова перевірка міцності тренду
        cvd_slope = self._calculate_cvd_slope(df)
        if abs(cvd_slope) > 0.1:  # Порог для значущого нахилу
            if cvd_slope > 0 and short_trend == "bullish":
                return "bullish"
            elif cvd_slope < 0 and short_trend == "bearish":
                return "bearish"

        return "neutral"

    def _calculate_cvd_slope(self, df: pd.DataFrame, period: int = 5) -> float:
        """Розраховує нахил CVD за останній період."""
        if len(df) < period:
            return 0.0

        recent_cvd = df['cvd'].tail(period).values
        x = np.arange(len(recent_cvd))
        slope = np.polyfit(x, recent_cvd, 1)[0]
        return float(slope)

    def _calculate_cvd_strength(self, candle: pd.Series, df: pd.DataFrame) -> str:
        """
        Розраховує силу сигналу CVD.
        """
        if len(df) < 10:
            return "weak"

        current_cvd = candle['cvd']
        prev_cvd = df['cvd'].iloc[-2]
        cvd_change = current_cvd - prev_cvd

        # Відносна зміна
        if prev_cvd != 0:
            relative_change = abs(cvd_change / abs(prev_cvd))
        else:
            relative_change = 0

        # Стандартне відхилення для контексту
        cvd_std = df['cvd'].tail(20).std()
        if pd.isna(cvd_std):
            cvd_std = 0

        avg_cvd_change = df['cvd'].diff().abs().tail(20).mean()
        if pd.isna(avg_cvd_change):
            avg_cvd_change = 0

        # Комбінована оцінка сили з використанням avg_cvd_change
        if cvd_std > 0:
            z_score = abs(cvd_change) / cvd_std
        else:
            z_score = 0

        # Нормалізація зміни відносно середньої зміни
        if avg_cvd_change > 0:
            normalized_change = abs(cvd_change) / avg_cvd_change
        else:
            normalized_change = 0

        # Оновлена формула з використанням усіх метрик
        strength_score = (
                z_score * 0.4 +  # 40% - статистична значущість
                normalized_change * 0.4 +  # 40% - відносно середньої зміни
                relative_change * 0.2  # 20% - відносна зміна
        )

        # Класифікація
        if strength_score > 2.0:
            return 'strong'
        elif strength_score > 1.0:
            return 'medium'
        else:
            return 'weak'


class EMAAnalyzer:
    """Клас для розрахунку EMA 50 індикатора"""

    def __init__(self, period: int = 50):
        self.period = period

    def calculate_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Розраховує EMA 50 індикатор з використанням pandas_ta.
        """
        df = df.copy()

        # Розрахунок EMA 50 за допомогою pandas_ta
        df['ema'] = ta.ema(df['high'], length=self.period)

        # Розрахунок сигналів EMA 50
        df = self._calculate_ema_signals(df)

        return df

    def _calculate_ema_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Розраховує сигнали на основі EMA 50"""
        df['ema_signal'] = 'neutral'
        df['ema_trend'] = 'neutral'
        df['ema_strength'] = 'neutral'

        for i in range(1, len(df)):
            current_close = df['close'].iloc[i]
            current_ema = df['ema'].iloc[i] if not pd.isna(df['ema'].iloc[i]) else current_close
            prev_close = df['close'].iloc[i - 1]
            prev_ema = df['ema'].iloc[i - 1] if not pd.isna(df['ema'].iloc[i - 1]) else prev_close

            # Визначення тренду
            if current_close > current_ema and prev_close <= prev_ema:
                trend = 'bullish_cross'
                signal = 'buy'
                strength = 'strong'
            elif current_close < current_ema and prev_close >= prev_ema:
                trend = 'bearish_cross'
                signal = 'sell'
                strength = 'strong'
            elif current_close > current_ema:
                trend = 'bullish'
                signal = 'hold_bullish'
                strength = 'medium'
            elif current_close < current_ema:
                trend = 'bearish'
                signal = 'hold_bearish'
                strength = 'medium'
            else:
                trend = 'neutral'
                signal = 'hold'
                strength = 'weak'

            # Розрахунок відстані від EMA для сили сигналу
            if current_ema != 0:
                distance_percent = abs((current_close - current_ema) / current_ema) * 100
                if distance_percent > 2.0:
                    strength = 'strong'
                elif distance_percent > 1.0:
                    strength = 'medium'
                else:
                    strength = 'weak'

            df.loc[df.index[i], 'ema_signal'] = signal
            df.loc[df.index[i], 'ema_trend'] = trend
            df.loc[df.index[i], 'ema_strength'] = strength

        return df

    def get_ema_analysis(self, df: pd.DataFrame) -> Dict:
        """Повертає аналіз EMA 50 для поточного стану"""
        if len(df) == 0:
            return {
                'ema': 0,
                'signal': 'neutral',
                'trend': 'neutral',
                'strength': 'neutral',
                'distance_percent': 0
            }

        latest = df.iloc[-1]
        current_close = latest['close']
        current_ema = latest['ema'] if not pd.isna(latest['ema']) else current_close

        # Розрахунок відстані у відсотках
        if current_ema != 0:
            distance_percent = ((current_close - current_ema) / current_ema) * 100
        else:
            distance_percent = 0

        return {
            'ema': current_ema,
            'signal': latest.get('ema_signal', 'neutral'),
            'trend': latest.get('ema_trend', 'neutral'),
            'strength': latest.get('ema_strength', 'neutral'),
            'distance_percent': round(distance_percent, 2)
        }


class MFIAnalyzer:
    """Клас для розрахунку MFI (Money Flow Index) індикатора"""

    def __init__(self, mfi_period: int = 14):
        self.mfi_period = mfi_period

    def calculate_mfi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Розраховує MFI індикатор з використанням pandas_ta.
        """
        df = df.copy()

        # Розрахунок MFI за допомогою pandas_ta
        df['mfi'] = ta.mfi(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume'],
            length=self.mfi_period
        )

        # Розрахунок сигналів MFI
        df = self._calculate_mfi_signals(df)

        return df

    def _calculate_mfi_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Розраховує сигнали на основі MFI"""
        df['mfi_signal'] = 'neutral'
        df['mfi_strength'] = 'neutral'
        df['mfi_trend'] = 'neutral'

        for i in range(len(df)):
            mfi = df['mfi'].iloc[i] if not pd.isna(df['mfi'].iloc[i]) else 50

            # Визначення сигналу
            if mfi > 70:
                signal = 'overbought'
            elif mfi < 30:
                signal = 'oversold'
            else:
                signal = 'neutral'

            # Визначення сили
            if mfi > 80 or mfi < 20:
                strength = 'strong'
            elif mfi > 70 or mfi < 30:
                strength = 'medium'
            else:
                strength = 'weak'

            # Визначення тренду
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

    def get_mfi_analysis(self, df: pd.DataFrame) -> Dict:
        """Повертає аналіз MFI для поточного стану"""
        if len(df) == 0:
            return {'mfi': 50, 'signal': 'neutral', 'strength': 'neutral', 'trend': 'neutral'}

        latest = df.iloc[-1]

        return {
            'mfi': latest['mfi'] if not pd.isna(latest['mfi']) else 50,
            'signal': latest['mfi_signal'],
            'strength': latest['mfi_strength'],
            'trend': latest['mfi_trend']
        }


class RSIAnalyzer:
    """Клас для розрахунку RSI індикатора"""

    def __init__(self, rsi_period: int = 14):
        self.rsi_period = rsi_period

    def calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Розраховує RSI індикатор з використанням pandas_ta.
        """
        df = df.copy()

        # Розрахунок RSI за допомогою pandas_ta
        df['rsi'] = ta.rsi(df['close'], length=self.rsi_period)

        # Розрахунок сигналів RSI
        df = self._calculate_rsi_signals(df)

        return df

    def _calculate_rsi_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Розраховує сигнали на основі RSI"""
        df['rsi_signal'] = 'neutral'
        df['rsi_strength'] = 'neutral'
        df['rsi_trend'] = 'neutral'

        for i in range(len(df)):
            rsi = df['rsi'].iloc[i] if not pd.isna(df['rsi'].iloc[i]) else 50

            # Визначення сигналу
            if rsi > 65:
                signal = 'overbought'
            elif rsi < 35:
                signal = 'oversold'
            else:
                signal = 'neutral'

            # Визначення сили
            if rsi > 75 or rsi < 25:
                strength = 'strong'
            elif rsi > 65 or rsi < 35:
                strength = 'medium'
            else:
                strength = 'weak'

            # Визначення тренду
            if i > 0:
                prev_rsi = df['rsi'].iloc[i - 1] if not pd.isna(df['rsi'].iloc[i - 1]) else 50
                if rsi > prev_rsi:
                    trend = 'bullish'
                elif rsi < prev_rsi:
                    trend = 'bearish'
                else:
                    trend = 'neutral'
            else:
                trend = 'neutral'

            df.loc[df.index[i], 'rsi_signal'] = signal
            df.loc[df.index[i], 'rsi_strength'] = strength
            df.loc[df.index[i], 'rsi_trend'] = trend

        return df

    def get_rsi_analysis(self, df: pd.DataFrame) -> Dict:
        """Повертає аналіз RSI для поточного стану"""
        if len(df) == 0:
            return {'rsi': 50, 'signal': 'neutral', 'strength': 'neutral', 'trend': 'neutral'}

        latest = df.iloc[-1]

        return {
            'rsi': latest['rsi'] if not pd.isna(latest['rsi']) else 50,
            'signal': latest['rsi_signal'],
            'strength': latest['rsi_strength'],
            'trend': latest['rsi_trend']
        }


class SuperTrendAnalyzer:
    """Клас для розрахунку SuperTrend індикатора"""

    def __init__(self, period: int = 10, multiplier: float = 3.0):
        self.period = period
        self.multiplier = multiplier

    def calculate_super_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Розраховує SuperTrend індикатор з використанням pandas_ta.
        """
        df = df.copy()

        # Розрахунок SuperTrend за допомогою pandas_ta
        super_trend_result = ta.supertrend(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            period=int(self.period),
            multiplier=self.multiplier
        )

        # Додаємо результати SuperTrend до DataFrame
        if super_trend_result is not None and len(super_trend_result) > 0:
            # SuperTrend повертає кілька стовпців - SUPERT_10_3.0, SUPERTd_10_3.0 тощо
            for col in super_trend_result.columns:
                df[f'st_{col}'] = super_trend_result[col]

        # Розрахунок сигналів SuperTrend
        df = self._calculate_super_trend_signals(df)

        return df

    def _calculate_super_trend_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Розраховує сигнали на основі SuperTrend"""
        # Знаходимо стовпець з напрямком тренду (зазвичай закінчується на 'd')
        trend_columns = [col for col in df.columns if col.startswith('st_SUPERTd')]

        if trend_columns:
            trend_col = trend_columns[0]
            df['super_trend_signal'] = df[trend_col].map({1: 'bullish', -1: 'bearish'})
        else:
            df['super_trend_signal'] = 'neutral'

        # Знаходимо стовпець з значенням SuperTrend (зазвичай закінчується без 'd')
        value_columns = [col for col in df.columns if col.startswith('st_SUPERT_') and not col.endswith('d')]

        if value_columns:
            value_col = value_columns[0]
            df['super_trend_value'] = df[value_col]
        else:
            df['super_trend_value'] = df['close']

        return df

    def get_super_trend_analysis(self, df: pd.DataFrame) -> Dict:
        """Повертає аналіз SuperTrend для поточного стану"""
        if len(df) == 0:
            return {
                'super_trend': 0,
                'signal': 'neutral',
                'value': 0
            }

        latest = df.iloc[-1]

        return {
            'super_trend': latest.get('super_trend_value', latest['close']),
            'signal': latest.get('super_trend_signal', 'neutral'),
            'value': latest.get('super_trend_value', latest['close'])
        }


class DataFetcher:
    """Клас для отримання даних з бази"""

    def __init__(self, db_user: str, db_pass: str, db_host: str, db_port: str, db_name: str):
        self.db_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

    def fetch_candle_data(self, table: str, limit: int = 500) -> pd.DataFrame:
        """
        Отримує дані свічок з таблиці PostgreSQL.
        """
        query = f"""
            SELECT open_time, close_time, symbol, open, close, high, low, cvd, volume
            FROM {table} 
            WHERE close_time < (NOW() AT TIME ZONE 'UTC')
            ORDER BY open_time DESC 
            LIMIT {limit}
        """

        try:
            engine = create_engine(self.db_url)
            with engine.begin() as conn:
                df = pd.read_sql(text(query), conn)

            # Сортування за часом (від старого до нового)
            df = df.sort_values(by='open_time').reset_index(drop=True)

            # Перевірка обов'язкових стовпців
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Відсутні обов'язкові стовпці: {required_columns}")

            return df

        except Exception as e:
            raise Exception(f"Помилка отримання даних з бази: {e}")


class TrendBot:
    """
    Основний клас бота для розрахунку технічних індикаторів
    """

    def __init__(self, rsi_period: int = 14, mfi_period: int = 14):
        self.rsi_analyzer = RSIAnalyzer(rsi_period)
        self.super_trend_analyzer = SuperTrendAnalyzer()
        self.mfi_analyzer = MFIAnalyzer(mfi_period)
        self.ema_analyzer = EMAAnalyzer()
        self.gmma_analyzer = GMMAAnalyzer()  # Змінено з AlligatorAnalyzer на GMMAAnalyzer
        self.volume_analyzer = VolumeAnalyzer()
        self.cvd_analyzer = CVDAnalyzer()
        self.data_fetcher = None

    def initialize_data_fetcher(self, db_user: str, db_pass: str, db_host: str, db_port: str, db_name: str):
        """Ініціалізує об'єкт для отримання даних"""
        self.data_fetcher = DataFetcher(db_user, db_pass, db_host, db_port, db_name)

    def calculate_trend_for_symbol(self, symbol: str, is_test: bool = False) -> TrendResult:
        """
        Основна функція для розрахунку технічних індикаторів.
        """
        if self.data_fetcher is None:
            self.initialize_data_fetcher(DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME)

        # Отримання даних
        if is_test:
            table_name = f"_candles_trading_data.{str(symbol).lower()}_p_candles_test_data"
        else:
            table_name = f"_candles_trading_data.{str(symbol).lower()}_p_candles"

        data = self.data_fetcher.fetch_candle_data(table_name, limit=500)

        if len(data) < 100:
            raise ValueError(f"Недостатньо даних для аналізу. Отримано {len(data)} свічок, потрібно мінімум 100.")

        # Розрахунок GMMA
        data_with_gmma = self.gmma_analyzer.calculate_gmma(data)

        # Розрахунок Volume Analysis (включає ATR)
        data_with_volume = self.volume_analyzer.calculate_volume_analysis(data_with_gmma)

        # Розрахунок EMA 50
        data_with_ema = self.ema_analyzer.calculate_ema(data_with_volume)

        # Розрахунок MFI
        data_with_mfi = self.mfi_analyzer.calculate_mfi(data_with_ema)

        # Розрахунок SuperTrend
        data_with_super_trend = self.super_trend_analyzer.calculate_super_trend(data_with_mfi)

        # Розрахунок RSI
        data_with_indicators = self.rsi_analyzer.calculate_rsi(data_with_super_trend)

        if len(data_with_indicators) == 0:
            raise ValueError(f"Не вдалося розрахувати індикатори. Можливо недостатньо даних після обробки.")

        # Аналіз CVD
        cvd_analysis = self.cvd_analyzer.analyze(data_with_indicators.iloc[-1], data_with_indicators)

        # Аналіз об'ємів
        volume_analysis = self.volume_analyzer.get_volume_analysis(data_with_indicators)

        # Аналіз GMMA
        gmma_analysis = self.gmma_analyzer.get_gmma_analysis(data_with_indicators)

        # Отримання поточних значень
        result = self._get_combined_result(data_with_indicators, cvd_analysis, volume_analysis, gmma_analysis)

        return result

    def _get_combined_result(self, df: pd.DataFrame, cvd_analysis: Dict, volume_analysis: Dict, gmma_analysis: Dict) -> TrendResult:
        """Об'єднує результати всіх індикаторів"""
        if len(df) == 0:
            raise ValueError("DataFrame порожній")

        latest = df.iloc[-1]

        # Отримуємо аналіз RSI
        rsi_analysis = self.rsi_analyzer.get_rsi_analysis(df)

        # Отримуємо аналіз SuperTrend
        super_trend_analysis = self.super_trend_analyzer.get_super_trend_analysis(df)

        # Отримуємо аналіз MFI
        mfi_analysis = self.mfi_analyzer.get_mfi_analysis(df)

        # Отримуємо аналіз EMA 50
        ema_analysis = self.ema_analyzer.get_ema_analysis(df)

        # Формуємо дані останньої свічки
        candle_data = {
            'open_time': latest.get('open_time'),
            'close_time': latest.get('close_time'),
            'open': latest['open'],
            'close': latest['close'],
            'high': latest['high'],
            'low': latest['low'],
            'volume': latest.get('volume', 0),
            'symbol': latest.get('symbol', 'UNKNOWN'),
            'cvd': latest.get('cvd', 0)
        }

        # Оновлюємо словник індикаторів даними всіх індикаторів
        indicators_dict = {
            'close': latest['close'],
            'high': latest['high'],
            'low': latest['low'],
            'open': latest['open'],
            'volume': latest.get('volume', 0),
            'atr': latest.get('atr', 0),
            'rsi': rsi_analysis['rsi'],
            'rsi_signal': rsi_analysis['signal'],
            'rsi_strength': rsi_analysis['strength'],
            'rsi_trend': rsi_analysis['trend'],
            'super_trend': super_trend_analysis['super_trend'],
            'super_trend_signal': super_trend_analysis['signal'],
            'super_trend_value': super_trend_analysis['value'],
            'mfi': mfi_analysis['mfi'],
            'mfi_signal': mfi_analysis['signal'],
            'mfi_strength': mfi_analysis['strength'],
            'mfi_trend': mfi_analysis['trend'],
            'ema': ema_analysis['ema'],
            'ema_signal': ema_analysis['signal'],
            'ema_trend': ema_analysis['trend'],
            'ema_strength': ema_analysis['strength'],
            'ema_distance_percent': ema_analysis['distance_percent'],
            'cvd': latest.get('cvd', 0),
            'cvd_analysis': cvd_analysis,
            'volume_analysis': volume_analysis,
            'gmma_analysis': gmma_analysis  # Змінено з alligator_analysis на gmma_analysis
        }

        return TrendResult(
            atr=latest.get('atr', 0),
            rsi=rsi_analysis['rsi'],
            rsi_signal=rsi_analysis['signal'],
            super_trend=super_trend_analysis['super_trend'],
            super_trend_signal=super_trend_analysis['signal'],
            mfi=mfi_analysis['mfi'],
            mfi_signal=mfi_analysis['signal'],
            ema=ema_analysis['ema'],
            ema_signal=ema_analysis['signal'],
            candle=candle_data,
            indicators=indicators_dict,
            gmma_analysis=gmma_analysis,  # Змінено з alligator_analysis на gmma_analysis
            volume_analysis=volume_analysis,
            cvd_analysis=cvd_analysis,
            timestamp=latest.get('open_time') or latest.get('close_time')
        )


def get_trend_data(symbol: str, is_test: bool = False) -> TrendResult:
    """
    Зручна функція для отримання даних технічних індикаторів.
    """
    bot = TrendBot()
    return bot.calculate_trend_for_symbol(symbol, is_test)


def get_trend_history(symbol: str, period: int = 100, is_test: bool = False) -> pd.DataFrame:
    """
    Отримує історичні дані технічних індикаторів.
    """
    bot = TrendBot()
    bot.initialize_data_fetcher(DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME)

    if is_test:
        table_name = f"_candles_trading_data.{str(symbol).lower()}_p_candles_test_data"
    else:
        table_name = f"_candles_trading_data.{str(symbol).lower()}_p_candles"

    data = bot.data_fetcher.fetch_candle_data(table_name, limit=period + 100)

    if len(data) < 100:
        raise ValueError(f"Недостатньо даних для аналізу. Отримано {len(data)} свічок.")

    # Розрахунок всіх індикаторів
    data_with_gmma = bot.gmma_analyzer.calculate_gmma(data)
    data_with_volume = bot.volume_analyzer.calculate_volume_analysis(data_with_gmma)
    data_with_ema = bot.ema_analyzer.calculate_ema(data_with_volume)
    data_with_mfi = bot.mfi_analyzer.calculate_mfi(data_with_ema)
    data_with_super_trend = bot.super_trend_analyzer.calculate_super_trend(data_with_mfi)
    data_with_indicators = bot.rsi_analyzer.calculate_rsi(data_with_super_trend)

    if len(data_with_indicators) == 0:
        raise ValueError(f"Не вдалося розрахувати історію індикаторів.")

    return data_with_indicators.tail(period)


def get_of_data(symbol: str, is_test: bool = False):
    """
    Основна функція для отримання даних технічних індикаторів.
    """
    try:
        trend_data = get_trend_data(symbol, is_test)
        indicators_history = get_trend_history(symbol, period=5, is_test=is_test)

        if is_test:
            # Вивід результатів
            print("=" * 50, "\n")
            if trend_data.timestamp is not None:
                print(f"Час закриття свічки: {trend_data.timestamp}")
            print("Технічний аналіз успішно завершено")
            print(f"Символ: {symbol}")

            print("\n=== Остання свічка ===")
            print(f"Час: {trend_data.candle['open_time']}")
            print(f"Open: {trend_data.candle['open']:.4f}")
            print(f"High: {trend_data.candle['high']:.4f}")
            print(f"Low: {trend_data.candle['low']:.4f}")
            print(f"Close: {trend_data.candle['close']:.4f}")
            print(f"Volume: {trend_data.candle['volume']:.2f}")
            print(f"CVD: {trend_data.candle.get('cvd', 'N/A')}")

            print("\n=== GMMA (Guppy Multiple Moving Average) ===")
            if trend_data.gmma_analysis:
                gmma = trend_data.gmma_analysis
                print(f"Сигнал: {gmma.get('signal', 'N/A')}")
                print(f"Тренд: {gmma.get('trend', 'N/A')}")
                print(f"Сила тренду: {gmma.get('trend_strength', 'N/A')}")
                print(f"Компресія: {'Так' if gmma.get('compression', False) else 'Ні'}")
                print(f"Експансія: {'Так' if gmma.get('expansion', False) else 'Ні'}")
                print(f"Середня коротких EMA: {gmma.get('avg_short', 'N/A'):.4f}")
                print(f"Середня довгих EMA: {gmma.get('avg_long', 'N/A'):.4f}")
                print(f"Різниця: {(gmma.get('avg_short', 0) - gmma.get('avg_long', 0)):.4f}")

            print("\n=== RSI ===")
            print(f"Поточний RSI: {trend_data.rsi:.2f}")
            print(f"RSI сигнал: {trend_data.rsi_signal}")

            print("\n=== SuperTrend ===")
            print(f"Поточний SuperTrend: {trend_data.super_trend:.4f}")
            print(f"SuperTrend сигнал: {trend_data.super_trend_signal}")

            print("\n=== MFI (Money Flow Index) ===")
            print(f"Поточний MFI: {trend_data.mfi:.2f}")
            print(f"MFI сигнал: {trend_data.mfi_signal}")

            print("\n=== Volume & ATR Analysis ===")
            if trend_data.volume_analysis:
                volume = trend_data.volume_analysis
                print(f"Поточний об'єм: {volume.get('current_volume', 'N/A'):.2f}")
                print(f"ATR: {volume.get('atr', 'N/A'):.4f}")
                print(f"Сигнал об'єму: {volume.get('volume_signal', 'N/A')}")
                print(f"Сигнал акумуляції: {volume.get('accumulation_signal', 'N/A')}")

            print("\n=== CVD (Cumulative Volume Delta) ===")
            if trend_data.cvd_analysis:
                cvd = trend_data.cvd_analysis
                print(f"Тренд CVD: {cvd.get('trend', 'N/A')}")

            print("\n=== Загальна інформація ===")
            print(f"Ціна закриття: {trend_data.indicators['close']:.4f}")
            print(f"ATR: {trend_data.atr:.4f}")

        # Отримання історичних даних для контексту
            print("\nОстанні 5 значень індикаторів:")
            historical_columns = ['open_time', 'close', 'super_trend_value',
                              'super_trend_signal', 'mfi', 'mfi_trend', 'mfi_signal',
                              'ema', 'ema_signal', 'ema_trend', 'volume', 'gmma_signal',
                              'gmma_trend']
            available_columns = [col for col in historical_columns if col in indicators_history.columns]
            print(indicators_history[available_columns].tail())
            print("=" * 50, "\n")
        return trend_data, indicators_history

    except Exception as e:
        print(f"Помилка при отриманні даних: {e}")
        raise


if __name__ == "__main__":
    try:
        # Приклад використання
        symbol = 'SOLUSDT'

        # Отримання поточних даних
        trend_data, indicators_history = get_of_data(symbol)

        # Комбінований аналіз сигналів
        print(f"\n🎯 КОМБІНОВАНИЙ АНАЛІЗ СИГНАЛІВ:")
        print(f"GMMA: {trend_data.gmma_analysis.get('signal', 'N/A') if trend_data.gmma_analysis else 'N/A'}")
        print(f"RSI: {trend_data.rsi_signal}")
        print(f"SuperTrend: {trend_data.super_trend_signal}")
        print(f"MFI: {trend_data.mfi_signal}")
        print(f"EMA 50: {trend_data.ema_signal}")
        if trend_data.volume_analysis:
            print(f"Volume Signal: {trend_data.volume_analysis.get('volume_signal', 'N/A')}")
        if trend_data.cvd_analysis:
            print(f"CVD: {trend_data.cvd_analysis.get('trend', 'N/A')}")

    except Exception as e:
        print(f"Помилка: {e}")
