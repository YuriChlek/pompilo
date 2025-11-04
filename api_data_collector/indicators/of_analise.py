from pprint import pprint
import numpy as np
import pandas as pd
import pandas_ta as ta
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from utils import DB_NAME, DB_HOST, DB_PASS, DB_PORT, DB_USER
from sqlalchemy import create_engine, text


@dataclass
class AlphaTrendResult:
    """Результат розрахунку AlphaTrend індикатора"""
    alpha_trend: float
    atr: float
    alpha_trend_signal: str
    rsi: float
    rsi_signal: str
    super_trend: float
    super_trend_signal: str
    mfi: float  # Додано MFI
    mfi_signal: str  # Додано MFI сигнал
    candle: Dict
    indicators: Dict
    cvd_analysis: Optional[Dict] = None  # Додано CVD аналіз
    sinewave_analysis: Optional[Dict] = None  # Додано SineWave аналіз
    timestamp: Optional[pd.Timestamp] = None


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


class SineWaveAnalyzer:
    """Клас для розрахунку Even Better SineWave індикатора"""

    def __init__(self, period: int = 40):
        self.period = period

    def calculate_sinewave(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Розраховує Even Better SineWave індикатор.
        """
        df = df.copy()

        try:
            # Використовуємо pandas_ta для розрахунку Even Better SineWave
            sinewave_result = ta.ebsw(df['close'], length=self.period)

            if sinewave_result is not None:
                # Додаємо результати до DataFrame
                df['sinewave'] = sinewave_result

                # Розрахунок сигналів SineWave
                df = self._calculate_sinewave_signals(df)
            else:
                # Резервний варіант, якщо індикатор не працює
                df = self._calculate_basic_sinewave(df)

        except Exception as e:
            print(f"Помилка розрахунку SineWave: {e}")
            # Резервний розрахунок
            df = self._calculate_basic_sinewave(df)

        return df

    def _calculate_basic_sinewave(self, df: pd.DataFrame) -> pd.DataFrame:
        """Резервний метод розрахунку SineWave"""
        # Проста реалізація на основі ковзних середніх
        df['sinewave'] = ta.ema(df['close'], length=self.period)
        df = self._calculate_sinewave_signals(df)
        return df

    def _calculate_sinewave_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Розраховує сигнали на основі SineWave"""
        df['sinewave_signal'] = 'neutral'
        df['sinewave_trend'] = 'neutral'
        df['sinewave_strength'] = 'neutral'

        for i in range(1, len(df)):
            current_sine = df['sinewave'].iloc[i] if not pd.isna(df['sinewave'].iloc[i]) else 0
            prev_sine = df['sinewave'].iloc[i - 1] if not pd.isna(df['sinewave'].iloc[i - 1]) else 0

            # Визначення тренду
            if current_sine > prev_sine:
                trend = 'bullish'
            elif current_sine < prev_sine:
                trend = 'bearish'
            else:
                trend = 'neutral'

            # Визначення сили на основі зміни
            sine_change = abs(current_sine - prev_sine)
            if i > 1:
                avg_change = df['sinewave'].diff().abs().tail(10).mean()
                if not pd.isna(avg_change) and avg_change > 0:
                    relative_strength = sine_change / avg_change
                    if relative_strength > 2.0:
                        strength = 'strong'
                    elif relative_strength > 1.0:
                        strength = 'medium'
                    else:
                        strength = 'weak'
                else:
                    strength = 'neutral'
            else:
                strength = 'neutral'

            # Визначення сигналу
            if trend == 'bullish' and strength in ['medium', 'strong']:
                signal = 'buy'
            elif trend == 'bearish' and strength in ['medium', 'strong']:
                signal = 'sell'
            else:
                signal = 'hold'

            df.loc[df.index[i], 'sinewave_signal'] = signal
            df.loc[df.index[i], 'sinewave_trend'] = trend
            df.loc[df.index[i], 'sinewave_strength'] = strength

        return df

    def get_sinewave_analysis(self, df: pd.DataFrame) -> Dict:
        """Повертає аналіз SineWave для поточного стану"""
        if len(df) == 0:
            return {
                'sinewave': 0,
                'signal': 'hold',
                'trend': 'neutral',
                'strength': 'neutral'
            }

        latest = df.iloc[-1]

        return {
            'sinewave': latest['sinewave'] if not pd.isna(latest['sinewave']) else 0,
            'signal': latest.get('sinewave_signal', 'hold'),
            'trend': latest.get('sinewave_trend', 'neutral'),
            'strength': latest.get('sinewave_strength', 'neutral')
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
                if mfi > prev_mfi and mfi > 40:
                    trend = 'bullish'
                elif mfi < prev_mfi and mfi < 60:
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

    def __init__(self, period: int = 10, multiplier: float = 3):
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


class AlphaTrendAnalyzer:
    """Клас для розрахунку AlphaTrend індикатора"""

    def __init__(self, atr_period: int = 10, atr_multiplier: float = 3):
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def calculate_alpha_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Розраховує AlphaTrend індикатор для DataFrame.
        """
        df = df.copy()

        # Розрахунок ATR
        atr = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)
        df['atr'] = atr

        # Розрахунок AlphaTrend
        df = self._calculate_alpha_trend_signal(df)

        # Видаляємо тільки рядки де alpha_trend все ще NaN (перші atr_period + 1 рядків)
        return df[df['alpha_trend'].notna()]

    def _calculate_alpha_trend_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Розраховує сигнали AlphaTrend на основі ATR.
        """
        # Ініціалізація стовпців
        df['alpha_trend'] = np.nan
        df['alpha_trend_signal'] = 'neutral'
        df['trend_direction'] = 0

        if len(df) == 0:
            return df

        # Чекаємо поки ATR буде розраховано
        start_index = self.atr_period
        if len(df) <= start_index:
            return df

        # Початкові значення
        df.loc[df.index[start_index], 'alpha_trend'] = df['close'].iloc[start_index]
        df.loc[df.index[start_index], 'trend_direction'] = 1

        # Розрахунок AlphaTrend для решти точок
        for i in range(start_index + 1, len(df)):
            current_close = df['close'].iloc[i]
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            current_atr = df['atr'].iloc[i]

            prev_alpha_trend = df['alpha_trend'].iloc[i - 1]
            prev_trend_direction = df['trend_direction'].iloc[i - 1]

            # AlphaTrend logic
            if prev_trend_direction == 1:  # Попередній тренд bullish
                alpha_trend_value = prev_alpha_trend
                new_trend_direction = 1

                if current_close < prev_alpha_trend - self.atr_multiplier * current_atr:
                    alpha_trend_value = max(prev_alpha_trend, current_high - self.atr_multiplier * current_atr)
                    new_trend_direction = -1
                elif current_high > prev_alpha_trend:
                    alpha_trend_value = current_high - self.atr_multiplier * current_atr

            else:  # Попередній тренд bearish
                alpha_trend_value = prev_alpha_trend
                new_trend_direction = -1

                if current_close > prev_alpha_trend + self.atr_multiplier * current_atr:
                    alpha_trend_value = min(prev_alpha_trend, current_low + self.atr_multiplier * current_atr)
                    new_trend_direction = 1
                elif current_low < prev_alpha_trend:
                    alpha_trend_value = current_low + self.atr_multiplier * current_atr

            # Визначення сигналу
            signal = 'bullish' if new_trend_direction == 1 else 'bearish'

            df.loc[df.index[i], 'alpha_trend'] = alpha_trend_value
            df.loc[df.index[i], 'trend_direction'] = new_trend_direction
            df.loc[df.index[i], 'alpha_trend_signal'] = signal

        return df


class MarketTrendAnalyzer:
    """Покращений аналізатор тренду з Bollinger Bands"""

    def __init__(self, bb_period: int = 20, bb_std: int = 2):
        self.bb_period = bb_period
        self.bb_std = bb_std

    def determine_trend_with_bb(self, df: pd.DataFrame) -> str:
        """
        Визначає тренд з використанням Bollinger Bands + MA
        """
        if len(df) < self.bb_period:
            return "neutral"

        # Розраховуємо Bollinger Bands
        df = self._calculate_bollinger_bands(df)

        current_price = df['close'].iloc[-1]
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        bb_middle = df['bb_middle'].iloc[-1]

        # Комбінована логіка
        bb_signal = self._analyze_bb_position(current_price, bb_upper, bb_lower, bb_middle)
        ma_signal = self._analyze_ma_cross(df)
        volatility_signal = self._analyze_bb_volatility(df)

        return self._combine_signals(bb_signal, ma_signal, volatility_signal)

    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Розраховує Bollinger Bands"""
        df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()
        df['bb_std'] = df['close'].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * self.bb_std)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * self.bb_std)
        return df

    def _analyze_bb_position(self, price: float, bb_upper: float, bb_lower: float, bb_middle: float) -> str:
        """
        Аналізує позицію ціни відносно Bollinger Bands
        """
        bb_position = (price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5

        if price > bb_upper:
            return "strong_bullish"  # Сильний бичачий тренд
        elif price < bb_lower:
            return "strong_bearish"  # Сильний ведмежий тренд
        elif bb_position > 0.7:
            return "bullish"  # Бичачий тренд (верхня частина каналу)
        elif bb_position < 0.3:
            return "bearish"  # Ведмежий тренд (нижня частина каналу)
        elif bb_position > 0.5:
            return "weak_bullish"  # Слабкий бичачий
        elif bb_position < 0.5:
            return "weak_bearish"  # Слабкий ведмежий
        else:
            return "neutral"

    def _analyze_ma_cross(self, df: pd.DataFrame) -> str:
        """Аналіз перетину ковзних середніх"""
        ma_fast = df['close'].rolling(20).mean()
        ma_slow = df['close'].rolling(50).mean()

        if ma_fast.iloc[-1] > ma_slow.iloc[-1] and ma_fast.iloc[-2] <= ma_slow.iloc[-2]:
            return "bullish_cross"
        elif ma_fast.iloc[-1] < ma_slow.iloc[-1] and ma_fast.iloc[-2] >= ma_slow.iloc[-2]:
            return "bearish_cross"
        elif ma_fast.iloc[-1] > ma_slow.iloc[-1]:
            return "bullish_ma"
        elif ma_fast.iloc[-1] < ma_slow.iloc[-1]:
            return "bearish_ma"
        else:
            return "neutral_ma"

    def _analyze_bb_volatility(self, df: pd.DataFrame) -> str:
        """Аналіз волатильності за Bollinger Bands"""
        bb_width = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        current_width = bb_width.iloc[-1]
        avg_width = bb_width.mean()

        if current_width > avg_width * 1.5:
            return "high_volatility"  # Сильний тренд
        elif current_width < avg_width * 0.7:
            return "low_volatility"  # Консолідація
        else:
            return "normal_volatility"

    def _combine_signals(self, bb_signal: str, ma_signal: str, volatility_signal: str) -> str:
        """Комбінує всі сигнали в остаточний тренд"""

        # Ваги для різних сигналів
        signals_score = {
            "strong_bullish": 3, "bullish": 2, "weak_bullish": 1,
            "strong_bearish": -3, "bearish": -2, "weak_bearish": -1,
            "bullish_cross": 2, "bearish_cross": -2,
            "bullish_ma": 1, "bearish_ma": -1,
            "high_volatility": 1, "low_volatility": -1
        }

        score = (signals_score.get(bb_signal, 0) +
                 signals_score.get(ma_signal, 0) +
                 signals_score.get(volatility_signal, 0))

        if score >= 3:
            return "bullish"
        elif score <= -3:
            return "bearish"
        else:
            return "neutral"


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


class AlphaTrendBot:
    """
    Основний клас бота для розрахунку AlphaTrend індикатора
    """

    def __init__(self, atr_period: int = 14, atr_multiplier: float = 1.5, rsi_period: int = 14, mfi_period: int = 14,
                 sinewave_period: int = 60):
        self.alpha_trend_analyzer = AlphaTrendAnalyzer(atr_period, atr_multiplier)
        self.rsi_analyzer = RSIAnalyzer(rsi_period)
        self.super_trend_analyzer = SuperTrendAnalyzer()
        self.mfi_analyzer = MFIAnalyzer(mfi_period)
        self.cvd_analyzer = CVDAnalyzer()  # Додано CVD аналізатор
        self.sinewave_analyzer = SineWaveAnalyzer(sinewave_period)  # Додано SineWave аналізатор
        self.market_trend_analyzer = MarketTrendAnalyzer()  # Додано MarketTrendAnalyzer
        self.data_fetcher = None

    def initialize_data_fetcher(self, db_user: str, db_pass: str, db_host: str, db_port: str, db_name: str):
        """Ініціалізує об'єкт для отримання даних"""
        self.data_fetcher = DataFetcher(db_user, db_pass, db_host, db_port, db_name)

    def calculate_alpha_trend_for_symbol(self, symbol: str, is_test: bool = False) -> AlphaTrendResult:
        """
        Основна функція для розрахунку AlphaTrend індикатора.
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

        # Розрахунок SineWave (додаємо першим, щоб мати більше даних для аналізу)
        data_with_sinewave = self.sinewave_analyzer.calculate_sinewave(data)

        # Розрахунок MFI
        data_with_mfi = self.mfi_analyzer.calculate_mfi(data_with_sinewave)

        # Розрахунок SuperTrend
        data_with_super_trend = self.super_trend_analyzer.calculate_super_trend(data_with_mfi)

        # Розрахунок RSI
        data_with_rsi = self.rsi_analyzer.calculate_rsi(data_with_super_trend)

        # Розрахунок AlphaTrend
        data_with_indicators = self.alpha_trend_analyzer.calculate_alpha_trend(data_with_rsi)

        if len(data_with_indicators) == 0:
            raise ValueError(f"Не вдалося розрахувати AlphaTrend. Можливо недостатньо даних після обробки.")

        # Аналіз CVD
        cvd_analysis = self.cvd_analyzer.analyze(data_with_indicators.iloc[-1], data_with_indicators)

        # Аналіз SineWave
        sinewave_analysis = self.sinewave_analyzer.get_sinewave_analysis(data_with_indicators)

        # Аналіз ринкового тренду з використанням MarketTrendAnalyzer
        alpha_trend_data = self.market_trend_analyzer.determine_trend_with_bb(data_with_indicators)

        # Отримання поточних значень
        result = self._get_combined_result(data_with_indicators, cvd_analysis, sinewave_analysis, alpha_trend_data)

        return result

    def _get_combined_result(self, df: pd.DataFrame, cvd_analysis: Dict, sinewave_analysis: Dict, alpha_trend_data: str) -> AlphaTrendResult:
        """Об'єднує результати AlphaTrend, RSI, SuperTrend, MFI, CVD та SineWave аналізу"""
        if len(df) == 0:
            raise ValueError("DataFrame порожній")

        latest = df.iloc[-1]

        # Отримуємо аналіз RSI
        rsi_analysis = self.rsi_analyzer.get_rsi_analysis(df)

        # Отримуємо аналіз SuperTrend
        super_trend_analysis = self.super_trend_analyzer.get_super_trend_analysis(df)

        # Отримуємо аналіз MFI
        mfi_analysis = self.mfi_analyzer.get_mfi_analysis(df)

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
            'cvd': latest.get('cvd', 0)  # Додано CVD
        }

        # Оновлюємо словник індикаторів даними всіх індикаторів
        indicators_dict = {
            'close': latest['close'],
            'high': latest['high'],
            'low': latest['low'],
            'open': latest['open'],
            'volume': latest.get('volume', 0),
            'atr': latest['atr'],
            'alpha_trend': latest['alpha_trend'],
            'trend_direction': latest.get('trend_direction', 0),
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
            'cvd': latest.get('cvd', 0),
            'cvd_analysis': cvd_analysis,
            'sinewave': sinewave_analysis['sinewave'],
            'sinewave_signal': sinewave_analysis['signal'],
            'sinewave_trend': sinewave_analysis['trend'],
            'sinewave_strength': sinewave_analysis['strength'],
            'sinewave_analysis': sinewave_analysis,
            'market_trend': alpha_trend_data  # Додано результати MarketTrendAnalyzer
        }

        return AlphaTrendResult(
            alpha_trend=latest['alpha_trend'],
            atr=latest['atr'],
            alpha_trend_signal=latest['alpha_trend_signal'],
            rsi=rsi_analysis['rsi'],
            rsi_signal=rsi_analysis['signal'],
            super_trend=super_trend_analysis['super_trend'],
            super_trend_signal=super_trend_analysis['signal'],
            mfi=mfi_analysis['mfi'],
            mfi_signal=mfi_analysis['signal'],
            candle=candle_data,
            indicators=indicators_dict,
            cvd_analysis=cvd_analysis,
            sinewave_analysis=sinewave_analysis,  # Додано SineWave аналіз до результату
            timestamp=latest.get('open_time') or latest.get('close_time')
        )


def get_alpha_trend_data(symbol: str, is_test: bool = False) -> AlphaTrendResult:
    """
    Зручна функція для отримання даних AlphaTrend індикатора.
    """
    bot = AlphaTrendBot()
    return bot.calculate_alpha_trend_for_symbol(symbol, is_test)


def get_alpha_trend_history(symbol: str, period: int = 100, is_test: bool = False) -> pd.DataFrame:
    """
    Отримує історичні дані AlphaTrend індикатора.
    """
    bot = AlphaTrendBot()
    bot.initialize_data_fetcher(DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME)

    if is_test:
        table_name = f"_candles_trading_data.{str(symbol).lower()}_p_candles_test_data"
    else:
        table_name = f"_candles_trading_data.{str(symbol).lower()}_p_candles"

    data = bot.data_fetcher.fetch_candle_data(table_name, limit=period + 100)

    if len(data) < 100:
        raise ValueError(f"Недостатньо даних для аналізу. Отримано {len(data)} свічок.")

    # Розрахунок всіх індикаторів включаючи SineWave
    data_with_sinewave = bot.sinewave_analyzer.calculate_sinewave(data)
    data_with_mfi = bot.mfi_analyzer.calculate_mfi(data_with_sinewave)
    data_with_super_trend = bot.super_trend_analyzer.calculate_super_trend(data_with_mfi)
    data_with_rsi = bot.rsi_analyzer.calculate_rsi(data_with_super_trend)
    data_with_indicators = bot.alpha_trend_analyzer.calculate_alpha_trend(data_with_rsi)

    if len(data_with_indicators) == 0:
        raise ValueError(f"Не вдалося розрахувати AlphaTrend історію.")

    return data_with_indicators.tail(period)


def get_of_data(symbol: str, is_test: bool = False):
    """
    Основна функція для отримання даних AlphaTrend.
    """
    try:
        alpha_trend_data = get_alpha_trend_data(symbol, is_test)
        indicators_history = get_alpha_trend_history(symbol, period=5, is_test=is_test)

        # Отримуємо результати MarketTrendAnalyzer з alpha_trend_data
        alpha_trend_data_value = alpha_trend_data.indicators.get('market_trend', 'neutral')

        if is_test:
            # Вивід результатів
            if alpha_trend_data.timestamp is not None:
                print(f"Час закриття свічки: {alpha_trend_data.timestamp}")
            print("Технічний аналіз успішно завершено")
            print(f"Символ: {symbol}")

            print("\n=== Остання свічка ===")
            print(f"Час: {alpha_trend_data.candle['open_time']}")
            print(f"Open: {alpha_trend_data.candle['open']:.4f}")
            print(f"High: {alpha_trend_data.candle['high']:.4f}")
            print(f"Low: {alpha_trend_data.candle['low']:.4f}")
            print(f"Close: {alpha_trend_data.candle['close']:.4f}")
            print(f"Volume: {alpha_trend_data.candle['volume']:.2f}")
            print(f"CVD: {alpha_trend_data.candle.get('cvd', 'N/A')}")

            print("\n=== AlphaTrend ===")
            print(f"Поточний AlphaTrend: {alpha_trend_data.alpha_trend:.4f}")
            print(f"ATR: {alpha_trend_data.atr:.4f}")
            print(f"Сигнал: {alpha_trend_data.alpha_trend_signal}")

            print("\n=== RSI ===")
            print(f"Поточний RSI: {alpha_trend_data.rsi:.2f}")
            print(f"RSI сигнал: {alpha_trend_data.rsi_signal}")
            print(f"Сила сигналу: {alpha_trend_data.indicators['rsi_strength']}")
            print(f"Тренд RSI: {alpha_trend_data.indicators['rsi_trend']}")

            print("\n=== SuperTrend ===")
            print(f"Поточний SuperTrend: {alpha_trend_data.super_trend:.4f}")
            print(f"SuperTrend сигнал: {alpha_trend_data.super_trend_signal}")

            print("\n=== MFI (Money Flow Index) ===")
            print(f"Поточний MFI: {alpha_trend_data.mfi:.2f}")
            print(f"MFI сигнал: {alpha_trend_data.mfi_signal}")
            print(f"Сила сигналу MFI: {alpha_trend_data.indicators['mfi_strength']}")
            print(f"Тренд MFI: {alpha_trend_data.indicators['mfi_trend']}")

            print("\n=== CVD (Cumulative Volume Delta) ===")
            if alpha_trend_data.cvd_analysis:
                cvd = alpha_trend_data.cvd_analysis
                print(f"Значення CVD: {cvd.get('value', 'N/A')}")
                print(f"Тренд CVD: {cvd.get('trend', 'N/A')}")
                print(f"Сила CVD: {cvd.get('strength', 'N/A')}")
                print(f"Довіра: {cvd.get('confidence', 'N/A')}")
                print(f"Якість сигналу: {cvd.get('signal_quality', 'N/A')}")

            print("\n=== Even Better SineWave ===")
            if alpha_trend_data.sinewave_analysis:
                sine = alpha_trend_data.sinewave_analysis
                print(f"Значення SineWave: {sine.get('sinewave', 'N/A'):.4f}")
                print(f"Сигнал: {sine.get('signal', 'N/A')}")
                print(f"Тренд: {sine.get('trend', 'N/A')}")
                print(f"Сила: {sine.get('strength', 'N/A')}")

            print("\n=== Market Trend Analyzer ===")
            print(f"Ринковий тренд: {alpha_trend_data_value}")

            print("\n=== Загальна інформація ===")
            print(f"Ціна закриття: {alpha_trend_data.indicators['close']:.4f}")
            print(
                f"Відношення ціни до AlphaTrend: {(alpha_trend_data.indicators['close'] / alpha_trend_data.alpha_trend - 1) * 100:.2f}%")

        # Отримання історичних даних для контексту
            print("\nОстанні 5 значень індикаторів:")
            historical_columns = ['open_time', 'close', 'alpha_trend', 'alpha_trend_signal', 'rsi', 'rsi_signal',
                              'super_trend_value', 'super_trend_signal', 'mfi', 'mfi_signal', 'sinewave',
                              'sinewave_signal']
            available_columns = [col for col in historical_columns if col in indicators_history.columns]
            print(indicators_history[available_columns].tail())

        # Додатковий вивід даних свічки
            #print(f"\n📊 Дані останньої свічки:")
            #pprint(alpha_trend_data.candle)

        # Вивід CVD аналізу
            if alpha_trend_data.cvd_analysis:
                print(f"\n📈 CVD Аналіз:")
                pprint(alpha_trend_data.cvd_analysis)

        # Вивід SineWave аналізу
            if alpha_trend_data.sinewave_analysis:
                print(f"\n📊 SineWave Аналіз:")
                pprint(alpha_trend_data.sinewave_analysis)

        # Вивід результатів MarketTrendAnalyzer
            print(f"\n📈 Market Trend Analyzer Results:")
            print(f"Ринковий тренд: {alpha_trend_data_value}")

        return alpha_trend_data, indicators_history

    except Exception as e:
        print(f"Помилка при отриманні даних: {e}")
        raise


if __name__ == "__main__":
    try:
        # Приклад використанся
        symbol = 'SOLUSDT'

        # Отримання поточних даних
        alpha_trend_data, indicators_history = get_of_data(symbol)

        # Комбінований аналіз сигналів
        print(f"\n🎯 КОМБІНОВАНИЙ АНАЛІЗ СИГНАЛІВ:")
        print(f"AlphaTrend: {alpha_trend_data.alpha_trend_signal}")
        print(f"RSI: {alpha_trend_data.rsi_signal}")
        print(f"SuperTrend: {alpha_trend_data.super_trend_signal}")
        print(f"MFI: {alpha_trend_data.mfi_signal}")
        if alpha_trend_data.cvd_analysis:
            print(f"CVD: {alpha_trend_data.cvd_analysis.get('trend', 'N/A')}")
        if alpha_trend_data.sinewave_analysis:
            print(f"SineWave: {alpha_trend_data.sinewave_analysis.get('signal', 'N/A')}")
        # Додаємо вивід результату MarketTrendAnalyzer
        print(f"Market Trend: {alpha_trend_data.indicators.get('market_trend', 'N/A')}")

    except Exception as e:
        print(f"Помилка: {e}")