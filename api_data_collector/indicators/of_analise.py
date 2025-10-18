import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from utils import DB_NAME, DB_HOST, DB_PASS, DB_PORT, DB_USER
from sqlalchemy import create_engine, text


@dataclass
class MarketAnalysis:
    """Результат аналізу ринку"""
    vpoc_cluster: Dict
    current_vpoc: Dict
    volume: Dict
    cvd: Dict
    price: Dict
    market_trend: str
    enhanced_market_trend: str
    indicators: pd.Series


@dataclass
class TechnicalIndicators:
    """Технічні індикатори"""
    rsi: float
    atr: float
    volume_sma: float
    cvd_trend: float
    vpoc_distance: float
    vpoc_relation: str


class TechnicalAnalyzer:
    """Клас для розрахунку технічних індикаторів"""

    def __init__(self, rsi_period: int = 14, atr_period: int = 14, volume_window: int = 20):
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.volume_window = volume_window

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Розраховує всі технічні індикатори та додає їх у DataFrame.

        Args:
            df: Вхідні дані свічок.

        Returns:
            DataFrame з доданими індикаторами.
        """
        df = df.copy()

        df = self._calculate_rsi(df)
        df = self._calculate_atr(df)
        df = self._calculate_volume_indicators(df)
        df = self._calculate_cvd_momentum(df)
        df = self._calculate_vpoc_analysis(df)

        return df.dropna()

    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Розраховує індикатор RSI (Relative Strength Index).

        Args:
            df: Вхідні дані свічок.

        Returns:
            DataFrame з доданим стовпцем rsi.
        """
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        return df

    def _calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Розраховує ATR (Average True Range).

        Args:
            df: Вхідні дані свічок.

        Returns:
            DataFrame з доданим стовпцем atr.
        """
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = true_range.rolling(window=self.atr_period).mean()
        return df

    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Розраховує середній об’єм торгів (SMA).

        Args:
            df: Вхідні дані свічок.

        Returns:
            DataFrame з доданим стовпцем volume_sma.
        """
        df['volume_sma'] = df['volume'].rolling(window=self.volume_window).mean()
        return df

    def _calculate_cvd_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Розраховує зміни та тренд CVD (Cumulative Volume Delta).

        Args:
            df: Вхідні дані свічок.

        Returns:
            DataFrame з доданими стовпцями cvd_change та cvd_trend.
        """
        df['cvd_change'] = df['cvd'].diff()
        df['cvd_trend'] = df['cvd_change'].rolling(window=5).mean()
        return df

    def _calculate_vpoc_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Виконує аналіз VPOC (Volume Point of Control).

        Args:
            df: Вхідні дані свічок.

        Returns:
            DataFrame з доданими стовпцями vpoc_distance та vpoc_relation.
        """
        df['vpoc_distance'] = (df['close'] - df['poc']) / df['close'] * 100
        df['vpoc_relation'] = np.where(df['close'] > df['poc'], 'above', 'below')
        return df


class VPOCCalculator:
    """Клас для аналізу VPOC"""

    def __init__(self, lookback_period: int = 20, sensitivity: float = 0.005):
        self.lookback_period = lookback_period
        self.sensitivity = sensitivity

    def analyze_cluster(self, df: pd.DataFrame) -> Dict:
        """
        Виконує аналіз кластера VPOC за заданий період.

        Args:
            df: Дані свічок з VPOC.

        Returns:
            Словник з найближчим опором, підтримкою та силою кластерів.
        """
        recent_vpoc = df['poc'].tail(self.lookback_period)
        current_price = df['close'].iloc[-1]

        resistance_info = self._find_resistance_levels(recent_vpoc, current_price)
        support_info = self._find_support_levels(recent_vpoc, current_price)

        return {
            'nearest_resistance': resistance_info['level'],
            'nearest_support': support_info['level'],
            'resistance_cluster_strength': resistance_info['cluster_strength'],
            'support_cluster_strength': support_info['cluster_strength'],
            'current_vpoc': df['poc'].iloc[-1],
            'current_relation': 'above' if current_price > df['poc'].iloc[-1] else 'below'
        }

    def _find_resistance_levels(self, vpoc_series: pd.Series, current_price: float) -> Dict:
        """
        Знаходить рівні опору на основі VPOC.

        Args:
            vpoc_series: Історичні значення VPOC.
            current_price: Поточна ціна активу.

        Returns:
            Словник з найближчим рівнем опору та силою кластера.
        """
        vpoc_above = vpoc_series[vpoc_series > current_price]

        if vpoc_above.empty:
            return {'level': None, 'cluster_strength': 0}

        nearest_resistance = vpoc_above.min()
        resistance_cluster = len(vpoc_above[vpoc_above <= nearest_resistance * 1.01])

        return {'level': nearest_resistance, 'cluster_strength': resistance_cluster}

    def _find_support_levels(self, vpoc_series: pd.Series, current_price: float) -> Dict:
        """
        Знаходить рівні підтримки на основі VPOC.

        Args:
            vpoc_series: Історичні значення VPOC.
            current_price: Поточна ціна активу.

        Returns:
            Словник з найближчим рівнем підтримки та силою кластера.
        """
        vpoc_below = vpoc_series[vpoc_series < current_price]

        if vpoc_below.empty:
            return {'level': None, 'cluster_strength': 0}

        nearest_support = vpoc_below.max()
        support_cluster = len(vpoc_below[vpoc_below >= nearest_support * 0.99])

        return {'level': nearest_support, 'cluster_strength': support_cluster}

    def analyze_current_vpoc(self, candle: pd.Series) -> Dict:
        """
        Аналізує поточний VPOC відносно ціни закриття.

        Args:
            candle: Дані однієї свічки.

        Returns:
            Словник з відстанню, напрямком та силою VPOC.
        """
        distance_pct = abs(candle['close'] - candle['poc']) / candle['close'] * 100
        relation = "above" if candle['close'] > candle['poc'] else "below"

        strength = self._classify_vpoc_strength(distance_pct)

        return {
            'value': candle['poc'],
            'relation': relation,
            'distance_pct': distance_pct,
            'strength': strength
        }

    def _classify_vpoc_strength(self, distance_pct: float) -> str:
        """
        Класифікує силу VPOC залежно від відстані до ціни.

        Args:
            distance_pct: Відстань у відсотках.

        Returns:
            Рядок зі значенням сили VPOC.
        """
        if distance_pct < 0.5:
            return 'very_strong'
        elif distance_pct < 1.0:
            return 'strong'
        elif distance_pct < 2.0:
            return 'medium'
        else:
            return 'weak'


class VolumeAnalyzer:
    """Аналіз об'єму торгів"""

    def __init__(self, volume_threshold: float = 1.5, momentum_window: int = 24):
        self.volume_threshold = volume_threshold
        self.momentum_window = momentum_window

    def analyze(self, candle: pd.Series, df: pd.DataFrame) -> Dict:
        """
        Аналізує об’єм поточної свічки та визначає аномалії.

        Args:
            candle: Поточна свічка.
            df: Історичні дані свічок.

        Returns:
            Словник з об’ємом, відношенням, наявністю спайку та трендом.
        """
        volume_ratio = self._calculate_volume_ratio(candle, df)
        volume_spike = volume_ratio > self.volume_threshold
        volume_trend = self._determine_volume_trend(candle, df)
        volume_momentum, volume_momentum_ratio = self._calculate_volume_momentum(df)

        return {
            'current': candle['volume'],
            'ratio': volume_ratio,
            'spike': volume_spike,
            'trend': volume_trend,
            'volume_momentum': volume_momentum,
            'volume_momentum_ratio': volume_momentum_ratio
        }

    def _calculate_volume_ratio(self, candle: pd.Series, df: pd.DataFrame) -> float:
        """
        Обчислює відношення об’єму поточної свічки до середнього.

        Args:
            candle: Поточна свічка.
            df: Історичні дані.

        Returns:
            Співвідношення об’єму.
        """
        volume_ma = df['volume'].rolling(20).mean().iloc[-1]
        return candle['volume'] / volume_ma if volume_ma > 0 else 1.0

    def _determine_volume_trend(self, candle: pd.Series, df: pd.DataFrame) -> str:
        """
        Визначає тренд об’єму (зростає чи зменшується).

        Args:
            candle: Поточна свічка.
            df: Історичні дані.

        Returns:
            Рядок: 'increasing' або 'decreasing'
        """
        return 'increasing' if candle['volume'] > df['volume'].iloc[-2] else 'decreasing'

    def _calculate_volume_momentum(self, df: pd.DataFrame) -> (float, float):
        """
        Обчислює моментум об’єму — зміну обсягу відносно попереднього періоду.

        Returns:
            (momentum, momentum_ratio)
        """
        if len(df) < self.momentum_window + 1:
            return 0.0, 1.0

        current_volume = df['volume'].iloc[-1]
        past_volume = df['volume'].iloc[-self.momentum_window - 1]

        volume_momentum = current_volume - past_volume
        volume_momentum_ratio = (current_volume / past_volume) if past_volume > 0 else 1.0

        return volume_momentum, volume_momentum_ratio




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
        divergence = self._check_divergence(df)
        confidence = self._calculate_cvd_confidence(df)

        # Комбінована оцінка
        signal_quality = self._assess_signal_quality(
            cvd_trend, cvd_strength, divergence, confidence
        )

        return {
            'value': candle['cvd'],
            'trend': cvd_trend,
            'strength': cvd_strength,
            'divergence': divergence,
            'confidence': round(confidence, 2),
            'signal_quality': signal_quality,
            'timestamp': candle.name if hasattr(candle, 'name') else None
        }

    def _assess_signal_quality(self, trend: str, strength: str,
                               divergence: Optional[str], confidence: float) -> str:
        """Оцінює загальну якість сигналу."""
        quality_score = 0

        # Бали за тренд
        if trend != "neutral":
            quality_score += 1

        # Бали за силу
        strength_scores = {'weak': 0, 'medium': 1, 'strong': 2}
        quality_score += strength_scores.get(strength, 0)

        # Бали за дивергенцію
        if divergence:
            quality_score += 2

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

    def _find_extremes(self, series: pd.Series, period: int = 5) -> Tuple[List[int], List[int]]:
        """Знаходить максимуми та мінімуми в ряді."""
        highs = []
        lows = []

        for i in range(period, len(series) - period):
            window = series.iloc[i - period:i + period + 1]
            if series.iloc[i] == window.max():
                highs.append(i)
            elif series.iloc[i] == window.min():
                lows.append(i)

        return highs, lows

    def _detect_bearish_divergence(self, df: pd.DataFrame, price_highs: List[int], cvd_highs: List[int]) -> bool:
        """Виявляє ведмежу дивергенцію."""
        if len(price_highs) < 2 or len(cvd_highs) < 2:
            return False

        # Останні два максимуми
        recent_price_high_idx = price_highs[-1]
        prev_price_high_idx = price_highs[-2]
        recent_cvd_high_idx = cvd_highs[-1]
        prev_cvd_high_idx = cvd_highs[-2]

        # Отримуємо фактичні значення за індексами
        recent_price_high = df['close'].iloc[recent_price_high_idx]
        prev_price_high = df['close'].iloc[prev_price_high_idx]
        recent_cvd_high = df['cvd'].iloc[recent_cvd_high_idx]
        prev_cvd_high = df['cvd'].iloc[prev_cvd_high_idx]

        # Ведмежа дивергенція: ціна робить вищий максимум, CVD - нижчий
        bearish_divergence = (recent_price_high > prev_price_high and
                              recent_cvd_high < prev_cvd_high)

        # Додаткові умови для підтвердження
        if bearish_divergence:
            # Перевіряємо, що дивергенція не застаріла
            is_recent = (len(df) - recent_price_high_idx) <= 5

            # Перевіряємо міцність сигналу
            price_change_pct = (recent_price_high - prev_price_high) / prev_price_high
            cvd_change_pct = (prev_cvd_high - recent_cvd_high) / abs(prev_cvd_high) if prev_cvd_high != 0 else 0

            strong_signal = (price_change_pct > 0.01 and cvd_change_pct > 0.1)

            return is_recent and strong_signal

        return False

    def _detect_bullish_divergence(self, df: pd.DataFrame, price_lows: List[int], cvd_lows: List[int]) -> bool:
        """Виявляє бичу дивергенцію."""
        if len(price_lows) < 2 or len(cvd_lows) < 2:
            return False

        # Останні два мінімуми
        recent_price_low_idx = price_lows[-1]
        prev_price_low_idx = price_lows[-2]
        recent_cvd_low_idx = cvd_lows[-1]
        prev_cvd_low_idx = cvd_lows[-2]

        # Отримуємо фактичні значення за індексами
        recent_price_low = df['close'].iloc[recent_price_low_idx]
        prev_price_low = df['close'].iloc[prev_price_low_idx]
        recent_cvd_low = df['cvd'].iloc[recent_cvd_low_idx]
        prev_cvd_low = df['cvd'].iloc[prev_cvd_low_idx]

        # Бича дивергенція: ціна робить нижчий мінімум, CVD - вищий
        bullish_divergence = (recent_price_low < prev_price_low and
                              recent_cvd_low > prev_cvd_low)

        # Додаткові умови для підтвердження
        if bullish_divergence:
            # Перевіряємо, що дивергенція не застаріла
            is_recent = (len(df) - recent_price_low_idx) <= 5

            # Перевіряємо міцність сигналу
            price_change_pct = (prev_price_low - recent_price_low) / prev_price_low
            cvd_change_pct = (recent_cvd_low - prev_cvd_low) / abs(prev_cvd_low) if prev_cvd_low != 0 else 0

            strong_signal = (price_change_pct > 0.01 and cvd_change_pct > 0.1)

            return is_recent and strong_signal

        return False

    def _check_divergence(self, df: pd.DataFrame) -> Optional[str]:
        """
        Перевіряє наявність дивергенції між ціною та CVD.
        """
        if len(df) < 15:
            return None

        # Знаходимо екстремуми ціни та CVD
        price_highs, price_lows = self._find_extremes(df['close'], period=5)
        cvd_highs, cvd_lows = self._find_extremes(df['cvd'], period=5)

        # Перевіряємо дивергенції
        bullish_div = self._detect_bullish_divergence(df, price_lows, cvd_lows)
        bearish_div = self._detect_bearish_divergence(df, price_highs, cvd_highs)

        if bullish_div:
            return "bullish_divergence"
        elif bearish_div:
            return "bearish_divergence"

        return None


class PriceActionAnalyzer:
    """Аналіз цінової дії"""

    @staticmethod
    def analyze(current: pd.Series, previous: pd.Series) -> Dict:
        """
        Аналізує поточну свічку відносно попередньої.

        Args:
            current: Поточна свічка.
            previous: Попередня свічка.

        Returns:
            Словник з типом, співвідношенням тіла та хвостів.
        """
        body_size, total_range = PriceActionAnalyzer._calculate_candle_metrics(current)
        body_ratio = PriceActionAnalyzer._calculate_body_ratio(body_size, total_range)
        candle_type = PriceActionAnalyzer._classify_candle_type(current, body_ratio)

        return {
            'type': candle_type,
            'body_ratio': body_ratio,
            'wick_ratio': PriceActionAnalyzer._calculate_wick_ratio(body_size, total_range)
        }

    @staticmethod
    def _calculate_candle_metrics(candle: pd.Series) -> Tuple[float, float]:
        """
        Обчислює розмір тіла та діапазон свічки.

        Args:
            candle: Дані однієї свічки.

        Returns:
            Кортеж (розмір тіла, загальний діапазон).
        """
        body_size = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        return body_size, total_range

    @staticmethod
    def _calculate_body_ratio(body_size: float, total_range: float) -> float:
        """
        Обчислює співвідношення тіла свічки до діапазону.

        Args:
            body_size: Розмір тіла.
            total_range: Загальний діапазон.

        Returns:
            Значення співвідношення.
        """
        return body_size / total_range if total_range > 0 else 0

    @staticmethod
    def _calculate_wick_ratio(body_size: float, total_range: float) -> float:
        """
        Обчислює співвідношення хвостів свічки.

        Args:
            body_size: Розмір тіла.
            total_range: Загальний діапазон.

        Returns:
            Значення співвідношення.
        """
        return (total_range - body_size) / total_range if total_range > 0 else 0

    @staticmethod
    def _classify_candle_type(candle: pd.Series, body_ratio: float) -> str:
        """
        Визначає тип свічки (бичача, ведмежа, сильна чи doji).

        Args:
            candle: Дані свічки.
            body_ratio: Співвідношення тіла.

        Returns:
            Рядок з типом свічки.
        """
        base_type = "bullish" if candle['close'] > candle['open'] else "bearish"

        if body_ratio < 0.3:
            return "doji"
        elif body_ratio > 0.7:
            return f"{base_type}_strong"
        else:
            return base_type


class EnhancedMarketTrendAnalyzer:
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

    def _analyze_bb_position(self, price: float, bb_upper: float,  # ⬅️ ВИДАЛИТЬ @staticmethod
                             bb_lower: float, bb_middle: float) -> str:
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

    def _analyze_ma_cross(self, df: pd.DataFrame) -> str:  # ⬅️ ВИДАЛИТЬ @staticmethod
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

    def _analyze_bb_volatility(self, df: pd.DataFrame) -> str:  # ⬅️ ВИДАЛИТЬ @staticmethod
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

    def _combine_signals(self, bb_signal: str, ma_signal: str, volatility_signal: str) -> str:  # ⬅️ ВИДАЛИТЬ @staticmethod
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


class MarketTrendAnalyzer:
    """Аналізатор ринкового тренду"""

    @staticmethod
    def determine_trend(df: pd.DataFrame) -> str:
        """
        Визначає загальний тренд ринку на основі ковзних середніх.

        Args:
            df: Історичні дані.

        Returns:
            'bullish', 'bearish' або 'neutral'.
        """
        if len(df) < 20:
            return "neutral"

        ma20 = df['close'].rolling(20).mean()
        ma50 = df['close'].rolling(50).mean()
        current_price = df['close'].iloc[-1]

        if ma20.iloc[-1] > ma50.iloc[-1] and current_price > ma20.iloc[-1]:
            return "bullish"
        elif ma20.iloc[-1] < ma50.iloc[-1] and current_price < ma20.iloc[-1]:
            return "bearish"
        else:
            return "neutral"


class CryptoTraderBot:
    """
    Основний клас торгового бота з VPOC аналізом
    """

    def __init__(self):
        # Ініціалізація аналізаторів
        self.technical_analyzer = TechnicalAnalyzer()
        self.vpoc_calculator = VPOCCalculator()
        self.volume_analyzer = VolumeAnalyzer()
        self.cvd_analyzer = CVDAnalyzer()
        self.price_action_analyzer = PriceActionAnalyzer()
        self.trend_analyzer = MarketTrendAnalyzer()
        self.enhanced_trend_analyzer = EnhancedMarketTrendAnalyzer()

    def generate_signal(self, df: pd.DataFrame) -> MarketAnalysis:
        """
        Генерує торговий сигнал на основі аналізу VPOC та індикаторів.

        Args:
            df: Історичні дані свічок.

        Returns:
            Об'єкт MarketAnalysis з результатами аналізу.
        """
        if len(df) < 50:
            raise ValueError("Недостатньо даних для аналізу. Потрібно мінімум 50 свічок.")

        df_with_indicators = self.technical_analyzer.calculate_all_indicators(df)
        market_analysis = self._analyze_market_structure(df_with_indicators)

        return market_analysis

    def _analyze_market_structure(self, df: pd.DataFrame) -> MarketAnalysis:
        """
        Виконує комплексний аналіз ринкової структури.

        Args:
            df: Історичні дані свічок з індикаторами.

        Returns:
            Об'єкт MarketAnalysis.
        """
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        analysis = MarketAnalysis(
            vpoc_cluster=self.vpoc_calculator.analyze_cluster(df),
            current_vpoc=self.vpoc_calculator.analyze_current_vpoc(latest),
            volume=self.volume_analyzer.analyze(latest, df),
            cvd=self.cvd_analyzer.analyze(latest, df),
            price=self.price_action_analyzer.analyze(latest, prev),
            market_trend=self.trend_analyzer.determine_trend(df),
            enhanced_market_trend=self.enhanced_trend_analyzer.determine_trend_with_bb(df),
            indicators=latest
        )

        return analysis


class DataFetcher:
    """Клас для отримання даних з бази"""

    def __init__(self, db_user: str, db_pass: str, db_host: str, db_port: str, db_name: str):
        self.db_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

    def fetch_candle_data(self, table: str, limit: int = 400) -> pd.DataFrame:
        """
        Отримує дані свічок з таблиці PostgreSQL.

        Args:
            table: Назва таблиці.
            limit: Кількість рядків для вибірки.

        Returns:
            DataFrame з даними свічок.
        """
        query = f"""
            SELECT open_time, close_time, symbol, open, close, high, low, cvd, volume, poc 
            FROM {table} 
            ORDER BY open_time DESC 
            LIMIT {limit}
        """

        with create_engine(self.db_url).begin() as conn:
            df = pd.read_sql(text(query), conn)

        return df.sort_values(by='open_time')


def get_of_data(symbol, is_test=False) -> MarketAnalysis:
    """
    Основна функція для отримання даних і запуску аналізу.

    Returns:
        Об'єкт MarketAnalysis з результатами аналізу.
    """
    # Ініціалізація
    bot = CryptoTraderBot()
    data_fetcher = DataFetcher(DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME)

    # Отримання даних
    if is_test:
        table_name = f"_candles_trading_data.{str(symbol).lower()}_p_candles_test_data"
    else:
        table_name = f"_candles_trading_data.{str(symbol).lower()}_p_candles"

    data = data_fetcher.fetch_candle_data(table_name, limit=400)

    # Генерація сигналу
    analysis = bot.generate_signal(data)

    return analysis


if __name__ == "__main__":
    try:
        result = get_of_data('SOLUSDT')
        print("Аналіз ринку успішно завершено")
        print(f"Тренд ринку: {result.market_trend}")
        print(f"Поточний VPOC: {result.current_vpoc['value']:.4f}")

    except ValueError as e:
        print(f"Помилка: {e}")
    except Exception as e:
        print(f"Неочікувана помилка: {e}")
