import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from utils import DB_NAME, DB_HOST, DB_PASS, DB_PORT, DB_USER, TradeSignal
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

    def __init__(self, volume_threshold: float = 1.5):
        self.volume_threshold = volume_threshold

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

        return {
            'current': candle['volume'],
            'ratio': volume_ratio,
            'spike': volume_spike,
            'trend': volume_trend
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

        return {
            'value': candle['cvd'],
            'trend': cvd_trend,
            'strength': cvd_strength,
            'divergence': divergence
        }

    def _determine_cvd_trend(self, candle: pd.Series, df: pd.DataFrame) -> str:
        """
        Визначає тренд CVD (бичачий або ведмежий).

        Args:
            candle: Поточна свічка.
            df: Історичні дані.

        Returns:
            'bullish' або 'bearish'.
        """
        return "bullish" if candle['cvd'] > df['cvd'].iloc[-2] else "bearish"

    def _calculate_cvd_strength(self, candle: pd.Series, df: pd.DataFrame) -> str:
        """
        Розраховує силу сигналу CVD.

        Args:
            candle: Поточна свічка.
            df: Історичні дані.

        Returns:
            Рядок: 'strong', 'medium' або 'weak'.
        """
        cvd_change = abs(candle['cvd'] - df['cvd'].iloc[-2])
        avg_cvd_change = df['cvd'].abs().mean()
        strength_ratio = cvd_change / avg_cvd_change if avg_cvd_change > 0 else 0

        if strength_ratio > 0.8:
            return 'strong'
        elif strength_ratio > 0.5:
            return 'medium'
        else:
            return 'weak'

    def _check_divergence(self, df: pd.DataFrame) -> Optional[str]:
        """
        Перевіряє наявність дивергенції між ціною та CVD.

        Args:
            df: Історичні дані.

        Returns:
            'bullish_divergence', 'bearish_divergence' або None.
        """
        if len(df) < 10:
            return None

        # Bullish divergence
        if (df['close'].iloc[-1] < df['close'].iloc[-3] and
                df['cvd'].iloc[-1] > df['cvd'].iloc[-3]):
            return "bullish_divergence"

        # Bearish divergence
        if (df['close'].iloc[-1] > df['close'].iloc[-3] and
                df['cvd'].iloc[-1] < df['cvd'].iloc[-3]):
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


def get_of_data() -> MarketAnalysis:
    """
    Основна функція для отримання даних і запуску аналізу.

    Returns:
        Об'єкт MarketAnalysis з результатами аналізу.
    """
    # Ініціалізація
    bot = CryptoTraderBot()
    data_fetcher = DataFetcher(DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME)

    # Отримання даних
    table_name = "_candles_trading_data.solusdt_p_candles_test_data"
    data = data_fetcher.fetch_candle_data(table_name, limit=400)

    # Генерація сигналу
    analysis = bot.generate_signal(data)

    return analysis


if __name__ == "__main__":
    try:
        result = get_of_data()
        print("Аналіз ринку успішно завершено")
        print(f"Тренд ринку: {result.market_trend}")
        print(f"Поточний VPOC: {result.current_vpoc['value']:.4f}")

    except ValueError as e:
        print(f"Помилка: {e}")
    except Exception as e:
        print(f"Неочікувана помилка: {e}")
