import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import enum
from dataclasses import dataclass
from utils import DB_NAME, DB_HOST, DB_PASS, DB_PORT, DB_USER, TradeSignal
from sqlalchemy import create_engine, text


class OrderType(enum.Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


@dataclass
class TradeDecision:
    time: str
    signal: TradeSignal
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float
    reasoning: str


class CryptoTraderBot:
    def __init__(self, initial_balance: float = 1000.0, risk_per_trade: float = 0.02):
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.position = 0.0
        self.entry_price = 0.0
        self.trade_history = []

        # Параметри стратегії
        self.volume_threshold = 1.5
        self.cvd_threshold = 0.7
        self.vpoc_sensitivity = 0.005  # 0.5% від ціни для VPOC аналізу
        self.rsi_period = 14
        self.atr_period = 14
        self.vpoc_lookback = 20  # Аналіз VPOC за останні 20 свічок

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Розрахунок технічних індикаторів
        """
        df = df.copy()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = true_range.rolling(window=self.atr_period).mean()

        # Volume SMA
        df['volume_sma'] = df['volume'].rolling(window=20).mean()

        # CVD momentum
        df['cvd_change'] = df['cvd'].diff()
        df['cvd_trend'] = df['cvd_change'].rolling(window=5).mean()

        # VPOC аналіз
        df['vpoc_distance'] = (df['close'] - df['poc']) / df['close'] * 100
        df['vpoc_relation'] = np.where(df['close'] > df['poc'], 'above', 'below')

        return df.dropna()

    def analyze_vpoc_cluster(self, df: pd.DataFrame) -> Dict:
        """
        Аналіз кластера VPOC за останній період
        """
        recent_vpoc = df['poc'].tail(self.vpoc_lookback)
        current_price = df['close'].iloc[-1]

        # Знаходимо найближчі VPOC рівні
        vpoc_above = recent_vpoc[recent_vpoc > current_price]
        vpoc_below = recent_vpoc[recent_vpoc < current_price]

        # Найближчі рівні support/resistance
        nearest_resistance = vpoc_above.min() if len(vpoc_above) > 0 else None
        nearest_support = vpoc_below.max() if len(vpoc_below) > 0 else None

        # Сила кластера (кількість VPOC на рівні)
        if nearest_resistance:
            resistance_cluster = len(vpoc_above[vpoc_above <= nearest_resistance * 1.01])
        else:
            resistance_cluster = 0

        if nearest_support:
            support_cluster = len(vpoc_below[vpoc_below >= nearest_support * 0.99])
        else:
            support_cluster = 0

        return {
            'nearest_resistance': nearest_resistance,
            'nearest_support': nearest_support,
            'resistance_cluster_strength': resistance_cluster,
            'support_cluster_strength': support_cluster,
            'current_vpoc': df['poc'].iloc[-1],
            'current_relation': 'above' if current_price > df['poc'].iloc[-1] else 'below'
        }

    def analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        """
        Аналіз ринкової структури
        """
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        # Аналіз VPOC кластера
        vpoc_cluster = self.analyze_vpoc_cluster(df)

        # Аналіз поточної свічки VPOC
        current_vpoc_analysis = self._analyze_current_vpoc(latest)

        # Аналіз об'єму
        volume_analysis = self._analyze_volume(latest, df)

        # Аналіз CVD
        cvd_analysis = self._analyze_cvd(latest, df)

        # Аналіз цінової дії
        price_analysis = self._analyze_price_action(latest, prev)

        return {
            'vpoc_cluster': vpoc_cluster,
            'current_vpoc': current_vpoc_analysis,
            'volume': volume_analysis,
            'cvd': cvd_analysis,
            'price': price_analysis,
            'market_trend': self._determine_market_trend(df)
        }

    def _analyze_current_vpoc(self, candle: pd.Series) -> Dict:
        """
        Аналіз поточного VPOC
        """
        distance_pct = abs(candle['close'] - candle['poc']) / candle['close'] * 100
        relation = "above" if candle['close'] > candle['poc'] else "below"

        # Сила VPOC на основі відстані
        if distance_pct < 0.5:
            strength = 'very_strong'
        elif distance_pct < 1.0:
            strength = 'strong'
        elif distance_pct < 2.0:
            strength = 'medium'
        else:
            strength = 'weak'

        return {
            'value': candle['poc'],
            'relation': relation,
            'distance_pct': distance_pct,
            'strength': strength
        }

    def _analyze_volume(self, candle: pd.Series, df: pd.DataFrame) -> Dict:
        """
        Аналіз об'єму торгів
        """
        volume_ratio = candle['volume'] / df['volume'].rolling(20).mean().iloc[-1]
        volume_spike = volume_ratio > self.volume_threshold

        return {
            'current': candle['volume'],
            'ratio': volume_ratio,
            'spike': volume_spike,
            'trend': 'increasing' if candle['volume'] > df['volume'].iloc[-2] else 'decreasing'
        }

    def _analyze_cvd(self, candle: pd.Series, df: pd.DataFrame) -> Dict:
        """
        Аналіз Cumulative Volume Delta
        """
        cvd_trend = "bullish" if candle['cvd'] > df['cvd'].iloc[-2] else "bearish"
        cvd_strength = abs(candle['cvd'] - df['cvd'].iloc[-2]) / df['cvd'].abs().mean()

        return {
            'value': candle['cvd'],
            'trend': cvd_trend,
            'strength': 'strong' if cvd_strength > 0.8 else 'medium' if cvd_strength > 0.5 else 'weak',
            'divergence': self._check_cvd_divergence(df)
        }

    def _analyze_price_action(self, current: pd.Series, previous: pd.Series) -> Dict:
        """
        Аналіз цінової дії
        """
        body_size = abs(current['close'] - current['open'])
        total_range = current['high'] - current['low']
        body_ratio = body_size / total_range if total_range > 0 else 0

        candle_type = "bullish" if current['close'] > current['open'] else "bearish"
        if body_ratio < 0.3:
            candle_type = "doji"
        elif body_ratio > 0.7:
            candle_type += "_strong"

        return {
            'type': candle_type,
            'body_ratio': body_ratio,
            'wick_ratio': (total_range - body_size) / total_range if total_range > 0 else 0
        }

    def _determine_market_trend(self, df: pd.DataFrame) -> str:
        """
        Визначення загального тренду
        """
        if len(df) < 20:
            return "neutral"

        ma20 = df['close'].rolling(20).mean()
        ma50 = df['close'].rolling(50).mean()

        if ma20.iloc[-1] > ma50.iloc[-1] and df['close'].iloc[-1] > ma20.iloc[-1]:
            return "bullish"
        elif ma20.iloc[-1] < ma50.iloc[-1] and df['close'].iloc[-1] < ma20.iloc[-1]:
            return "bearish"
        else:
            return "neutral"

    def _check_cvd_divergence(self, df: pd.DataFrame) -> Optional[str]:
        """
        Перевірка дивергенції CVD
        """
        if len(df) < 10:
            return None

        # Bullish divergence (price lower low, CVD higher low)
        if (df['close'].iloc[-1] < df['close'].iloc[-3] and
                df['cvd'].iloc[-1] > df['cvd'].iloc[-3]):
            return "bullish_divergence"

        # Bearish divergence (price higher high, CVD lower high)
        if (df['close'].iloc[-1] > df['close'].iloc[-3] and
                df['cvd'].iloc[-1] < df['cvd'].iloc[-3]):
            return "bearish_divergence"

        return None

    def generate_signal(self, df: pd.DataFrame) -> TradeDecision:
        """
        Генерація торгового сигналу з VPOC для кожної свічки
        """
        if len(df) < 50:
            return TradeDecision(
                signal=TradeSignal.HOLD,
                confidence=0.0,
                entry_price=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                size=0.0,
                reasoning="Not enough data for analysis"
            )

        df_with_indicators = self.calculate_technical_indicators(df)
        market_analysis = self.analyze_market_structure(df_with_indicators)
        latest = df_with_indicators.iloc[-1]

        signal_strength = 0.0
        reasoning = []

        # 1. Аналіз поточного VPOC
        current_vpoc = market_analysis['current_vpoc']
        return market_analysis
        if current_vpoc['strength'] in ['very_strong', 'strong']:
            if current_vpoc['relation'] == 'above':
                signal_strength += 0.2
                reasoning.append(f"Price strongly above VPOC ({current_vpoc['distance_pct']:.2f}%)")
            else:
                signal_strength -= 0.2
                reasoning.append(f"Price strongly below VPOC ({current_vpoc['distance_pct']:.2f}%)")

        # 2. Аналіз VPOC кластера
        vpoc_cluster = market_analysis['vpoc_cluster']
        if vpoc_cluster['nearest_support'] and vpoc_cluster['support_cluster_strength'] >= 3:
            signal_strength += 0.15
            reasoning.append(f"Strong VPOC support cluster ({vpoc_cluster['support_cluster_strength']} candles)")

        if vpoc_cluster['nearest_resistance'] and vpoc_cluster['resistance_cluster_strength'] >= 3:
            signal_strength -= 0.15
            reasoning.append(f"Strong VPOC resistance cluster ({vpoc_cluster['resistance_cluster_strength']} candles)")

        # 3. Аналіз CVD
        if market_analysis['cvd']['trend'] == 'bullish':
            signal_strength += 0.25
            reasoning.append("Bullish CVD trend")
        elif market_analysis['cvd']['trend'] == 'bearish':
            signal_strength -= 0.25
            reasoning.append("Bearish CVD trend")

        if market_analysis['cvd']['divergence'] == 'bullish_divergence':
            signal_strength += 0.3
            reasoning.append("Bullish CVD divergence")
        elif market_analysis['cvd']['divergence'] == 'bearish_divergence':
            signal_strength -= 0.3
            reasoning.append("Bearish CVD divergence")

        # 4. Аналіз об'єму
        if market_analysis['volume']['spike']:
            if market_analysis['cvd']['trend'] == 'bullish':
                signal_strength += 0.15
                reasoning.append("Volume spike with bullish CVD")
            else:
                signal_strength -= 0.15
                reasoning.append("Volume spike with bearish CVD")

        # 5. Аналіз RSI
        if latest['rsi'] < 30:
            signal_strength += 0.1
            reasoning.append("Oversold (RSI < 30)")
        elif latest['rsi'] > 70:
            signal_strength -= 0.1
            reasoning.append("Overbought (RSI > 70)")

        # Визначення сигналу
        if signal_strength > 0.6:
            signal = TradeSignal.BUY
        elif signal_strength < -0.6:
            signal = TradeSignal.SELL
        else:
            signal = TradeSignal.HOLD

        # Розрахунок параметрів угоди
        if signal != TradeSignal.HOLD:
            entry_price = latest['close']
            atr = latest['atr']

            if signal == TradeSignal.BUY:
                stop_loss = entry_price - 2 * atr
                take_profit = entry_price + 3 * atr
            else:
                stop_loss = entry_price + 2 * atr
                take_profit = entry_price - 3 * atr

            risk_amount = self.balance * self.risk_per_trade
            position_size = risk_amount / abs(entry_price - stop_loss)

            reasoning_str = "; ".join(reasoning)

            return TradeDecision(
                time=df.iloc[-1]['close_time'] ,
                signal=signal,
                confidence=min(abs(signal_strength), 1.0),
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                size=position_size,
                reasoning=reasoning_str
            )
        else:
            return TradeDecision(
                time=df.iloc[0]['close_time'],
                signal=TradeSignal.HOLD,
                confidence=0.0,
                entry_price=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                size=0.0,
                reasoning="No strong signal detected"
            )

    def execute_trade(self, decision: TradeDecision) -> None:
        """
        Виконання торгової операції
        """
        if decision.signal == TradeSignal.HOLD:
            print("HOLD: No trade executed")
            return

        if decision.signal == TradeSignal.BUY:
            if self.position > 0:
                print("Already in long position")
                return

            # Закрити коротку позицію якщо є
            if self.position < 0:
                self._close_position()

            # Відкрити довгу позицію
            self.position = decision.size
            self.entry_price = decision.entry_price
            self.trade_history.append({
                'type': 'BUY',
                'price': decision.entry_price,
                'size': decision.size,
                'stop_loss': decision.stop_loss,
                'take_profit': decision.take_profit,
                'reason': decision.reasoning
            })

        elif decision.signal == TradeSignal.SELL:
            if self.position < 0:
                print("Already in short position")
                return

            # Закрити довгу позицію якщо є
            if self.position > 0:
                self._close_position()

            # Відкрити коротку позицію
            self.position = -decision.size
            self.entry_price = decision.entry_price
            self.trade_history.append({
                'type': 'SELL',
                'price': decision.entry_price,
                'size': decision.size,
                'stop_loss': decision.stop_loss,
                'take_profit': decision.take_profit,
                'reason': decision.reasoning
            })

    def _close_position(self) -> None:
        """Закриття поточної позиції"""
        # Тут буде логіка закриття позиції
        self.position = 0.0
        self.entry_price = 0.0

    def monitor_positions(self, current_price: float) -> None:
        """
        Моніторинг відкритих позицій
        """
        if self.position == 0:
            return

        # Перевірка стоп-лосу та тейк-профіту
        if self.position > 0:  # Long position
            if current_price <= self.trade_history[-1]['stop_loss']:
                print("Stop loss triggered for long position")
                self._close_position()
            elif current_price >= self.trade_history[-1]['take_profit']:
                print("Take profit triggered for long position")
                self._close_position()

        else:  # Short position
            if current_price >= self.trade_history[-1]['stop_loss']:
                print("Stop loss triggered for short position")
                self._close_position()
            elif current_price <= self.trade_history[-1]['take_profit']:
                print("Take profit triggered for short position")
                self._close_position()


# db_url: str, table: str, symbol: str = None, limit: int = 400
def fetch_data(
        table: str = "_candles_trading_data.solusdt_p_candles_test_data",
        limit: int = 400
    ):
    db_url = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    eng = create_engine(db_url)
    table = "_candles_trading_data.solusdt_p_candles_test_data"
    query = f"""
                SELECT open_time, close_time, symbol, open, close, high, low, cvd, volume, poc 
                FROM {table} 
                ORDER BY open_time DESC 
                LIMIT {limit}
            """

    with eng.begin() as conn:
        df = pd.read_sql(text(query), conn)

    return df



# Приклад використання
def get_of_signal():
    # Створення бота
    bot = CryptoTraderBot(initial_balance=1000.0, risk_per_trade=0.02)
    data = fetch_data()
    df_sorted_desc = data.sort_values(by='open_time')
    #decision = bot.generate_signal(df_sorted_desc)
    #return decision

    decision = bot.generate_signal(df_sorted_desc)
    """
    if decision.signal != TradeSignal.HOLD:
        print("\n")
        print("=" * 60)
        print("Time", decision.time)
        print("Position", decision.signal)
        print("Confidence", decision.confidence)
        print("Take profit", decision.take_profit)
        print('Entry price', decision.entry_price)
        print('Stop loss', decision.stop_loss)
        print('Reasoning', decision.reasoning)
        print("=" * 60)
        print("\n")
    """
    return decision

    # Виконання торгівлі
    # bot.execute_trade(decision)

    # Моніторинг позицій
    # bot.monitor_positions(current_price=df['close'].iloc[-1])


if __name__ == "__main__":
    get_of_signal()