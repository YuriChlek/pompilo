import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import enum
from dataclasses import dataclass
from sqlalchemy import create_engine, text
from utils import DB_NAME, DB_HOST, DB_PASS, DB_PORT, DB_USER


class GridDirection(enum.Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"


@dataclass
class GridLevel:
    price: float
    order_type: str
    quantity: float
    filled: bool = False
    order_id: Optional[str] = None
    current_price: Optional[float] = None  # Додано поточну ціну


class SpotGridBot:
    def __init__(
            self,
            usdt_balance: float = 1000.0,
            asset_balance: float = 0.0,
            grid_levels: int = 10,
            grid_spacing: float = 0.02,
            risk_per_trade: float = 0.1
    ):
        self.usdt_balance = usdt_balance
        self.asset_balance = asset_balance
        self.grid_levels = grid_levels
        self.grid_spacing = grid_spacing
        self.risk_per_trade = risk_per_trade
        self.grid_lines: List[GridLevel] = []
        self.trade_history = []
        self.average_buy_price = 0.0
        self.total_invested = 0.0
        self.current_price = 0.0  # Додано поточну ціну

        # Параметри для аналізу
        self.vpoc_lookback = 50
        self.volume_threshold = 1.5
        self.rsi_period = 14
        self.atr_period = 14

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Розрахунок технічних індикаторів"""
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
        if 'cvd' in df.columns:
            df['cvd_change'] = df['cvd'].diff()
            df['cvd_trend'] = df['cvd_change'].rolling(window=5).mean()

        # VPOC аналіз
        df['vpoc_distance'] = (df['close'] - df['poc']) / df['close'] * 100
        df['vpoc_relation'] = np.where(df['close'] > df['poc'], 'above', 'below')

        return df.dropna()

    def analyze_market_conditions(self, df: pd.DataFrame) -> Dict:
        """Аналіз ринкових умов для купівлі"""
        df_with_indicators = self.calculate_technical_indicators(df)
        latest = df_with_indicators.iloc[-1]

        return {
            'rsi': self._analyze_rsi(latest),
            'atr': self._analyze_atr(latest, df_with_indicators),
            'volume': self._analyze_volume(latest, df_with_indicators),
            'cvd': self._analyze_cvd(latest, df_with_indicators) if 'cvd' in df.columns else {'value': 0,
                                                                                              'trend': 'neutral'},
            'vpoc': self._analyze_vpoc(latest),
            'trend': self._determine_market_trend(df_with_indicators),
            'volatility': self._analyze_volatility(df_with_indicators)
        }

    def _analyze_rsi(self, candle: pd.Series) -> Dict:
        """Аналіз RSI для купівлі"""
        rsi_value = candle.get('rsi', 50)

        if rsi_value < 35:
            return {'value': rsi_value, 'signal': 'strong_buy', 'strength': 0.3}
        elif rsi_value < 45:
            return {'value': rsi_value, 'signal': 'buy', 'strength': 0.2}
        elif rsi_value > 70:
            return {'value': rsi_value, 'signal': 'avoid', 'strength': -0.2}
        else:
            return {'value': rsi_value, 'signal': 'neutral', 'strength': 0.1}

    def _analyze_atr(self, candle: pd.Series, df: pd.DataFrame) -> Dict:
        """Аналіз ATR"""
        atr_value = candle.get('atr', 0)
        atr_percent = (atr_value / candle['close']) * 100 if candle['close'] > 0 else 0

        return {'value': atr_value, 'percent': atr_percent}

    def _analyze_volume(self, candle: pd.Series, df: pd.DataFrame) -> Dict:
        """Аналіз об'єму"""
        volume_ratio = candle['volume'] / df['volume_sma'].iloc[-1] if df['volume_sma'].iloc[-1] > 0 else 1

        if volume_ratio > 2.0:
            return {'ratio': volume_ratio, 'strength': 'very_strong', 'signal': 0.3}
        elif volume_ratio > 1.5:
            return {'ratio': volume_ratio, 'strength': 'strong', 'signal': 0.2}
        else:
            return {'ratio': volume_ratio, 'strength': 'weak', 'signal': 0.1}

    def _analyze_cvd(self, candle: pd.Series, df: pd.DataFrame) -> Dict:
        """Аналіз CVD для купівлі"""
        cvd_value = candle['cvd']
        cvd_trend = "bullish" if cvd_value > df['cvd'].iloc[-2] else "bearish"

        if cvd_trend == "bullish":
            return {'value': cvd_value, 'trend': cvd_trend, 'signal': 0.2}
        else:
            return {'value': cvd_value, 'trend': cvd_trend, 'signal': -0.1}

    def _analyze_vpoc(self, candle: pd.Series) -> Dict:
        """Аналіз VPOC для купівлі"""
        distance_pct = abs(candle['close'] - candle['poc']) / candle['close'] * 100
        relation = "above" if candle['close'] > candle['poc'] else "below"

        # Краще купувати коли ціна нижче VPOC
        if relation == "below":
            return {'relation': relation, 'distance_pct': distance_pct, 'signal': 0.2}
        else:
            return {'relation': relation, 'distance_pct': distance_pct, 'signal': -0.1}

    def _determine_market_trend(self, df: pd.DataFrame) -> Dict:
        """Визначення тренду"""
        if len(df) < 20:
            return {'direction': 'neutral', 'signal': 0.0}

        ma20 = df['close'].rolling(20).mean()
        ma50 = df['close'].rolling(50).mean()

        # Для купівлі краще ведмежий або нейтральний тренд
        if ma20.iloc[-1] < ma50.iloc[-1]:
            return {'direction': 'bearish', 'signal': 0.3}  # Краще купувати на падінні
        else:
            return {'direction': 'bullish', 'signal': 0.1}

    def _analyze_volatility(self, df: pd.DataFrame) -> Dict:
        """Аналіз волатильності"""
        volatility = df['close'].pct_change().std() * 100

        if volatility > 4.0:
            return {'level': 'high', 'signal': 0.3}  # Висока волатильність - краще для купівлі
        else:
            return {'level': 'low', 'signal': 0.1}

    def calculate_buy_signal_strength(self, df: pd.DataFrame) -> float:
        """Розрахунок загальної сили сигналу для купівлі"""
        analysis = self.analyze_market_conditions(df)

        # Сумуємо всі сигнали
        total_signal = (
                analysis['rsi']['strength'] +
                analysis['volume']['signal'] +
                analysis['cvd']['signal'] +
                analysis['vpoc']['signal'] +
                analysis['trend']['signal'] +
                analysis['volatility']['signal']
        )

        return max(0.0, min(1.0, total_signal))

    def setup_buy_grid(self, df: pd.DataFrame) -> None:
        """Налаштування сітки тільки для купівлі"""
        current_price = df['close'].iloc[-1]
        self.current_price = current_price  # Зберігаємо поточну ціну
        signal_strength = self.calculate_buy_signal_strength(df)

        # Визначаємо діапазон для купівлі на основі сигналу
        if signal_strength > 0.6:
            # Сильний сигнал - ширший діапазон купівлі
            lower_bound = current_price * (1 - self.grid_spacing * 8)
            upper_bound = current_price * (1 - self.grid_spacing * 1)  # Трохи нижче поточної ціни
        elif signal_strength > 0.3:
            # Середній сигнал
            lower_bound = current_price * (1 - self.grid_spacing * 6)
            upper_bound = current_price * (1 - self.grid_spacing * 2)
        else:
            # Слабкий сигнал - вузький діапазон
            lower_bound = current_price * (1 - self.grid_spacing * 4)
            upper_bound = current_price * (1 - self.grid_spacing * 3)

        print(f"Setting up BUY grid from {lower_bound:.4f} to {upper_bound:.4f}")
        print(f"Current price: {current_price:.4f}")
        print(f"Buy signal strength: {signal_strength:.2f}/1.0")

        # Очищаємо попередню сітку
        self.grid_lines = []

        # Створюємо рівні тільки для купівлі
        price_levels = np.linspace(lower_bound, upper_bound, self.grid_levels)

        # Розподіляємо баланс
        investment_per_level = self.usdt_balance / self.grid_levels * self.risk_per_trade

        for price in price_levels:
            quantity = investment_per_level / price

            self.grid_lines.append(GridLevel(
                price=price,
                order_type='buy',
                quantity=quantity,
                current_price=current_price  # Додаємо поточну ціну до кожного ордера
            ))

        print(f"Created {len(self.grid_lines)} BUY orders")

    def check_buy_execution(self, current_price: float) -> List[Dict]:
        """Перевіряє виконання buy ордерів"""
        executed_trades = []
        self.current_price = current_price  # Оновлюємо поточну ціну

        for level in self.grid_lines:
            # Оновлюємо поточну ціну для кожного ордера
            level.current_price = current_price

            if not level.filled and level.order_type == 'buy' and current_price <= level.price:
                cost = level.quantity * current_price
                if cost <= self.usdt_balance:
                    self.usdt_balance -= cost
                    self.asset_balance += level.quantity
                    self.total_invested += cost

                    # Оновлюємо середню ціну купівлі
                    if self.asset_balance > 0:
                        self.average_buy_price = self.total_invested / self.asset_balance

                    level.filled = True
                    executed_trades.append({
                        'type': 'BUY',
                        'price': current_price,
                        'quantity': level.quantity,
                        'cost': cost,
                        'avg_buy_price': self.average_buy_price
                    })

        return executed_trades

    def run_buy_strategy(self, df: pd.DataFrame) -> None:
        """Основна логіка роботи з купівлями"""
        current_price = df['close'].iloc[-1]
        self.current_price = current_price  # Оновлюємо поточну ціну

        if not self.grid_lines:
            print("Setting up initial BUY grid...")
            self.setup_buy_grid(df)
        else:
            # Перевіряємо чи потрібно оновити сітку
            active_prices = [level.price for level in self.grid_lines if not level.filled]
            if not active_prices or current_price < min(active_prices):
                print("Price moved below grid, resetting...")
                self.setup_buy_grid(df)

        # Перевіряємо виконання buy ордерів
        executed_trades = self.check_buy_execution(current_price)

        if executed_trades:
            print(f"\nExecuted {len(executed_trades)} BUY trades:")
            for trade in executed_trades:
                print(f"  BUY {trade['quantity']:.6f} at {trade['price']:.4f}")

        # Виводимо поточний стан
        total_value = self.usdt_balance + (self.asset_balance * current_price)
        print(f"\nCurrent Status:")
        print(f"USDT Balance: {self.usdt_balance:.2f}")
        print(f"Asset Balance: {self.asset_balance:.6f}")
        print(f"Total Value: {total_value:.2f} USDT")
        print(f"Current Price: {current_price:.4f}")  # Додано вивід поточної ціни

        if self.average_buy_price > 0:
            unrealized_pnl = ((current_price - self.average_buy_price) / self.average_buy_price) * 100
            print(f"Avg Buy Price: {self.average_buy_price:.4f}")
            print(f"Unrealized PnL: {unrealized_pnl:+.2f}%")

        print(f"Active BUY orders: {sum(1 for level in self.grid_lines if not level.filled)}")

    def get_grid_with_current_price(self) -> List[GridLevel]:
        """Повертає список ордерів з поточною ціною"""
        return self.grid_lines


# Функція для отримання даних
def fetch_data(
        table: str = "_candles_trading_data.solusdt_p_candles_test_data",
        limit: int = 200
) -> pd.DataFrame:
    db_url = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    eng = create_engine(db_url)

    query = f"""
        SELECT open_time, close_time, symbol, open, close, high, low, cvd, volume, poc 
        FROM {table} 
        ORDER BY open_time DESC 
        LIMIT {limit}
    """

    with eng.begin() as conn:
        df = pd.read_sql(text(query), conn)

    return df.sort_values('open_time')


def start_grid_bot(balance: int = 1000):
    bot = SpotGridBot(
        usdt_balance=balance,
        asset_balance=0.0,
        grid_levels=6,
        grid_spacing=0.008,
        risk_per_trade=0.08
    )

    # Отримуємо дані
    df = fetch_data(limit=400)

    if len(df) < 50:
        print("Not enough data for analysis")
        return

    # Запускаємо стратегію
    bot.run_buy_strategy(df)

    # Повертаємо список ордерів з поточною ціною
    grid_with_current_price = bot.get_grid_with_current_price()

    # Виводимо інформацію про ордери з поточною ціною
    print(f"\nGrid orders with current price ({bot.current_price:.4f}):")
    for i, order in enumerate(grid_with_current_price):
        status = "FILLED" if order.filled else "ACTIVE"
        price_diff = ((bot.current_price - order.price) / order.price) * 100
        print(f"Order {i + 1}: {order.order_type.upper()} at {order.price:.4f} "
              f"(Current: {bot.current_price:.4f}, Diff: {price_diff:+.2f}%) - {status}")

    return grid_with_current_price


if __name__ == "__main__":
    grid_orders = start_grid_bot()