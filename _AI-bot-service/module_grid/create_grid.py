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
            base_grid_levels: int = 10,
            grid_spacing: float = 0.02,
            risk_per_trade: float = 0.1
    ):
        self.usdt_balance = usdt_balance
        self.asset_balance = asset_balance
        self.base_grid_levels = base_grid_levels - 1
        self.grid_spacing = grid_spacing
        self.risk_per_trade = risk_per_trade
        self.grid_lines: List[GridLevel] = []
        self.trade_history = []
        self.average_buy_price = 0.0
        self.total_invested = 0.0
        self.current_price = 0.0

        """
            Ініціалізує нового grid-бота.

            Args:
                usdt_balance: Початковий баланс у USDT.
                asset_balance: Початковий баланс у базовому активі.
                base_grid_levels: Базова кількість рівнів сітки.
                grid_spacing: Відстань між рівнями сітки (у відсотках).
                risk_per_trade: Частка балансу, що використовується на один ордер.
        """

    def _calculate_grid_levels_based_on_trend(self) -> int:
        """
            Повертає кількість рівнів сітки з урахуванням ринкового тренду.

            Returns:
                int: Кількість рівнів сітки.
        """
        return self.base_grid_levels

    def _calculate_grid_bounds(self, current_price: float, market_trend: str) -> Tuple[float, float]:
        """
            Розраховує верхню та нижню межу сітки на основі поточної ціни та тренду.

            Args:
                current_price: Поточна ринкова ціна.
                market_trend: Тренд ринку ('bullish', 'bearish', 'neutral').

            Returns:
                Tuple[float, float]: Нижня та верхня межі сітки.
        """
        trend_adjustments = {
            'bullish': {
                'lower_multiplier': 3,
                'grid_spacing': 0.005
            },
            'bearish': {
                'lower_multiplier': 6,
                'grid_spacing': 0.012
            },
            'neutral': {
                'lower_multiplier': 4,
                'grid_spacing': 0.01
            }
        }

        adjustment = trend_adjustments.get(market_trend.lower(), trend_adjustments.get(market_trend.lower(), 1.0))
        self.grid_spacing = adjustment['grid_spacing']
        self.base_grid_levels = adjustment['lower_multiplier']

        lower_bound = current_price * (1 - self.grid_spacing * adjustment['lower_multiplier'])
        upper_bound = current_price

        return lower_bound, upper_bound

    def setup_buy_grid(self, current_price: float, market_trend: str = 'bearish') -> None:
        """
            Налаштовує сітку ордерів для купівлі.

            Args:
                current_price: Поточна ринкова ціна.
                market_trend: Тренд ринку ('bullish', 'bearish', 'neutral').
        """
        self.current_price = current_price

        # Розрахунок параметрів сітки на основі тренду
        lower_bound, upper_bound = self._calculate_grid_bounds(current_price, market_trend)
        grid_levels = self._calculate_grid_levels_based_on_trend()

        print(f"Setting up BUY grid from {lower_bound:.4f} to {upper_bound:.4f}")
        print(f"Current price: {current_price:.4f}")
        print(f"Market trend: {market_trend.upper()}")
        print(f"Grid levels: {grid_levels}")

        # Очищаємо попередню сітку
        self.grid_lines = []

        # Створюємо рівні тільки для купівлі
        price_levels = np.linspace(lower_bound, current_price, grid_levels)

        # Розподіляємо баланс
        investment_per_level = self.usdt_balance / grid_levels * self.risk_per_trade

        for price in price_levels:
            quantity = investment_per_level / price

            self.grid_lines.append(GridLevel(
                price=price,
                order_type='buy',
                quantity=quantity,
                current_price=current_price
            ))

        print(f"Created {len(self.grid_lines)} BUY orders")

    def run_buy_strategy(self, current_price: float, market_trend: str = 'bearish') -> None:
        """
            Запускає стратегію купівлі з урахуванням тренду та поточної ціни.

            Args:
                current_price: Поточна ринкова ціна.
                market_trend: Тренд ринку ('bullish', 'bearish', 'neutral').
        """
        self.current_price = current_price

        if not self.grid_lines:
            print("Setting up initial BUY grid...")
            self.setup_buy_grid(current_price, market_trend)
        else:
            # Перевіряємо чи потрібно оновити сітку
            active_prices = [level.price for level in self.grid_lines if not level.filled]
            if not active_prices or current_price < min(active_prices):
                print("Price moved below grid, resetting...")
                self.setup_buy_grid(current_price, market_trend)

    def _print_current_status(self, current_price: float) -> None:
        """
            Виводить у консоль поточний стан балансу та угод.

            Args:
                current_price: Поточна ринкова ціна.
        """
        total_value = self.usdt_balance + (self.asset_balance * current_price)

        print(f"\nCurrent Status:")
        print(f"USDT Balance: {self.usdt_balance:.2f}")
        print(f"Asset Balance: {self.asset_balance:.6f}")
        print(f"Total Value: {total_value:.2f} USDT")
        print(f"Current Price: {current_price:.4f}")

        if self.average_buy_price > 0:
            unrealized_pnl = ((current_price - self.average_buy_price) / self.average_buy_price) * 100
            print(f"Avg Buy Price: {self.average_buy_price:.4f}")
            print(f"Unrealized PnL: {unrealized_pnl:+.2f}%")

        active_orders = sum(1 for level in self.grid_lines if not level.filled)
        print(f"Active BUY orders: {active_orders}")

    def get_grid_with_current_price(self) -> List[GridLevel]:
        """
            Повертає список ордерів сітки з оновленою поточною ціною.

            Returns:
                List[GridLevel]: Список рівнів сітки.
        """
        return self.grid_lines


def start_grid_bot(current_price: float, market_trend: str = 'bearish', balance: int = 1000) -> List[GridLevel]:
    """
        Запускає grid-бота з заданими параметрами.

        Args:
            current_price: Поточна ринкова ціна.
            market_trend: Тренд ринку ('bullish', 'bearish', 'sideways', 'neutral').
            balance: Баланс USDT для торгівлі.

        Returns:
            List[GridLevel]: Список ордерів сітки з поточною ціною.
    """
    bot = SpotGridBot(
        usdt_balance=balance,
        asset_balance=0.0,
        base_grid_levels=6,
        grid_spacing=0.008,
        risk_per_trade=0.08
    )

    bot.run_buy_strategy(current_price, market_trend)
    grid_with_current_price = bot.get_grid_with_current_price()

    return grid_with_current_price
