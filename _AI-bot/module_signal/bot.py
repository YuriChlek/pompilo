from module_ai import get_ai_signal
from module_indicators import get_of_data, weighted_signal, advanced_weighted_signal
from module_grid import start_grid_bot
from utils import TradeSignal
from pprint import pprint


def get_trading_signal(symbol):
    ai_signal = get_ai_signal(symbol)
    of_data = get_of_data(symbol)
    print("=" * 60 + "\n")

    weight = weighted_signal(of_data)

    ai_rez_path = f"{str(symbol).lower()}_ai_rez.txt"
    max_rsi_val = int(40) if of_data.market_trend == 'bearish' else int(55)

    if (
            ai_signal.signal == TradeSignal.BUY and weight['signal'] == TradeSignal.BUY and
            int(round(of_data.indicators['rsi'])) < max_rsi_val and
            of_data.cvd['trend'] == 'bullish' and
            (of_data.cvd['strength'] == 'strong' or of_data.cvd['strength'] == 'very_strong')
    ):
        grid = start_grid_bot(ai_signal.entry_price, of_data.market_trend)

        with open(ai_rez_path, "a", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(f"Time {ai_signal.time}\n")
            f.write(f"Position {ai_signal.signal}\n")
            f.write(f"Entry price {ai_signal.entry_price}\n")
            f.write(f"Market trend {of_data.market_trend}\n")
            f.write(f"CVD trend {of_data.cvd['trend']}\n")
            f.write(f"CVD strength {of_data.cvd['strength']}\n")
            f.write(f"RSI {of_data.indicators['rsi']}\n")
            f.write(f"ATR {of_data.indicators['atr']}\n")
            f.write(f"Signal {weight['signal']}\n")
            f.write(f"Confidence {weight['confidence']}\n")
            f.write(f"Score {weight['score']}\n")

            for i, level in enumerate(grid):
                status = "FILLED" if level.filled else "ACTIVE"
                f.write(f"Level {i + 1}: {level.order_type.upper()} at {level.price:.4f} - {status}\n")
            f.write("=" * 60 + "\n")

            return {
                'order_side': 'Buy',
                'time': ai_signal.time,
                'price': None,
                'symbol': symbol,
                'order_price': None,
                'order_grid': grid
            }
    elif (((weight['signal'] == TradeSignal.SELL and (weight['confidence']) > float(60)) or TradeSignal.SELL == ai_signal.signal) and
          of_data.indicators['rsi'] > 70 and
          of_data.cvd['trend'] == 'bearish' and
          (of_data.cvd['strength'] == 'strong' or of_data.cvd['strength'] == 'very_strong') and
          of_data.market_trend != 'neutral'
    ):
        with open(ai_rez_path, "a", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(f"Time {ai_signal.time}\n")
            f.write(f"Position {TradeSignal.SELL}\n")
            f.write(f"Entry price {ai_signal.entry_price}\n")
            f.write(f"Market trend {of_data.market_trend}\n")
            f.write(f"CVD trend {of_data.cvd['trend']}\n")
            f.write(f"CVD strength {of_data.cvd['strength']}\n")
            f.write(f"RSI {of_data.indicators['rsi']}\n")
            f.write(f"ATR {of_data.indicators['atr']}\n")
            f.write(f"Signal {weight['signal']}\n")
            f.write(f"Confidence {weight['confidence']}\n")
            f.write(f"Score {weight['score']}\n")
            f.write("=" * 60 + "\n")

        return {
            'order_side': 'Sell',
            'time': ai_signal.time,
            'price': ai_signal.entry_price,
            'symbol': symbol,
            'order_price': ai_signal.entry_price,
            'order_grid': None
        }
    return None
