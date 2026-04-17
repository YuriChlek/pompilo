# Order Flow Scalp Bot

Точка входу бота: [main.py](/home/yurii/Proj/pompilo/_OF-rain-bot/main.py)

## Команди запуску

### 1. Live режим

Бот запускається у live-режимі, якщо стартувати його без прапорців:

```bash
python3 main.py
```

Альтернатива через локальне віртуальне середовище:

```bash
.venv/bin/python main.py
```

У цьому режимі бот:
- підключається до spot market data streams
- аналізує стакан на spot ринку Bybit, Binance та OKX
- відкриває позиції на Bybit linear futures

Bybit endpoints розділені:
- market data REST: `BYBIT_MARKET_DATA_API_ENDPOINT` за замовчуванням `https://api.bybit.com`
- market data WebSocket: `BYBIT_MARKET_DATA_WS_ENDPOINT` за замовчуванням `wss://stream.bybit.com/v5/public/spot`
- trading REST: `BYBIT_TRADING_API_ENDPOINT` за замовчуванням `https://api-demo.bybit.com`

### 2. Dry-run режим

Dry-run вмикається окремим прапорцем:

```bash
python3 main.py --dry-run
```

Альтернатива через локальне віртуальне середовище:

```bash
.venv/bin/python main.py --dry-run
```

У цьому режимі бот:
- аналізує spot ринок як звичайно
- рахує `entry`, `stop loss`, `take profit`, `size`
- записує події в БД
- не відправляє реальні ордери на Bybit futures

## Що зараз підтримує CLI

Поточний entrypoint підтримує лише:
- запуск без аргументів
- `--dry-run`

Перевірити це можна так:

```bash
python3 main.py --help
```

## Поточна модель ринків

- Аналіз книги заявок і стрічки угод виконується по spot-ринку на Bybit, Binance та OKX.
- Reference book для побудови сигналу за замовчуванням береться з Bybit spot.
- Виконання ордерів і керування позицією виконується окремо через Bybit linear futures.
