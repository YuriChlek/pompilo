# План виправлень торгової логіки _OF-rain-bot

> Мета: зробити бота придатним до live-торгівлі та прибутковим.  
> Проблеми ранжовані за впливом на P&L.  
> Для кожної вказано: файл, рядки, діагноз, конкретна зміна коду.

---

## Зміст

1. [КРИТИЧНО — Spot-ціни на futures без basis-коригування](#fix-1)
2. [КРИТИЧНО — Live позиція не управляється після відкриття](#fix-2)
3. [КРИТИЧНО — TEST_EQUITY захардкоджений, risk-параметри ігноруються](#fix-3)
4. [КРИТИЧНО — Протилежний сигнал не скасовує pending-ордер](#fix-4)
5. [КРИТИЧНО — Відсутній таймаут для pending-ордеру](#fix-5)
6. [СЕРЙОЗНО — Цикл стратегії 1 раз/сек](#fix-6)
7. [СЕРЙОЗНО — Відсутній spread filter перед входом](#fix-7)
8. [СЕРЙОЗНО — Take-profit для LONG розраховується неправильно](#fix-8)
9. [СЕРЙОЗНО — Агрегована tape не використовується у фільтрах](#fix-9)
10. [НЕЗНАЧНО — Мертвий код у `_is_long_setup`](#fix-10)
11. [НЕЗНАЧНО — Dead-код risk-констант у config.py](#fix-11)
12. [СЕРЙОЗНО — `tick_size=0.0` у dry-run позиціях блокує break-even](#fix-12)
13. [КРИТИЧНО — `wall is None` блокує будь-яке скасування pending-ордеру](#fix-13)
14. [НЕЗНАЧНО — Подвійний `_transition_to_degraded` у `run_cycle`](#fix-14)

---

<a name="fix-1"></a>
## Fix 1 — Spot-ціни на futures без basis-коригування

**Пріоритет:** КРИТИЧНИЙ  
**Файли:** `orderflow/runtime/orchestrator.py:492-501`, `orderflow/market_data/adapters/bybit.py`, `utils/config.py`

### Діагноз

Бот аналізує **spot order book** (Bybit/Binance/OKX spot) і обчислює `entry_price`, `stop_price`, `take_profit_price` у spot-цінах. Потім ці ціни без змін подаються на **Bybit linear futures** через `BybitExecutionService`.

```python
# orchestrator.py — _prepare_signal_for_execution
def _prepare_signal_for_execution(self, signal: ScalpSignal) -> ScalpSignal:
    signal.execution_entry_price = signal.analysis_entry_price   # spot-ціна!
    signal.execution_stop_price = signal.analysis_stop_price     # spot-ціна!
    signal.execution_take_profit_price = signal.analysis_take_profit_price  # spot-ціна!
    signal.basis_bps = None  # basis ніколи не рахується
    return signal
```

**Наслідок:** Futures-ціна = spot ± basis. У contango (переважна більшість часу) futures торгується ВИЩЕ spot. Лімітний buy-ордер виставлений за spot-ціною на futures-ринку буде або:
- Ніколи не заповнений (ціна вища, ордер лежить під ринком)
- Заповнений значно пізніше, вже в інших умовах ринку

### Рішення

**Крок 1.** Додати REST-запит до Bybit futures API щоб отримувати поточну mark price / last price futures для кожного символу.

**Крок 2.** У `BybitMarketDataAdapter` або окремому `FuturesPriceProvider` зберігати поточний futures bid/ask (або хоча б mark price) для кожного символу.

**Крок 3.** У `_prepare_signal_for_execution` рахувати basis і коригувати ціни:

```python
def _prepare_signal_for_execution(self, signal: ScalpSignal, futures_mid: float | None = None) -> ScalpSignal:
    if signal.direction == SignalDirection.NONE:
        return signal

    spot_mid = signal.wall.price if signal.wall else signal.analysis_entry_price
    if futures_mid is not None and spot_mid and spot_mid > 0:
        basis_bps = (futures_mid - spot_mid) / spot_mid * 10_000
        signal.basis_bps = round(basis_bps, 2)
        offset = futures_mid - spot_mid
    else:
        signal.basis_bps = None
        offset = 0.0

    signal.execution_entry_price = round((signal.analysis_entry_price or 0) + offset, 8)
    signal.execution_stop_price = round((signal.analysis_stop_price or 0) + offset, 8)
    signal.execution_take_profit_price = round((signal.analysis_take_profit_price or 0) + offset, 8)
    signal.execution_invalidation_price = round((signal.analysis_invalidation_price or 0) + offset, 8)
    return signal
```

**Крок 4.** Підписатися на futures mark price WebSocket Bybit (`tickers.{symbol}`) щоб мати актуальну futures-ціну без REST-запитів на кожному циклі.

**Крок 5.** Додати конфіг-параметр `ORDERFLOW_MAX_BASIS_BPS` (наприклад, 50bps) — якщо basis занадто великий, сигнал пропускається (аномалія, не варто входити).

```python
# utils/config.py
ORDERFLOW_MAX_BASIS_BPS = float(os.getenv("ORDERFLOW_MAX_BASIS_BPS", "50"))
```

---

<a name="fix-2"></a>
## Fix 2 — Live позиція не управляється після відкриття

**Пріоритет:** КРИТИЧНИЙ  
**Файл:** `orderflow/runtime/orchestrator.py:434-490`

### Діагноз

У live-режимі `_handle_open_position` завжди повертається після першого блоку — вся логіка управління позицією (break-even, trailing stop, вихід за сигналом) ніколи не виконується:

```python
async def _handle_open_position(self, runtime_state, now_ms, reference_book):
    position = runtime_state.position
    if not self.dry_run:
        live_position = await self.executor.detect_live_position(runtime_state.symbol)
        if live_position is None or float(live_position.get("size") or 0.0) <= 0:
            # запис закриття...
        return  # ← ВСЕ ПІСЛЯ ЦЬОГО — МЕРТВИЙ КОД ДЛЯ LIVE
    # dry-run логіка (break-even, signal exit, тощо) — ніколи не виконується live
```

**Наслідок:** Break-even не переміщується для звичайних fills. Немає виходу якщо зникає стіна або з'являється протилежний сигнал. Позиція закривається тільки коли Bybit сам виконає SL/TP.

### Рішення

**Крок 1.** Після перевірки "чи позиція ще існує" — не виходити, а продовжувати логіку управління:

```python
async def _handle_open_position(self, runtime_state, now_ms, reference_book):
    position = runtime_state.position
    if position is None:
        await self._set_state(runtime_state, BotState.IDLE, "missing_position")
        return

    if not self.dry_run:
        live_position = await self.executor.detect_live_position(runtime_state.symbol)
        if live_position is None or float(live_position.get("size") or 0.0) <= 0:
            market_price = self._current_market_price(position.side, reference_book, position.entry_price)
            exit_reason = self._infer_exchange_exit_reason(position, market_price)
            await self._record_position_closed(...)
            runtime_state.position = None
            await self._enter_cooldown(runtime_state, now_ms, exit_reason, {"closed_by": "exchange"})
            return
        # ← ПРИБРАТИ return тут, продовжувати управління

    # Спільна логіка для live та dry-run:
    reference_exchange = None if reference_book is None else reference_book.exchange
    current_signal = self.signal_engine.evaluate_with_reference(...)
    current_signal = self._prepare_signal_for_execution(current_signal)
    market_price = self._current_market_price(position.side, reference_book, position.entry_price)
    position.best_price_seen = self._best_price_seen(position, market_price)

    # Break-even для live (переміщення стопу на біржі)
    if not self.dry_run and position.tick_size > 0:
        new_stop = self._break_even_stop_price(
            position.side, position.entry_price, market_price, position.tick_size
        )
        if new_stop is not None and self._stop_improves(position.side, position.stop_price, new_stop):
            position.stop_price = new_stop
            await self.executor.move_stop_to_breakeven(
                runtime_state.symbol, position.side, new_stop, market_price, dry_run=False
            )

    exit_reason = self._position_exit_reason(position, current_signal, market_price, now_ms)
    if exit_reason is None:
        return
    # ... вихід
```

**Крок 2.** Додати `_stop_improves` — щоб не рухати стоп в гіршу сторону:

```python
@staticmethod
def _stop_improves(side: str, current_stop: float, new_stop: float) -> bool:
    if side.lower() == "buy":
        return new_stop > current_stop
    return new_stop < current_stop
```

**Крок 3.** Зберігати `tick_size` позиції при активації з futures mark price (а не зі spot-книги), щоб break-even коректно рахував відстань на futures.

**Крок 4.** Додати вихід за протилежним сигналом у live-режимі (якщо стіна зникла і сигнал став протилежним):

```python
# у _position_exit_reason додати:
if current_signal.direction != SignalDirection.NONE and current_signal.direction != original_direction:
    return "signal_reversal"
```

---

<a name="fix-3"></a>
## Fix 3 — TEST_EQUITY захардкоджений, risk-параметри ігноруються

**Пріоритет:** КРИТИЧНИЙ  
**Файли:** `orderflow/runtime/orchestrator.py:22,177`, `orderflow/execution/risk_manager.py:9-10,34-35`, `utils/config.py:118-122`

### Діагноз

```python
# orchestrator.py
TEST_EQUITY = 1000.0
...
equity = TEST_EQUITY  # реальна equity ніколи не читається з Bybit

# risk_manager.py
TEST_EQUITY = Decimal("1000")
TEST_RISK_PER_TRADE_PCT = Decimal("0.5")

def build_order(self, signal, equity):
    del equity  # ← прийнятий параметр видаляється!
    risk_amount = TEST_EQUITY * (TEST_RISK_PER_TRADE_PCT / 100)
```

Параметри у `config.py` (`ORDERFLOW_RISK_PER_TRADE_PCT`, `ORDERFLOW_MAX_DAILY_LOSS_PCT`, `ORDERFLOW_MAX_TRADES_PER_DAY`, `ORDERFLOW_MAX_CONSECUTIVE_LOSSES`) визначені але ніде не імпортуються і не використовуються.

### Рішення

**Крок 1.** Додати метод в `AsyncBybitTradingClient` для отримання балансу рахунку:

```python
async def get_wallet_balance(self) -> float:
    data = await self.transport.request(
        "GET",
        "/v5/account/wallet-balance",
        {"accountType": "UNIFIED"},
    )
    result = self._unwrap_result(data, "get_wallet_balance")
    accounts = result.get("list") or []
    if not accounts:
        return 0.0
    return float(accounts[0].get("totalEquity") or 0.0)
```

**Крок 2.** Читати equity перед кожним циклом (або раз на N секунд):

```python
# orchestrator.py — run_cycle
async def run_cycle(self) -> None:
    now_ms = int(time.time() * 1000)
    equity = await self._get_current_equity()
    ...

async def _get_current_equity(self) -> float:
    if self.dry_run:
        return TEST_EQUITY
    try:
        balance = await self.executor.trading_client.get_wallet_balance()
        return balance if balance > 0 else TEST_EQUITY
    except Exception:
        return self._last_known_equity or TEST_EQUITY
```

**Крок 3.** Підключити реальні параметри ризику у `RiskManager`:

```python
from utils.config import (
    ORDERFLOW_RISK_PER_TRADE_PCT,
    ORDERFLOW_MAX_DAILY_LOSS_PCT,
    ORDERFLOW_MAX_TRADES_PER_DAY,
    ORDERFLOW_MAX_CONSECUTIVE_LOSSES,
)

class RiskManager:
    def __init__(self) -> None:
        self._daily_trades = 0
        self._daily_loss_pct = 0.0
        self._consecutive_losses = 0
        self._day_start = datetime.utcnow().date()

    def build_order(self, signal: ScalpSignal, equity: float) -> dict | None:
        if not self._check_daily_limits():
            return None
        risk_amount = Decimal(str(equity)) * (Decimal(str(ORDERFLOW_RISK_PER_TRADE_PCT)) / 100)
        ...

    def _check_daily_limits(self) -> bool:
        today = datetime.utcnow().date()
        if today != self._day_start:
            self._reset_daily_counters()
        if self._daily_trades >= ORDERFLOW_MAX_TRADES_PER_DAY:
            return False
        if self._daily_loss_pct >= ORDERFLOW_MAX_DAILY_LOSS_PCT:
            return False
        if self._consecutive_losses >= ORDERFLOW_MAX_CONSECUTIVE_LOSSES:
            return False
        return True

    def record_trade_result(self, pnl_pct: float) -> None:
        self._daily_trades += 1
        if pnl_pct < 0:
            self._daily_loss_pct += abs(pnl_pct)
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
```

**Крок 4.** Викликати `risk_manager.record_trade_result()` після кожного закриття позиції в `_record_position_closed`.

---

<a name="fix-4"></a>
## Fix 4 — Протилежний сигнал не скасовує pending-ордер

**Пріоритет:** КРИТИЧНИЙ  
**Файл:** `orderflow/runtime/orchestrator.py:725-753`

### Діагноз

```python
def _pending_entry_invalidation_reason(self, pending, current_signal, reference_book, now_ms, dry_run):
    del dry_run
    if pending.signal.wall is None:  # ← окремий баг: якщо wall=None — ніколи не скасовує (Fix 13)
        return None

    wall_is_active = self.signal_engine.wall_is_active(...)

    # ← ПОМИЛКА: скасовує тільки якщо signal змінився І стіна зникла (обидві умови!)
    if current_signal.direction not in {SignalDirection.NONE, pending.signal.direction} and not wall_is_active:
        return "wall_disappeared"

    if current_signal.reason in {"missing_analysis_book", "stale_analysis_book"}:
        return None

    if reference_book is None:
        return None

    if not wall_is_active:
        return "wall_disappeared"

    return None
```

Якщо чекаємо **LONG** і з'явився **SHORT**-сигнал (захищена ask-стіна), але bid-стіна ще технічно активна — ордер залишається. Ринок змінив напрямок, а бот продовжує чекати входу в протилежну сторону.

### Рішення

Розділити дві незалежні умови скасування:

```python
def _pending_entry_invalidation_reason(self, pending, current_signal, reference_book, now_ms, dry_run) -> str | None:
    del dry_run

    # Не скасовуємо при стейл/відсутній книзі — wall_is_active поверне False через stale-check,
    # що призведе до хибного "wall_disappeared" при тимчасово відсутньому feed'і
    if current_signal.reason in {"missing_analysis_book", "stale_analysis_book"}:
        return None

    # 1. Протилежний сигнал — скасовуємо НЕЗАЛЕЖНО від стану стіни
    if (current_signal.direction != SignalDirection.NONE
            and current_signal.direction != pending.signal.direction):
        return "signal_reversed"

    if pending.signal.wall is None:
        return None

    if reference_book is None:
        return None

    wall_is_active = self.signal_engine.wall_is_active(
        pending.symbol, pending.signal.wall, now_ms, ORDERFLOW_PENDING_WALL_TOLERANCE_TICKS
    )

    # 2. Стіна зникла
    if not wall_is_active:
        return "wall_disappeared"

    return None
```

---

<a name="fix-5"></a>
## Fix 5 — Відсутній таймаут для pending-ордеру

**Пріоритет:** КРИТИЧНИЙ  
**Файл:** `orderflow/runtime/orchestrator.py:299-433`, `utils/config.py`

### Діагноз

У `_handle_pending_entry` немає жодної перевірки часу. Лімітний ордер може чекати виконання нескінченно довго. У реальній торгівлі: стіна може бути активна 5-15 хвилин (ціна відійшла від неї), ордер виконається пізніше у невигідний момент.

### Рішення

**Крок 1.** Додати параметр у config:

```python
# utils/config.py
ORDERFLOW_PENDING_ORDER_MAX_AGE_SECONDS = int(os.getenv("ORDERFLOW_PENDING_ORDER_MAX_AGE_SECONDS", "30"))
```

**Крок 2.** Додати перевірку на початку `_handle_pending_entry`:

```python
async def _handle_pending_entry(self, runtime_state, now_ms, reference_book):
    pending = runtime_state.pending_order
    if pending is None:
        await self._set_state(runtime_state, BotState.IDLE, "missing_pending_order")
        return

    # Таймаут ордеру
    age_ms = now_ms - pending.created_at_ms
    if age_ms > ORDERFLOW_PENDING_ORDER_MAX_AGE_SECONDS * 1000:
        await self.executor.cancel_entry(
            runtime_state.symbol, pending.order_id, dry_run=self.dry_run, reason="order_timeout"
        )
        await self._enter_cooldown(runtime_state, now_ms, "order_timeout", {"age_ms": age_ms})
        return

    # ... решта логіки без змін
```

---

<a name="fix-6"></a>
## Fix 6 — Цикл стратегії 1 раз/сек

**Пріоритет:** СЕРЙОЗНИЙ  
**Файл:** `orderflow/runtime/orchestrator.py:78-83`, `utils/config.py`

### Діагноз

```python
async def _strategy_loop(self) -> None:
    while True:
        self._log_runtime_heartbeat()
        self._log_runtime_state_change()
        await self.run_cycle()
        await asyncio.sleep(1)  # 1 секунда між ітераціями
```

Стіни живуть 800ms–2000ms. При 1-секундному циклі бот може:
- Побачити стіну вже після її зникнення
- Пропустити момент оптимального входу
- Обробити pending/position занадто рідко

### Рішення

**Крок 1.** Параметризувати інтервал циклу:

```python
# utils/config.py
ORDERFLOW_STRATEGY_LOOP_MS = int(os.getenv("ORDERFLOW_STRATEGY_LOOP_MS", "200"))
```

**Крок 2.** Замінити `asyncio.sleep(1)` на параметризований інтервал (logging-виклики вже є):

```python
async def _strategy_loop(self) -> None:
    while True:
        self._log_runtime_heartbeat()
        self._log_runtime_state_change()
        await self.run_cycle()
        await asyncio.sleep(ORDERFLOW_STRATEGY_LOOP_MS / 1000)
```

**Важливо:** При більш частому циклі збільшиться кількість REST-запитів до Bybit (poll_entry, detect_live_position). Потрібно обмежити їх частоту окремим debounce на рівні символу або перейти на WebSocket-потік для статусів ордерів.

---

<a name="fix-7"></a>
## Fix 7 — Відсутній spread filter перед входом

**Пріоритет:** СЕРЙОЗНИЙ  
**Файл:** `orderflow/strategy/scalp_signal_engine.py`, `utils/config.py`

### Діагноз

Перед входом ніхто не перевіряє поточний spread. Для скальпу зі стопом 2-3 тіки це критично: якщо spread = 2 тіки, а entry offset = 1 тік, ризик на слідж поглинає половину стопу ще до будь-якого руху ціни.

Крім того, бот ставить лімітні ордери на futures за spot-цінами. Spread на futures часто інший.

### Рішення

**Крок 1.** Додати параметр:

```python
# utils/config.py
ORDERFLOW_MAX_SPREAD_TICKS = int(os.getenv("ORDERFLOW_MAX_SPREAD_TICKS", "3"))
```

**Крок 2.** Перевірку додати у `_is_long_setup` та `_is_short_setup`:

```python
@staticmethod
def _is_long_setup(reference_book, bid_wall, ask_wall, buy_notional, sell_notional, cross_confirmations):
    # Новий фільтр: spread занадто великий
    if reference_book.spread_ticks > ORDERFLOW_MAX_SPREAD_TICKS:
        return False
    ...
```

**Крок 3.** Після впровадження Fix 1 (basis) — додати аналогічний spread-фільтр для futures-книги (якщо вона підключена).

---

<a name="fix-8"></a>
## Fix 8 — Take-profit для LONG розраховується неправильно

**Пріоритет:** СЕРЙОЗНИЙ  
**Файл:** `orderflow/strategy/scalp_signal_engine.py:316-334`

### Діагноз

```python
@staticmethod
def _take_profit_price(entry_price, risk, target_wall_price, direction, tick_size):
    if direction == "long":
        fallback = entry_price + risk * ORDERFLOW_TAKE_PROFIT_R_MULTIPLE
        if target_wall_price is None:
            return fallback
        wall_target = target_wall_price - tick_size * ORDERFLOW_ENTRY_OFFSET_TICKS
        return max(fallback, wall_target)  # ← ПОМИЛКА: бере БІЛЬШЕ
```

Логіка: якщо протилежна (ask) стіна стоїть близько — TP має бути трохи нижче неї, щоб ціна не відбилась і не перетворила профіт на стоп. Але `max(fallback, wall_target)` обирає ціль ДАЛІ, тобто TP ставиться ЗА стіну. Ціна майже гарантовано відіб'ється від стіни раніше.

**Приклад:** entry=100, risk=0.3, fallback=100.45 (1.5R). Ask-стіна на 100.20. wall_target=100.19. `max(100.45, 100.19) = 100.45` — TP виставлено за стіну, ціна відіб'ється на 100.20 і розвернеться.

Правильна логіка: TP = **min(fallback, wall_target)** для LONG (зупинитись перед стіною).

### Рішення

```python
@staticmethod
def _take_profit_price(entry_price, risk, target_wall_price, direction, tick_size):
    if direction == "long":
        fallback = entry_price + risk * ORDERFLOW_TAKE_PROFIT_R_MULTIPLE
        if target_wall_price is None:
            return fallback
        wall_target = target_wall_price - tick_size * ORDERFLOW_ENTRY_OFFSET_TICKS
        # Зупинитись ПЕРЕД стіною (менше з двох)
        return min(fallback, wall_target) if wall_target > entry_price else fallback

    # SHORT
    fallback = entry_price - risk * ORDERFLOW_TAKE_PROFIT_R_MULTIPLE
    if target_wall_price is None:
        return fallback
    wall_target = target_wall_price + tick_size * ORDERFLOW_ENTRY_OFFSET_TICKS
    # Зупинитись ПЕРЕД стіною (більше з двох для short — ближче до entry)
    return max(fallback, wall_target) if wall_target < entry_price else fallback
```

**Додаткова перевірка:** якщо `wall_target <= entry_price` (для LONG) або `wall_target >= entry_price` (для SHORT) — стіна знаходиться з "неправильного боку", такий wall_target слід ігнорувати і використовувати fallback.

---

<a name="fix-9"></a>
## Fix 9 — Агрегована tape не використовується у фільтрах

**Пріоритет:** СЕРЙОЗНИЙ  
**Файл:** `orderflow/strategy/scalp_signal_engine.py:81-90, 219-247`

### Діагноз

```python
aggregate_tape = self.tape_store.get_aggregated_window_stats(symbol, now_ms, ORDERFLOW_TAPE_WINDOW_MS)
# Потрапляє тільки в metadata — не впливає на прийняття рішення про вхід
```

Tape pressure перевіряється тільки по `reference_tape` (одна біржа — Bybit spot). Але якщо на Binance та OKX у той же час йде протилежний потік — сигнал все одно пройде.

### Рішення

**Варіант A (консервативний):** Якщо aggregate tape суперечить reference tape — блокувати вхід:

```python
@staticmethod
def _is_long_setup(reference_book, bid_wall, ask_wall, buy_notional, sell_notional,
                   cross_confirmations, agg_buy_notional=0.0, agg_sell_notional=0.0):
    ...
    if sell_notional <= buy_notional * ORDERFLOW_MIN_TAPE_PRESSURE_RATIO:
        return False
    # Aggregate tape не має суперечити (якщо є дані від 2+ бірж)
    if agg_buy_notional > 0 or agg_sell_notional > 0:
        if agg_sell_notional <= agg_buy_notional * ORDERFLOW_MIN_TAPE_PRESSURE_RATIO:
            return False  # глобально переважає buying — не входимо в long
    ...
```

**Варіант B (м'який):** Використовувати aggregate tape для зважування confidence замість повного блокування.

**Рекомендація:** Варіант A. Він зменшить кількість угод, але підвищить якість сигналів.

Передавати aggregate stats у `_is_long_setup`:
```python
long_agg_buy = aggregate_tape.buy_notional
long_agg_sell = aggregate_tape.sell_notional

if best_bid_wall and self._is_long_setup(
    reference_book, best_bid_wall, best_ask_wall,
    reference_tape.buy_notional, reference_tape.sell_notional,
    long_cross_confirmations,
    agg_buy_notional=long_agg_buy,
    agg_sell_notional=long_agg_sell,
):
```

---

<a name="fix-10"></a>
## Fix 10 — Мертвий код у `_is_long_setup` / `_is_short_setup`

**Пріоритет:** НЕЗНАЧНИЙ  
**Файл:** `orderflow/strategy/scalp_signal_engine.py:227, 237, 257, 267`

### Діагноз

```python
if bid_wall.distance_ticks > ORDERFLOW_MAX_WALL_DISTANCE_TICKS:  # > 8 — МЕРТВИЙ
    return False
...
if bid_wall.distance_ticks > ORDERFLOW_TEST_TOUCH_TICKS:  # > 2 — РЕАЛЬНА перевірка
    return False
```

`ORDERFLOW_TEST_TOUCH_TICKS = 2 < ORDERFLOW_MAX_WALL_DISTANCE_TICKS = 8`. Перша перевірка ніколи не відсіює нічого, що не відсіює друга. Паралельно `LiquidityDetector` витрачає час на стіни 3-8 тіків, яких `ScalpSignalEngine` все одно відкине.

### Рішення

**Крок 1.** Прибрати дублікат у `_is_long_setup` та `_is_short_setup`.

**Крок 2.** У `LiquidityDetector._find_walls` замінити `ORDERFLOW_MAX_WALL_DISTANCE_TICKS` на `ORDERFLOW_TEST_TOUCH_TICKS` щоб не шукати стіни, які гарантовано будуть відкинуті:

```python
# liquidity_detector.py — _find_walls
for level in levels:
    if level.distance_ticks > ORDERFLOW_TEST_TOUCH_TICKS:  # замість MAX_WALL_DISTANCE_TICKS
        continue
```

Або виправити логіку — `ORDERFLOW_MAX_WALL_DISTANCE_TICKS` може відповідати за пошук стін для TP (не тільки для entry), тоді його варто залишити для детектора, але в `_is_long_setup` першу перевірку видалити.

---

<a name="fix-11"></a>
## Fix 11 — Dead-код risk-констант у config.py

**Пріоритет:** НЕЗНАЧНИЙ (вирішується після Fix 3)  
**Файл:** `utils/config.py:118-122`

### Діагноз

Чотири параметри визначені, але не імпортуються і не використовуються ніде в коді:

```python
ORDERFLOW_RISK_PER_TRADE_PCT = float(os.getenv("ORDERFLOW_RISK_PER_TRADE_PCT", "0.15"))
ORDERFLOW_MAX_DAILY_LOSS_PCT = float(os.getenv("ORDERFLOW_MAX_DAILY_LOSS_PCT", "1.5"))
ORDERFLOW_MAX_TRADES_PER_DAY = int(os.getenv("ORDERFLOW_MAX_TRADES_PER_DAY", "30"))
ORDERFLOW_MAX_CONSECUTIVE_LOSSES = int(os.getenv("ORDERFLOW_MAX_CONSECUTIVE_LOSSES", "5"))
```

### Рішення

Підключити у `RiskManager` (реалізація описана в Fix 3).

---

<a name="fix-12"></a>
## Fix 12 — `tick_size=0.0` у dry-run позиціях блокує break-even

**Пріоритет:** СЕРЙОЗНИЙ  
**Файл:** `orderflow/runtime/orchestrator.py:248, 358`

### Діагноз

У двох dry-run шляхах створення позиції `tick_size` захардкоджений у `0.0`:

```python
# orchestrator.py:243 — immediate fill
runtime_state.position = PositionState(
    ...
    tick_size=0.0,  # ← завжди 0 в dry-run
    ...
)

# orchestrator.py:353 — pending fill
runtime_state.position = PositionState(
    ...
    tick_size=0.0,  # ← завжди 0 в dry-run
    ...
)
```

Метод `_break_even_stop_price` одразу повертає `None` якщо `tick_size <= 0`:

```python
@staticmethod
def _break_even_stop_price(side, entry_price, current_price, tick_size):
    if tick_size <= 0:
        return None  # ← break-even ніколи не спрацює в dry-run
```

**Наслідок:** Break-even логіка неможливо протестувати в dry-run режимі. У live-режимі ця проблема відсутня (`_activate_position_from_fill` читає `reference_book.tick_size`).

### Рішення

Замінити `0.0` на реальний tick_size з reference_book в обох місцях:

```python
tick_size=float(reference_book.tick_size) if reference_book is not None else 0.0,
```

---

<a name="fix-13"></a>
## Fix 13 — `wall is None` блокує будь-яке скасування pending-ордеру

**Пріоритет:** КРИТИЧНИЙ  
**Файл:** `orderflow/runtime/orchestrator.py:734-736`

### Діагноз

На початку `_pending_entry_invalidation_reason` стоїть ранній return:

```python
if pending.signal.wall is None:
    return None  # ← ордер ніколи не скасовується якщо wall=None
```

Якщо `pending.signal.wall is None`, функція завжди повертає `None` — тобто ордер ніколи не буде скасований через `_pending_entry_invalidation_reason`, навіть якщо з'явився протилежний сигнал. Скасування відбудеться лише через таймаут (Fix 5), що означає очікування до 30 секунд у повністю невалідній ситуації.

**Взаємодія з Fix 4:** Рішення Fix 4 переносить перевірку протилежного сигналу ВИЩЕ цього раннього return. Після застосування Fix 4 протилежний сигнал буде скасовувати ордер навіть при `wall=None`. Wall-based invalidation при `wall=None` залишається коректно відключеною (нема стіни — нема що перевіряти).

### Рішення

Вирішується повністю при впровадженні Fix 4 (stale guard + opposite signal check переміщені вище `wall is None`). Окремих додаткових змін не потрібно.

---

<a name="fix-14"></a>
## Fix 14 — Подвійний `_transition_to_degraded` у `run_cycle`

**Пріоритет:** НЕЗНАЧНИЙ  
**Файл:** `orderflow/runtime/orchestrator.py:192-199`

### Діагноз

`run_cycle` складається з двох послідовних циклів по `symbol_states`. Перший цикл обробляє `ENTRY_PENDING` та `IN_POSITION`, а також викликає `_transition_to_degraded` для символів без reference_book:

```python
# Перший цикл (рядки 179-199)
for symbol, runtime_state in self.symbol_states.items():
    ...
    if runtime_state.state == BotState.ENTRY_PENDING ...:
        ...
        continue
    if runtime_state.state == BotState.IN_POSITION ...:
        ...
        continue
    if reference_book is None:
        await self._transition_to_degraded(...)  # ← перший виклик
        continue
```

Другий цикл робить те саме для тих самих символів (IDLE/CANDIDATE без reference_book):

```python
# Другий цикл (рядки 201-229)
for symbol, runtime_state in self.symbol_states.items():
    ...
    if reference_book is None:
        await self._transition_to_degraded(...)  # ← другий виклик за той самий цикл
        continue
```

**Наслідок:** Для IDLE/CANDIDATE символів без reference_book за кожну ітерацію:
- `get_best_reference_book()` викликається двічі
- `_transition_to_degraded` викликається двічі (DB-insert блокується `_should_skip_redundant_transition`, але решта логіки виконується)

### Рішення

Видалити DEGRADED-перевірку з першого циклу (рядки 192-199) — другий цикл вже обробляє цей випадок. Перший цикл повинен займатись виключно `ENTRY_PENDING` та `IN_POSITION` станами:

```python
# Перший цикл — тільки обробка активних станів
for symbol, runtime_state in self.symbol_states.items():
    if runtime_state.cooldown_until_ms > now_ms:
        continue
    reference_book = self.feed_manager.get_best_reference_book(symbol, now_ms=now_ms)
    if runtime_state.state == BotState.ENTRY_PENDING and runtime_state.pending_order is not None:
        await self._handle_pending_entry(runtime_state, now_ms, reference_book)
        continue
    if runtime_state.state == BotState.IN_POSITION and runtime_state.position is not None:
        await self._handle_open_position(runtime_state, now_ms, reference_book)
        continue
    # DEGRADED/idle/candidate — обробляються другим циклом
```

---

## Порядок реалізації

Процес розбитий на 4 логічні фази для поступового переходу від виправлення базових багів до складних архітектурних змін.

| Черга | Фаза | Fix | Складність | Вплив |
|-------|------|-----|------------|-------|
| **1** | **I. Логічна стабільність** | **Fix 5** — Таймаут pending-ордеру | Низька | КРИТИЧНИЙ |
| **2** | | **Fix 4 (+13)** — Скасування при реверсі/wall=None | Низька | КРИТИЧНИЙ |
| **3** | | **Fix 8** — Корекція TP для LONG (max → min) | Низька | СЕРЙОЗНИЙ |
| **4** | | **Fix 12** — tick_size для Dry-run (BE test) | Низька | СЕРЙОЗНИЙ |
| **5** | **II. Ризик-менеджмент** | **Fix 3 (+11)** — Реальна Equity та ліміти | Середня | КРИТИЧНИЙ |
| **6** | | **Fix 7** — Spread filter | Низька | СЕРЙОЗНИЙ |
| **7** | **III. Якість та Швидкість** | **Fix 6** — Цикл 200ms (замість 1s) | Середня | СЕРЙОЗНИЙ |
| **8** | | **Fix 9** — Глобальна стрічка (Aggregated tape) | Середня | СЕРЙОЗНИЙ |
| **9** | | **Fix 10** — Оптимізація пошуку стін | Низька | НЕЗНАЧНИЙ |
| **10** | **IV. Live Execution** | **Fix 2** — Управління позицією у Live (BE/Exit) | Висока | КРИТИЧНИЙ |
| **11** | | **Fix 1** — Basis adjustment (Spot vs Futures) | Висока | КРИТИЧНИЙ |
| **12** | | **Fix 14** — Очистка подвійних переходів | Низька | НЕЗНАЧНИЙ |

---

## Примітки

- **Fix 1 (basis)** — найскладніший технічно, але найважливіший для live-торгівлі. Без нього live-режим не може коректно виконувати угоди.
- **Fix 2 (position management)** — треба впроваджувати поступово: спочатку break-even, потім signal-exit.
- **Fix 6 (цикл 200ms)** — потребує аудиту частоти REST-запитів щоб не отримати rate-limit від Bybit.
- Після впровадження Fix 3 видалити `TEST_EQUITY` і `TEST_RISK_PER_TRADE_PCT` з `orchestrator.py` та `risk_manager.py`.
