# Release Plan: spot-greenwich-bot

## Мета

Довести бота до виробничого стану з підтримкою двох таймфреймів (D1 + 4H).
Поточний стан: архітектура завершена, критичні баги наявні, 4H не реалізовано.

---

## Ключовий торговий принцип: No-Loss

**Бот ніколи не продає актив у збиток.** Це жорстка інваріантна вимога, яка має
бути гарантована на кількох рівнях, а не лише через один параметр конфігурації.

Визначення збитку враховує торгові комісії:

```
min_sell_price = avg_entry_price × (1 + fee_rate_buy + fee_rate_sell + slippage_buffer)
```

При Bybit taker fee 0.1% на BUY, 0.1% на SELL і production safety/slippage buffer 0.8%:

```
min_sell_price = avg_entry_price × 1.01   (мінімум 1% понад собівартість)
```

Поточний стан реалізації:
- Захист **частково існує** у `domain/execution.py:72` через `MIN_PROFIT_RATIO`
- **Проблема 1:** `MIN_PROFIT_RATIO = 0` за замовчуванням — дозволяє продаж за ціною беззбитковості, яка після комісій стає збитком
- **Проблема 2:** захист застосовується лише на рівні `signal_price`, але не перевіряється повторно за фактичною `executed_price` (slippage може перетнути межу)
- **Проблема 3:** після примусової reconciliation `avg_entry_price` може змінитися, але вже прийняте рішення на продаж залишається у черзі

Реалізація принципу описана в **Фазі 4.5** нижче.

---

## Фази

| # | Назва | Пріоритет | Залежності |
|---|-------|-----------|------------|
| 1 | Критичні виправлення | блокує реліз | — |
| 2 | Надійність інфраструктури | блокує реліз | — |
| 3 | Multi-timeframe (1D + 4H) | нова функція | Фаза 1, 2 |
| 4 | Hardening виконання | блокує реліз | Фаза 1, 2; no-loss частина виконується до live, решта після Phase 3 contracts |
| 5 | Спостережуваність | важливо | Фаза 4 |
| 6 | Тестове покриття | блокує реліз | Фаза 3 |

---

## Фаза 1. Критичні виправлення

Ці баги ламають продакшн-поведінку і мають бути виправлені першими.

### 1.1. Неправильна екстракція `exchange_order_id`

**Файл:** `infrastructure/execution_service.py:169`

```python
# Зараз (завжди повертає "None"):
exchange_order_id = str(order_payload.get("orderId"))

# Має бути:
exchange_order_id = str((order_payload.get("result") or {}).get("orderId", ""))
```

Bybit API повертає структуру `{ retCode, result: { orderId, ... } }`.
Через цей баг кожен запис у `order_ledger.exchange_order_id` дорівнює рядку `"None"`,
що унеможливлює відстеження ордерів через exchange.

---

### 1.2. Щоденна синхронізація качає 3 роки даних

**Файл:** `infrastructure/market_data_synchronizer.py`

```python
# Зараз (1095 днів кожного запуску):
await run_api()

# Має бути (лише останні 3 дні, з overlap):
await run_api(days=3)
```

`run_api()` без аргументів використовує `DEFAULT_LOOKBACK_DAYS=1095`.
Для щоденного оновлення достатньо 2–3 днів. Повний re-sync (1095 днів)
доцільний лише при першому запуску (`sync` / `sync-3y` команди).

---

### 1.3. Scheduler має використовувати Binance/UTC час

**Файл:** `application/scheduler.py`

```python
# Зараз (local time — непередбачувано на сервері):
now = datetime.now()

# Має бути (Binance spot kline schedule = UTC):
from datetime import timezone
now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
```

Бот має повторювати trading iteration за Binance candle schedule, тобто за UTC.
Локальний timezone сервера і `APP_TIMEZONE` не повинні впливати на 4H runtime schedule.

---

### 1.4. Відсутня ізоляція помилок між символами

**Файл:** `application/trading_cycle_service.py`

```python
# Зараз: один виняток зупиняє всі символи:
for symbol in symbols:
    results[...] = await self.run(str(symbol), dry_run=dry_run)

# Має бути: ізольований try/except з логуванням per-symbol:
for symbol in symbols:
    try:
        results[...] = await self.run(str(symbol), dry_run=dry_run)
    except Exception as exc:
        logger.error("cycle_failed symbol=%s error=%s", symbol, exc, exc_info=True)
        results[symbol.upper()] = {"error": str(exc)}
```

---

### 1.5. `_apply_position_update` відкриває окреме з'єднання

**Файл:** `infrastructure/execution_service.py`

`get_position_state` і `_apply_position_update` відкривають кожен окреме DB-з'єднання.
Position upsert і ledger insert виконуються в одній транзакції, але в окремому з'єднанні від reconciliation.
Якщо процес впаде після position upsert, але до ledger insert — стан БД буде неузгодженим.

**Рішення:** обгорнути position upsert + ledger insert в одну транзакцію через `conn.transaction()`.

---

## Фаза 2. Надійність інфраструктури

### 2.1. CVD залишається legacy-колонкою

**Файл:** `api/binance_api.py:75`

CVD (`Cumulative Volume Delta`) починається з нуля при кожному запуску sync,
тому при re-sync значення CVD в таблиці перезаписуються іншими числами.
Стратегія CVD не використовує і не планує використовувати.

**Рішення:** позначити CVD як legacy-колонку і не використовувати її в signal filters,
backtests або execution policy. Нові volume-фільтри мають використовувати звичайний
`volume`, а не CVD.

---

### 2.2. `lru_cache` на `get_symbol_filters` не враховує зміни

**Файл:** `infrastructure/bybit_spot.py:130`

```python
@lru_cache(maxsize=256)
def get_symbol_filters(self, symbol: str) -> BybitSpotFilters:
```

`lru_cache` на instance-методі враховує `self` у cache key, але сам cache wrapper живе
на рівні функції й не має TTL. Якщо Bybit змінить лімітний розмір ордера або tick_size,
бот дізнається про це лише після рестарту або створення нового процесу. Для spot-торгівлі
фільтри рідко змінюються, але кеш без TTL ризикований для довготривалих процесів.

**Рішення:** замінити на per-symbol TTL cache (наприклад 1 година) або явно інвалідований
cache в `BybitSpotClient`. `cached_property` тут не підходить напряму, бо метод приймає
аргумент `symbol`.

---

### 2.3. Reconciliation не запускається автоматично при старті

**Файл:** `application/initialization_service.py` + `application/runtime_commands.py`

`initialize_runtime` має параметр `reconcile_positions=False`, але жоден CLI-шлях його не вмикає.
Позиція синхронізується лише при першому `get_position_state` в торговому циклі.

**Рішення:** додати прапорець `--reconcile` до CLI або вмикати reconciliation завжди при live-запуску.

---

### 2.4. `MIN_PROFIT_RATIO = 0` порушує принцип no-loss

За замовчуванням продаж дозволяється при будь-якому рівні вище avg_entry,
навіть якщо прибуток менший за мінімально дозволений поріг 1%.
Тобто при `MIN_PROFIT_RATIO = 0` бот може закривати позиції нижче production no-loss порогу.

**Рішення:** встановити мінімальне значення `MIN_PROFIT_RATIO = 0.01` (1%),
що гарантує реальний прибуток понад комісії і slippage. Це перший рівень no-loss захисту.
Повна реалізація принципу — у Фазі 4.5.

---

### 2.5. Відсутній `.env.example`

Немає шаблону змінних середовища. Нові розробники або деплой-середовища не мають
референсного файлу.

**Рішення:** створити `.env.example` з усіма змінними з `utils/config.py`, їх дефолтами і коментарями.

---

## Фаза 3. Multi-Timeframe (1D + 4H)

Найбільша зміна. Це не локальне розширення, а наскрізна переробка контрактів між
`application / domain / infrastructure`, бо поточний runtime всюди очікує один `D1`
DataFrame, один сигнал і один денний scheduler-цикл.

### 3.1. Концептуальна модель

Бот має торгувати на двох таймфреймах паралельно з одною позицією на символ:

- **D1 — фільтр ринкового режиму:** якщо на `D1` є `death cross` і `ADX >= 30`, нові `BUY` на 4H не відкриваються.
- **4H — entry/exit сигнали:** генерує BUY і SELL незалежно, але з урахуванням D1-фільтра режиму.
- **Одна позиція на символ:** обидва TF впливають на ту саму `position_state`.

Приклад логіки конфлікту сигналів:

| D1 режим | 4H сигнал | Дія |
|----------|-----------|-----|
| BUY allowed | buy | → BUY |
| BUY allowed | sell | → SELL |
| BUY allowed | hold | → SKIP |
| BUY blocked (`death cross` + `ADX >= 30`) | buy | → SKIP |
| BUY blocked (`death cross` + `ADX >= 30`) | sell | → SELL |
| BUY blocked (`death cross` + `ADX >= 30`) | hold | → SKIP |

---

### 3.2. База даних: нові таблиці для 4H свічок

**Статус:** базова міграція `migrations/003_create_4h_candles.sql` вже додана.
Нижче зафіксований цільовий стан схеми й runtime-інтеграції.

```sql
-- Для кожного символу додати таблицю:
CREATE TABLE IF NOT EXISTS _candles_trading_data.<symbol>_4h (
    open_time  TIMESTAMP NOT NULL,
    close_time TIMESTAMP NOT NULL,
    symbol     TEXT      NOT NULL,
    open       NUMERIC   NOT NULL,
    close      NUMERIC   NOT NULL,
    high       NUMERIC   NOT NULL,
    low        NUMERIC   NOT NULL,
    cvd        NUMERIC   NOT NULL,
    volume     NUMERIC   NOT NULL,
    candle_id  TEXT      NOT NULL UNIQUE
);
```

**Зміни в `utils/db_actions.py`:**

```python
H4_TABLE_SUFFIX = "_4h"

def h4_table_name(symbol: str) -> str:
    return f"{symbol.lower()}{H4_TABLE_SUFFIX}"
```

`init-db` має створювати D1 і H4 candle tables через спільний шаблон
`CREATE_CANDLES_TABLE_SQL`; окремий `CREATE_4H_CANDLES_TABLE_SQL` не потрібен.
Залишилось підключити runtime sync/provider logic для заповнення й читання H4-таблиць.

---

### 3.3. Синхронізація 4H свічок з Binance

**Зміни в `api/binance_api.py`:**

```python
BINANCE_H4_INTERVAL = "4h"
H4_ANALYSIS_DAYS = 180  # ~4320 свічок, достатньо для WMA-98
```

`fetch_and_store` уже частково параметризована через `timeframe` і `table_suffix`,
але цього недостатньо для повноцінного 4H-режиму. Поточний `run_api()` і
`BinanceMarketDataSynchronizer` усе ще зашиті під D1-потік, тому тут потрібна
переробка runtime sync API, а не лише новий виклик з іншими аргументами.

**Нові CLI-команди:**
- `sync-4h [--period N]` — синхронізувати 4H свічки (дефолт: 180 днів)
- `sync-full` — синхронізувати і D1 і 4H

---

### 3.4. Multi-timeframe MarketDataProvider

**Новий клас в `infrastructure/market_data_provider.py`:**

Поточний `DatabaseMarketDataProvider` повертає лише один `D1` DataFrame через
`get_symbol_history(symbol)`. Для multi-timeframe цього недостатньо, тому тут
міняється не лише інфраструктурний клас, а й контракт, який використовує
`TradingCycleService`.

```python
class MultiTimeframeMarketDataProvider:
    """Load D1 and H4 candle history for one symbol from PostgreSQL."""

    def get_d1_history(self, symbol: str) -> pd.DataFrame: ...
    def get_h4_history(self, symbol: str) -> pd.DataFrame: ...
    def get_symbol_history(self, symbol: str) -> dict[str, pd.DataFrame]:
        return {
            "d1": self.get_d1_history(symbol),
            "h4": self.get_h4_history(symbol),
        }
```

`ANALYSIS_WINDOW` для 4H: `max(240, GREENWICH_LENGTH * 8)` = 784 свічок × 4 год = ~130 днів.
Конфігурується окремо через `H4_ANALYSIS_WINDOW` env var.

---

### 3.5. Multi-timeframe Planner

**Нова модель сигналу в `domain/models.py`:**

Поточний `GreenwichSpotPlanner` працює лише з одним набором свічок і одним
`SpotSignal`. Отже, це не точкова заміна planner-класу, а зміна доменної моделі
плану, сигналів і викликів із application-рівня.

```python
@dataclass(frozen=True)
class MultiTimeframeSignal:
    symbol: str
    d1_regime_blocked: bool
    h4: SpotSignal
    resolved: SpotSignal  # фінальний після фільтрації
```

**Новий клас в `domain/planner.py`:**

```python
class MultiTimeframeSpotPlanner:
    """Apply the D1 regime filter to H4 entry signals."""

    def plan(
        self,
        symbol: str,
        candles: dict[str, pd.DataFrame],  # {"d1": df, "h4": df}
        position_state: PositionState,
        available_quote_balance: Decimal,
    ) -> SpotTradingPlan:
        d1_regime_blocked = self._is_d1_buy_blocked(candles["d1"])
        h4_signal = generate_spot_signal(symbol, candles["h4"])
        resolved = self._resolve(d1_regime_blocked, h4_signal)
        decision = decide_spot_execution(resolved, position_state, available_quote_balance)
        return SpotTradingPlan(signal=resolved, decision=decision)

    def _resolve(self, d1_regime_blocked: bool, h4: SpotSignal) -> SpotSignal:
        # D1 death cross + ADX>=30 блокує H4 BUY
        if d1_regime_blocked and h4.signal_type == BUY_SIGNAL:
            return SpotSignal(
                h4.symbol, HOLD_SIGNAL, h4.signal_price,
                h4.close_time, "d1_regime_blocks_h4_buy"
            )
        return h4  # 4H є головним; D1 — лише фільтр режиму
```

---

### 3.6. Scheduler: ітерація раз на 4 години за Binance time

**Зміни в `application/scheduler.py`:**

Поточний scheduler повністю денний: очікування, логування, синхронізація та цикл
виконання орієнтовані на один D1-run на добу. Тому тут потрібна не лише заміна
функції очікування, а й перегляд orchestration-логіки scheduler-а та bootstrap.

Поточний `wait_until_next_daily_run` замінюється на `wait_until_next_h4_run`.
Binance candle schedule для spot klines базується на UTC, тому scheduler має
орієнтуватися саме на UTC, а не на локальний час сервера або `APP_TIMEZONE`:

```python
async def wait_until_next_h4_run() -> None:
    """Sleep until shortly after the next 4-hour UTC candle close."""
    now = datetime.now(tz=timezone.utc)
    current_block = (now.hour // 4) * 4
    next_run = now.replace(hour=current_block, minute=0, second=1, microsecond=0)
    if now >= next_run:
        next_run = next_run + timedelta(hours=4)
    await asyncio.sleep((next_run - now).total_seconds())
```

Candle close times for Binance 4H candles (UTC): 03:59:59.999, 07:59:59.999,
11:59:59.999, 15:59:59.999, 19:59:59.999, 23:59:59.999. Бот запускає ітерацію
після формування нової 4H-свічки: `00:00:01`, `04:00:01`, `08:00:01`,
`12:00:01`, `16:00:01`, `20:00:01` UTC.

D1 синхронізація залишається щоденною (запускається при H4 run о 00:00:01 UTC).

---

### 3.7. Синхронізація 4H при кожному H4-циклі

**Зміни в `infrastructure/market_data_synchronizer.py`:**

Цей крок залежить від окремої параметризації `run_api()` по timeframe і table suffix.
У поточному коді synchronizer просто викликає D1 sync без аргументів, тому зміна
тягне за собою і рефакторинг API sync-рівня.

```python
class BinanceMarketDataSynchronizer:
    async def synchronize(self, timeframes: list[str] = ("d1", "h4")) -> None:
        if "d1" in timeframes:
            await run_api(days=3, timeframe=BINANCE_D1_INTERVAL, table_suffix=D1_TABLE_SUFFIX)
        if "h4" in timeframes:
            await run_api(days=2, timeframe=BINANCE_H4_INTERVAL, table_suffix=H4_TABLE_SUFFIX)
```

---

### 3.8. Зміни конфігурації

**Нові змінні в `utils/config.py`:**

```python
BINANCE_H4_INTERVAL = "4h"
H4_TABLE_SUFFIX = "_4h"
H4_ANALYSIS_WINDOW = int(os.getenv("H4_ANALYSIS_WINDOW", str(max(240, GREENWICH_LENGTH * 8))))
H4_SCHEDULER_ENABLED = os.getenv("H4_SCHEDULER_ENABLED", "true").lower() == "true"
D1_REGIME_FILTER_ENABLED = os.getenv("D1_REGIME_FILTER_ENABLED", "true").lower() == "true"
```

---

### 3.9. Зміни CLI

**Нові subcommands в `main.py`:**

Це окрема помітна підфаза, а не дрібний хвіст після core-логіки: нові команди
потребують змін у `main.py`, `application/command_dispatcher.py`,
`application/runtime_commands.py` і в наявних unit-тестах CLI/runtime шару.

```
sync-4h [--period N]      — sync 4H candles (default 180 days)
sync-full                 — sync both D1 and 4H
analyze --timeframe 4h    — one-off D1 regime + H4 execution analysis cycle
```

---

## Фаза 4. Hardening виконання

### 4.1. Retry для Bybit API

**Файл:** `infrastructure/bybit_spot.py`

`pybit` не має вбудованого retry для transient errors.
Додати декоратор з `tenacity` (вже є в `requirements.txt`):

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((RuntimeError, ConnectionError)),
)
def place_market_order(self, symbol, side, quantity): ...
```

---

### 4.2. Перевірка статусу ордера після виставлення

**Файл:** `infrastructure/execution_service.py`

Після `place_market_order` необхідно підтвердити виконання через `get_order_history`,
оскільки market order може бути відхилений (insufficient balance, symbol suspended тощо).

```python
# Після place_market_order:
order_id = (order_payload.get("result") or {}).get("orderId")
if order_id:
    confirmed = self.client.get_order_status(symbol, order_id)
    if confirmed["status"] != "Filled":
        raise RuntimeError(f"Order {order_id} not filled: {confirmed}")
```

---

### 4.3. Graceful shutdown

**Файл:** `main.py`

```python
import signal

def _shutdown(sig, frame):
    print("⏹️ Graceful shutdown initiated")
    # Дати можливість завершити поточний цикл
    raise SystemExit(0)

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)
```

---

### 4.4. Захист від дублювання ордерів

При рестарті бота після краш-ситуації можливе повторне відкриття ордера за тим же сигналом.

**Рішення:** зберігати candle identity у `order_ledger` і перевіряти наявність виконаного
ордера за той самий `symbol + action + signal_timeframe + signal_candle_id` перед виконанням.
Одного `created_at > NOW() - INTERVAL '4 hours'` недостатньо: воно не прив'язане до
конкретної свічки й може пропустити дубль після затримки або restart поза 4H-вікном.

```sql
SELECT COUNT(*) FROM _spot_trading_bot.order_ledger
WHERE symbol = $1
  AND side = $2
  AND status = 'executed'
  AND signal_timeframe = $3
  AND signal_candle_id = $4;
```

---

### 4.5. Багаторівневий захист no-loss

Реалізація принципу «бот ніколи не продає у збиток» через три незалежних шари перевірки.
За поточним станом коду це не лише зміна policy/config: фаза зачіпає `config`,
executor, ledger schema, статуси ордерів, модель результатів виконання і механізм
нотифікацій/аудиту.

#### Шар 1 — Domain (execution policy)

**Файл:** `domain/execution.py`

Вже існує, але потребує коректного значення `MIN_PROFIT_RATIO`.
Мінімальне значення має покривати round-trip комісії + буфер на slippage:

```python
# utils/config.py
MIN_PROFIT_RATIO = Decimal(os.getenv("MIN_PROFIT_RATIO", "0.01"))  # 1% мінімум

# domain/execution.py:
min_sell_price = position_state.avg_entry_price * (Decimal("1") + MIN_PROFIT_RATIO)
if signal.signal_price < min_sell_price:
    return ExecutionDecision("skip", ..., reason="sell_price_not_profitable")
```

Цей шар відсіює сигнали ще до відправки на біржу. SELL на рівні `min_sell_price`
дозволений; усе нижче 1% профіту блокується.

---

#### Шар 2 — Infrastructure (pre-execution guard)

**Файл:** `infrastructure/execution_service.py`

Перед виставленням реального ордера зробити фінальну перевірку
з актуальним станом позиції (після reconciliation) і поточною ринковою ціною:

```python
async def _execute_trade(self, decision, position_state, *, dry_run):
    # Якщо це продаж — перевірити no-loss на рівні інфраструктури
    if decision.action == "sell" and position_state.has_position:
        current_price = self.client.fetch_current_price(decision.symbol)
        min_sell = position_state.avg_entry_price * (Decimal("1") + MIN_PROFIT_RATIO)
        if current_price < min_sell:
            logger.warning(
                "no_loss_guard_triggered symbol=%s current_price=%s min_sell=%s avg_entry=%s",
                decision.symbol, current_price, min_sell, position_state.avg_entry_price,
            )
            await self._record_ledger(decision, position_state, None, None, "blocked_no_loss")
            return ExecutionResult(
                False, decision.symbol, "skip",
                "no_loss_guard_infrastructure", decision.signal_price,
            )
    # ... далі звичайне виконання
```

Цей шар захищає від ситуації коли між генерацією сигналу і його виконанням
ціна впала нижче мінімального SELL-порогу 1% (slippage між candle close і market execution).

---

#### Шар 3 — Post-execution audit (reconciliation)

**Файл:** `infrastructure/execution_service.py` + `utils/db_actions.py`

Після кожного SELL ордера перевірити фактичну `executed_price` відносно
мінімально дозволеної ціни продажу. Якщо через slippage угода виконалась нижче
порогу 1% профіту — записати в ledger окрему подію і надіслати alert:

```python
async def _apply_position_update(self, decision, position_state, executed_price, ...):
    if decision.action == "sell":
        min_sell = position_state.avg_entry_price * (Decimal("1") + MIN_PROFIT_RATIO)
        realized_pnl = (executed_price - position_state.avg_entry_price) * decision.quantity
        if executed_price < min_sell:
            logger.error(
                "no_loss_violation symbol=%s executed_price=%s min_sell=%s avg_entry=%s pnl=%s",
                decision.symbol, executed_price, min_sell, position_state.avg_entry_price, realized_pnl,
            )
            # Відправити Telegram alert (критичний рівень)
```

Цей шар не запобігає порушенню мінімального профіту (ордер вже виконаний), але
фіксує його для аудиту.

---

#### Нові значення `status` у `order_ledger`

| Значення | Коли встановлюється |
|----------|-------------------|
| `executed` | ордер успішно виконаний |
| `skipped` | рішення `skip` (без ордера) |
| `dry_run` | dry-run режим |
| `blocked_no_loss` | заблоковано шаром 2 (pre-execution guard) |
| `no_loss_violation` | шар 3 виявив SELL нижче мінімального профіту 1% після виконання |

---

#### Нові колонки у `order_ledger`

```sql
-- Міграція 004_add_no_loss_audit.sql
ALTER TABLE _spot_trading_bot.order_ledger
    ADD COLUMN realized_pnl_usdt NUMERIC,
    ADD COLUMN realized_pnl_pct  NUMERIC,
    ADD COLUMN no_loss_check_price NUMERIC,  -- ціна на момент pre-execution guard
    ADD COLUMN signal_timeframe TEXT,
    ADD COLUMN signal_candle_id TEXT;
```

---

#### Конфігурація no-loss

```env
# .env / .env.example
MIN_PROFIT_RATIO=0.01         # мінімальний профіт для SELL над avg_entry: 1%
NO_LOSS_GUARD_ENABLED=true    # вмикає/вимикає шар 2 (pre-execution guard)
```

Змінні з `utils/config.py`:

```python
MIN_PROFIT_RATIO = Decimal(os.getenv("MIN_PROFIT_RATIO", "0.01"))
NO_LOSS_GUARD_ENABLED = os.getenv("NO_LOSS_GUARD_ENABLED", "true").lower() == "true"

# Захист від помилкової конфігурації нижче мінімального дозволеного профіту:
_MIN_ALLOWED_PROFIT_RATIO = Decimal("0.01")
if MIN_PROFIT_RATIO < _MIN_ALLOWED_PROFIT_RATIO:
    raise ValueError(
        f"MIN_PROFIT_RATIO={MIN_PROFIT_RATIO} нижчий за мінімально допустимий "
        f"{_MIN_ALLOWED_PROFIT_RATIO} (мінімальний профіт для SELL — 1%)"
    )
```

---

## Фаза 5. Спостережуваність

### 5.1. Замінити `print` на структурований `logging`

**Проблема:** весь бот використовує `print()` замість `logging`.
`LoggingSignalNotifier` правильно використовує `logger.info`, але решта файлів — `print`.

**Рішення:**
- Налаштувати root logger в `main.py` (format: JSON або structured text)
- Замінити всі `print()` на `logger.info/warning/error`
- Додати `correlation_id` (run UUID) до кожного циклу для трасування

---

### 5.2. Telegram-нотифікації для виконаних угод

**Файл:** `infrastructure/notifications.py` — розширити `LoggingSignalNotifier`
або створити `TelegramSignalNotifier`.

Поточний notifier має лише один метод `notify(signal, result)` і логування в файл/stdout.
Для Telegram цього достатньо як точки інтеграції, але треба врахувати, що частина
подій no-loss і execution audit може потребувати або розширення `ExecutionResult`,
або окремих audit-подій.

Нотифікувати при:
- `result.executed = True` (реальна угода)
- `result.dry_run = True` і `result.action != "skip"` (dry-run preview)
- Помилках виконання

`python-telegram-bot` вже є в `requirements.txt`.

---

### 5.3. PnL-трекінг у `order_ledger`

Використовувати ті самі колонки, що додаються у Фазі 4.5 для no-loss аудиту, без дублювання схеми:

```sql
ALTER TABLE _spot_trading_bot.order_ledger
    ADD COLUMN realized_pnl_usdt NUMERIC,
    ADD COLUMN realized_pnl_pct  NUMERIC,
    ADD COLUMN no_loss_check_price NUMERIC,
    ADD COLUMN signal_timeframe TEXT,
    ADD COLUMN signal_candle_id TEXT;
```

Заповнювати при SELL:
```python
realized_pnl_usdt = (executed_price - position_state.avg_entry_price) * quantity
realized_pnl_pct  = realized_pnl_usdt / position_state.total_cost * 100
```

---

### 5.4. Періодичний звіт по PnL

Після кожного торгового циклу логувати / відправляти в Telegram summary:

```
📊 Daily Report 2026-05-01
ETHUSDT: pos=0.25 ETH, avg=1800, unrealized PnL=+$45 (+2.5%)
BTCUSDT: no position
Total realized PnL today: +$120
```

---

### 5.5. Health-check endpoint

Для моніторингу живого процесу додати простий HTTP-ендпоінт:

```python
# utils/healthcheck.py
from aiohttp import web

async def health(request):
    return web.json_response({"status": "ok", "last_cycle": last_cycle_ts})

app = web.Application()
app.router.add_get("/health", health)
```

Запускати як background task поряд з основним scheduler.

---

## Фаза 6. Тестове покриття

### 6.1. Виправити тести, що залежать від pandas/asyncpg

Це не лише питання залежностей. Після фаз 3–5 зміняться контракти planner/runtime/
scheduler/execution шарів, тому тут потрібне не лише відновлення оточення, а й
перебудова частини test matrix під нову архітектуру.

Встановити залежності в тестовому середовищі і запустити:
- `test_planner.py` (pandas)
- `test_greenwich_signals.py` (pandas)

Переконатись, що весь поточний тестовий набір проходить (`python -m pytest tests/ -v`).

---

### 6.2. Нові unit-тести для multi-timeframe логіки

**`tests/test_multitimeframe_planner.py`:**
- D1 `death cross` + `ADX >= 30` блокує H4 buy → resolved signal = hold
- D1 BUY allowed + H4 buy → resolved = buy
- D1 BUY blocked + H4 sell → resolved = sell
- D1 BUY allowed + H4 hold → resolved = hold

---

### 6.3. Тести для фіксованих багів

- `test_exchange_order_id_extraction` — перевірити що `orderId` береться з `result`
- `test_synchronizer_calls_with_limited_days` — перевірити що sync передає days=3
- `test_scheduler_uses_utc` — перевірити що scheduler не використовує local time
- `test_run_many_isolates_symbol_errors` — перевірити що виняток одного символу не зупиняє інші

---

### 6.4. Тести no-loss гарантій

**`tests/test_no_loss.py`:**

```python
# Шар 1: domain guard
def test_sell_blocked_when_price_below_min_profit_ratio():
    signal = SpotSignal("ETHUSDT", "sell", Decimal("100"), ...)
    state = PositionState("ETHUSDT", Decimal("1"), Decimal("100"), Decimal("100"))
    # MIN_PROFIT_RATIO=0.01 → min_sell = 101.00
    decision = decide_spot_execution(signal, state, Decimal("500"))
    assert decision.action == "skip"
    assert decision.reason == "sell_price_not_profitable"

# Шар 1: продаж дозволено на рівні мінімального порогу 1%
def test_sell_allowed_when_price_reaches_min_profit_ratio():
    signal = SpotSignal("ETHUSDT", "sell", Decimal("101"), ...)
    state = PositionState("ETHUSDT", Decimal("1"), Decimal("100"), Decimal("100"))
    decision = decide_spot_execution(signal, state, Decimal("500"))
    assert decision.action == "sell"

# Шар 2: pre-execution guard блокує при падінні між сигналом і виконанням
def test_no_loss_guard_blocks_sell_when_market_price_dropped():
    # signal_price=101 (проходить шар 1), але поточна ціна=99 (нижче 1% SELL-порогу)
    ...
    assert result.action == "skip"
    assert result.reason == "no_loss_guard_infrastructure"

# Конфігурація: MIN_PROFIT_RATIO нижче 1% не дозволяється
def test_min_profit_ratio_below_minimum_raises_on_startup():
    with pytest.raises(ValueError, match="нижчий за мінімально допустимий"):
        # Симулювати завантаження config з MIN_PROFIT_RATIO=0.001
        ...

# Шар 2: статус в ledger при блокуванні
def test_blocked_no_loss_status_recorded_in_ledger():
    ...
    assert ledger_entry["status"] == "blocked_no_loss"
```

---

### 6.5. Integration-тести з реальною БД

**`tests/integration/test_db_round_trip.py`** (вимагає PostgreSQL):
- Create tables → insert candle → fetch via `DatabaseMarketDataProvider`
- Execute trade → read `position_state` → confirm reconciliation

Запускати в CI через PostgreSQL service container. Це окремий bootstrap-крок:
у репозиторії наразі немає готового CI workflow, тому під цей етап треба буде
додати саму CI-обв'язку, а не лише нові integration-тести.

---

### 6.6. E2E dry-run тест

**`tests/e2e/test_full_dry_run.py`:**

```python
# Запускає повний цикл з реальними DB і Mock Bybit:
await trading_cycle.run("ETHUSDT", dry_run=True)
# Перевіряє order_ledger на запис з status="dry_run"
```

---

## Checklist релізу

### Блокери (must-fix перед першим prod-запуском)

- [ ] 1.1 Виправлено `exchange_order_id` extraction
- [ ] 1.2 Синхронізація передає `days=3` замість 1095
- [ ] 1.3 Scheduler використовує UTC
- [ ] 1.4 `run_many` ізолює помилки символів
- [ ] 1.5 Position upsert + ledger в одній транзакції
- [ ] 2.4 `MIN_PROFIT_RATIO` = 0.01 (1%) за замовчуванням
- [ ] 2.5 Файл `.env.example` присутній
- [ ] 4.1 Retry для Bybit API calls
- [ ] 4.3 Graceful shutdown
- [ ] 6.1 Всі існуючі тести проходять

### No-Loss (must-have перед першим prod-запуском)

- [ ] 4.5 Шар 1: `MIN_PROFIT_RATIO=0.01` як захищений дефолт з валідацією при старті
- [ ] 4.5 Шар 2: pre-execution guard у `BybitSpotExecutor._execute_trade` з fetch поточної ціни
- [ ] 4.5 Шар 3: post-execution audit — логування та alert при SELL нижче мінімального профіту 1%
- [ ] 4.5 Нові статуси `blocked_no_loss` / `no_loss_violation` у `order_ledger`
- [ ] 4.5 Міграція 004: колонки `realized_pnl_usdt`, `realized_pnl_pct`, `no_loss_check_price`, `signal_timeframe`, `signal_candle_id`
- [ ] 4.5 `NO_LOSS_GUARD_ENABLED` env var задокументовано у `.env.example`
- [ ] 6.4 Unit-тести всіх трьох шарів no-loss (домен, інфраструктура, конфігурація)

### Multi-timeframe (нова функція)

- [x] 3.2 Базова міграція 003 для H4-таблиць
- [ ] 3.2 Runtime wiring для створення/читання H4-таблиць у live flow
- [ ] 3.3 `sync-4h` команда
- [ ] 3.4 `MultiTimeframeMarketDataProvider`
- [ ] 3.5 `MultiTimeframeSpotPlanner` з D1 regime filter
- [ ] 3.6 H4 scheduler (6 разів на день)
- [ ] 3.7 Синхронізатор параметризований по timeframe
- [ ] 3.8 Нові env-змінні задокументовані
- [ ] 6.2 Unit-тести multi-timeframe плану
- [ ] 6.3 Тести для фіксованих багів

### Операційна готовність

- [ ] 5.1 `print` замінено на `logging`
- [ ] 5.2 Telegram-нотифікації для угод (включаючи `blocked_no_loss` alerts)
- [ ] 5.3 PnL-колонки в `order_ledger`
- [ ] 6.5 Integration-тести з реальною БД
- [ ] 6.6 E2E dry-run тест

---

## Порядок виконання (рекомендований)

```
Тиждень 1: Фаза 1 (баги) + Фаза 2.4, 2.5
Тиждень 2: Фаза 4.5 no-loss subset (3 шари + міграція 004 + тести 6.4)
Тиждень 3: Фаза 3.2–3.3 (DB + runtime sync API для 4H)
Тиждень 4: Фаза 3.4–3.6 (provider + planner + scheduler contracts)
Тиждень 5: Фаза 3.7–3.9 (synchronizer + CLI + config) + Фаза 6.2–6.3
Тиждень 6: Фаза 4.1–4.4 (execution hardening) + Фаза 6.1
Тиждень 7: Фаза 5 (observability) + Фаза 6.5–6.6 + фінальне QA на demo-акаунті
```
