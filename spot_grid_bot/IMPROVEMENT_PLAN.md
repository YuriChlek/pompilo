# Spot Grid Bot — План покращень

> Версія: 2026-05-01  
> Статус: чернетка для обговорення

---

## Зміст

- [P0 — Критична безпека](#p0--критична-безпека)
- [P1 — Висока пріоритетність](#p1--висока-пріоритетність)
- [P2 — Середня пріоритетність](#p2--середня-пріоритетність)
- [P3 — Довгострокові покращення](#p3--довгострокові-покращення)
- [Зведена таблиця](#зведена-таблиця)

---

## P0 — Критична безпека

Ці проблеми впливають на збереження капіталу або можуть призвести до втрати стану бота. Виправити до наступного live-запуску.

---

### P0-0. Cost basis рахується з execution history замість біржового `avgPrice` — хибне значення при довгих позиціях

**Файли:**
- `infrastructure/bybit_account_client.py` *(основна зміна)*
- `infrastructure/execution_gateway.py`
- `infrastructure/state_store.py` *(залежить від P0-2 і P1-6, див. примітку нижче)*
- `domain/runtime_models.py`
- `domain/spot_grid_planner.py`
- `infrastructure/market_data_provider.py`
- `application/analysis_batch_service.py`

**Проблема — поточна архітектура.**
Бот зараз визначає середню ціну купівлі активу через `CostBasisResolver`, який запитує останні **100** виконань з Bybit і рахує власний weighted average:
```python
# infrastructure/execution_gateway.py:384
response = self.client.get_executions(category=SPOT_CATEGORY, symbol=symbol, limit=100)
```

У `calculate_cost_basis_from_executions` є критична вразливість — якщо 100 записів не охоплюють всю поточну позицію, функція повертає середнє тільки тих fills, що потрапили у вибірку, **без жодного попередження**:
```python
if running_qty > actual_balance:
    ...  # коригується
# якщо running_qty < actual_balance — нічого не відбувається
return float(running_cost / running_qty)  # хибно занижена ціна
```

**Небезпечний сценарій:**
1. Куплено 0.5 ETH по $2000 три місяці тому.
2. З того часу — 200+ grid trades навколо $1580–1620.
3. Bybit повертає останні 100 — там лише нещодавні trades.
4. `running_qty ≈ 0.12 ETH`, але `actual_balance = 0.5 ETH`.
5. Функція повертає `cost_basis ≈ $1600` (середнє лише нещодавніх).
6. `minimum_exit_price = $1600 × 1.01 = $1616`.
7. Бот ставить SELL @ $1620 — вважає це no-loss.
8. Реальна середня собівартість ≈ $1870.
9. **Sell виконується зі збитком ~$250 на ETH** — no-loss правило повністю порушено.

**Правильне рішення — два рівні джерел.**

**Рівень 1 (primary): `avgPrice` з Bybit wallet balance.**
Bybit сам відстежує weighted average price поточної спотової позиції і повертає її у відповіді `GET /v5/account/wallet-balance` у полі `avgPrice` кожного coin-об'єкта. Це завжди точна ціна незалежно від терміну позиції і без обмеження на кількість виконань.

Поточний код у `bybit_account_client.py` отримує wallet balance, але **не бере** `avgPrice`:
```python
base_coin = coins.get(base_asset, {})
base_balance = float(base_coin.get("walletBalance") or 0.0)
# avgPrice тут доступний, але не використовується
cost_basis_price=self._resolve_cost_basis_price(symbol, base_balance),  # ← хибне джерело
```

**Рівень 2 (fallback): персистований `cost_basis_price` у PostgreSQL.**
Коли Bybit повертає `avgPrice = None` або `0` (наприклад, позиція щойно закрита, API недоступний) — використовується значення, збережене з попереднього циклу.

---

**Покрокові зміни:**

**Крок 1. `domain/runtime_models.py` — додати поле до `SymbolRuntimeState`.**
```python
@dataclass(slots=True)
class SymbolRuntimeState:
    symbol: str
    strategy_state: StrategyState
    risk_state: RiskRuntimeState = field(default_factory=RiskRuntimeState)
    cost_basis_price: float | None = None   # ← нове поле
```

**Крок 2. `infrastructure/state_store.py` — нова колонка.**
> ⚠️ Ця зміна залежить від P0-2 (fix `conn = None`) і P1-6 (connection pool). Рекомендується реалізувати всі три P0 за один PR, щоб не вносити колонку через сирий патерн.

```sql
-- у CREATE TABLE:
cost_basis_price DOUBLE PRECISION NULL
```
Додати у `SELECT`, `INSERT ... ON CONFLICT DO UPDATE`. При завантаженні:
```python
cost_basis_price=float(row["cost_basis_price"]) if row["cost_basis_price"] is not None else None,
```

**Крок 3. `infrastructure/bybit_account_client.py` — використати `avgPrice` як primary.**
Змінити `__init__`: прибрати параметр `resolve_cost_basis_price` (більше не потрібен).
Змінити `get_balances`: додати параметр `persisted_cost_basis: float | None = None`.
```python
def get_balances(self, symbol: str, *, persisted_cost_basis: float | None = None) -> InventorySnapshot:
    ...
    base_balance = float(base_coin.get("walletBalance") or 0.0)

    # Primary: avgPrice з Bybit — завжди точна, незалежно від довжини позиції
    avg_price_raw = base_coin.get("avgPrice")
    cost_basis_price: float | None = None
    if avg_price_raw is not None:
        avg = float(avg_price_raw)
        if avg > 0 and base_balance > 0:
            cost_basis_price = avg

    # Fallback: значення з попереднього циклу (PostgreSQL)
    if cost_basis_price is None and persisted_cost_basis is not None and persisted_cost_basis > 0:
        cost_basis_price = persisted_cost_basis
        logger.debug("cost_basis_fallback_to_persisted symbol=%s persisted=%.4f", symbol, persisted_cost_basis)

    if cost_basis_price is None and base_balance > 0:
        logger.warning(
            "cost_basis_unavailable symbol=%s base_balance=%.6f — sells will be blocked",
            symbol, base_balance,
        )

    return InventorySnapshot(
        base_balance=base_balance,
        quote_balance=float(quote_coin.get("walletBalance") or 0.0),
        reserved_quote=float(quote_coin.get("locked") or 0.0),
        mark_price=mark_price,
        cost_basis_price=cost_basis_price,
    )
```

**Крок 4. `infrastructure/execution_gateway.py` — прибрати `CostBasisResolver` з основного потоку.**
- Прибрати `self._cost_basis_resolver = CostBasisResolver(...)` з `BybitSpotExchange.__init__`.
- Прибрати callback `resolve_cost_basis_price` з `BybitSpotAccountClient(...)`.
- `BybitSpotExchange.get_balances` передає `persisted_cost_basis`:
```python
def get_balances(self, symbol: str, *, persisted_cost_basis: float | None = None) -> InventorySnapshot:
    self._ensure_symbol_supported(symbol)
    return self._account_client.get_balances(symbol, persisted_cost_basis=persisted_cost_basis)
```
- Метод `resolve_cost_basis_price` і `_fetch_executions` залишити як діагностичні утиліти, але не викликати в основному потоці. `cost_basis_resolver.py` не видаляти — може знадобитись для ручної діагностики.

**Крок 5. `domain/spot_grid_planner.py` — методи для роботи з персистованим значенням.**
```python
def get_persisted_cost_basis(self, symbol: str) -> float | None:
    """Return the last known cost basis for one symbol from in-memory runtime."""
    runtime = self._runtime_by_symbol.get(symbol.upper())
    return runtime.cost_basis_price if runtime is not None else None

def update_cost_basis(self, symbol: str, cost_basis_price: float) -> None:
    """Store a fresh cost basis received from the exchange into runtime state."""
    runtime = self._ensure_runtime(symbol)
    runtime.cost_basis_price = cost_basis_price
```
Оновити `export_symbol_runtime` — включити `cost_basis_price`:
```python
return SymbolRuntimeState(
    symbol=symbol.upper(),
    strategy_state=replace(runtime.strategy_state),
    risk_state=RiskRuntimeState(...),
    cost_basis_price=runtime.cost_basis_price,   # ← нове
)
```

**Крок 6. `infrastructure/market_data_provider.py` — прийняти і передати persisted fallback.**
```python
async def get_market_context(self, symbol: str, *, persisted_cost_basis: float | None = None) -> MarketContext:
    candles = await self.repository.fetch_recent_candles(...)
    inventory = self.exchange.get_balances(symbol, persisted_cost_basis=persisted_cost_basis)
    if inventory.mark_price <= 0:
        inventory.mark_price = candles[-1].close
    ...
```
> Оновити протокол `MarketDataProvider` у `application/ports.py` відповідно.

**Крок 7. `application/analysis_batch_service.py` — оркестрація між planner і provider.**
```python
for symbol in [...]:
    try:
        persisted_cost_basis = self.planner.get_persisted_cost_basis(symbol)
        context = await self.market_data_provider.get_market_context(
            symbol, persisted_cost_basis=persisted_cost_basis
        )
        # Якщо отримали свіже значення від Bybit — зберегти в runtime
        if context.inventory.cost_basis_price is not None:
            self.planner.update_cost_basis(symbol, context.inventory.cost_basis_price)
        analysis = self.planner.analyze(context)
    except Exception:
        ...
```

**Крок 8. `filter_no_loss_sells` і `_is_sell_order_allowed` — логування при блокуванні.**
```python
if min_allowed_price is None:
    logger.warning(
        "no_loss_sell_blocked_no_cost_basis symbol=%s base_balance=%.6f mark_price=%s",
        symbol, inventory.base_balance, inventory.mark_price,
    )
    return [order for order in target_orders if order.side != OrderSide.SELL]
```

**Тести:**
- `avgPrice = "1850.0"` у wallet response → `inventory.cost_basis_price = 1850.0`, `persisted_cost_basis` не використовується.
- `avgPrice = None`, `persisted_cost_basis = 1850.0` → `inventory.cost_basis_price = 1850.0` (fallback).
- `avgPrice = None`, `persisted_cost_basis = None`, `base_balance > 0` → логується warning, всі sells заблоковані.
- Sell @ $1620 при `cost_basis = $1870` → заблокований в `_is_sell_order_allowed`.
- Цикл зберігає `cost_basis_price` у PostgreSQL → наступний цикл відновлює через `restore_symbol_runtime`.

**Що відбувається з `cost_basis_resolver.py`:**
Файл залишається у кодовій базі як діагностичний інструмент, але `CostBasisResolver` більше не використовується в основному торговому потоці. Функція `calculate_cost_basis_from_executions` може використовуватись вручну для звірки.

---

### P0-1. Порядок guardrails може скидати валідні sell-ордери

**Файл:** `infrastructure/execution_guardrails.py`, рядки 20–23

**Проблема.**
Поточний порядок фільтрів:
```python
filtered_orders = filter_marketable_orders(filtered_orders, ...)   # 1
filtered_orders = dedupe_close_levels(filtered_orders, ...)         # 2
filtered_orders = filter_no_loss_sells(filtered_orders, ...)        # 3
```
`dedupe_close_levels` зберігає один з двох близьких ордерів (рядок 79: за розміром, не за ціною). Якщо виживає ордер із нижчою ціною — `filter_no_loss_sells` потім відкидає його. В результаті symbol втрачає sell-рівень, який мав бути дійсним. Крім того, `filter_marketable_orders` вже видаляє sells надто близько до ринку (рядок 120–128), але не перевіряє no-loss floor — тобто дві незалежні sell-фільтрації застосовуються в неузгодженому порядку.

**Правильний порядок:**
```
no_loss_sells → marketable_orders → dedupe_close_levels → throttle
```
1. Спочатку відрізати все, що порушує no-loss (домейн-обмеження).
2. Потім відрізати все, що занадто близько до ринку (safety).
3. Потім злити рівні, що залишились (cleanup).
4. Потім throttle за `max_new_orders` і `max_total`.

**Зміна:**
```python
# infrastructure/execution_guardrails.py, рядки 20–23
filtered_orders = filter_no_loss_sells(filtered_orders, inventory, strategy_config)
filtered_orders = filter_marketable_orders(filtered_orders, inventory, strategy_config)
filtered_orders = dedupe_close_levels(filtered_orders, strategy_config.execution.min_level_distance_bps)
```

**Тест:** додати кейс, де два sell-рівні близько до cost basis → після dedupe залишається рівень вище no-loss floor.

---

### P0-2. `state_store.py` — `AttributeError` маскує реальну причину збою БД

**Файл:** `infrastructure/state_store.py`, рядки 15–36, 40–61, 90–133

**Проблема.**
У всіх трьох методах (`initialize`, `load_symbol_state`, `save_symbol_state`) патерн:
```python
conn = await create_connection()   # якщо кидає — conn не призначений
try:
    ...
finally:
    await conn.close()  # AttributeError: 'coroutine' object has no attribute 'close'
```
Якщо PostgreSQL недоступний або пул переповнений, оператор бачить `AttributeError` замість `ConnectionRefusedError`. Реальна причина маскується.

**Зміна:**
```python
conn = None
try:
    conn = await create_connection()
    ...
finally:
    if conn is not None:
        await conn.close()
```
Застосувати у всіх трьох методах класу `PostgresStateStore`.

**Тест:** передати невалідні DB_HOST/DB_PORT → переконатись, що логується реальний `ConnectionRefusedError`.

---

### P0-3. `scheduler.py` — необроблений виняток зупиняє всі символи

**Файл:** `application/scheduler.py`, рядки 60–69

**Проблема.**
Головний цикл:
```python
while True:
    await wait_until_next_run(...)
    if self.market_data_synchronizer is not None:
        await self.market_data_synchronizer.synchronize()
    await self.trading_cycle.run_many(symbol_list)   # якщо кидає — процес падає
```
Будь-який необроблений виняток (мережевий таймаут, помилка Bybit API, OOM) зупиняє весь scheduler. Усі символи залишаються без обробки до ручного рестарту. Оператор дізнається про проблему тільки якщо дивиться на логи.

**Зміна:**
```python
while True:
    await wait_until_next_run(target_minute=target_minute, target_second=target_second)
    iteration += 1
    logger.info("scheduler_cycle_started iteration=%s symbols=%s", iteration, len(symbol_list))
    try:
        if self.market_data_synchronizer is not None:
            await self.market_data_synchronizer.synchronize()
        await self.trading_cycle.run_many(symbol_list)
    except Exception:
        logger.exception("scheduler_cycle_failed iteration=%s — продовжуємо наступний цикл", iteration)
    finally:
        logger.info("scheduler_cycle_finished iteration=%s", iteration)
```
Окремо: synchronize() і run_many() варто обернути в окремі try/except, щоб збій синхронізації свічок не блокував торговий цикл.

**Тест:** `market_data_synchronizer.synchronize()` кидає `RuntimeError` → scheduler виживає і запускає наступну ітерацію.

---

## P1 — Висока пріоритетність

Ці проблеми впливають на надійність роботи і якість торгових рішень.

---

### P1-1. `state_machine.py` — мутація стану замість іммутабельних переходів

**Файл:** `domain/state_machine.py`, рядки 12–46

**Проблема.**
`on_bar` мутує `self.state` in-place і одночасно повертає його:
```python
def on_bar(self, regime_snapshot) -> StrategyState:
    state = self.state          # alias, не копія
    state.bars_in_state += 1   # мутація
    ...
    return state                # повертає той самий об'єкт
```
У `symbol_analyzer.py` (рядок 39) аналіз повинен бути **non-committing** і тому робить `replace(runtime.strategy_state)` перед передачею у `StrategyStateMachine`. Але `StrategyState` — звичайний `@dataclass` (не `frozen`), тому `replace()` на Python dataclass — неглибока копія. Якщо `on_bar` мутує підполя (зараз не мутує, але архітектурно ніщо не захищає від цього) — аналіз стане committing. Семантика плутає майбутніх розробників і є джерелом прихованих регресій при рефакторингу.

**Зміна:**
1. Зробити `StrategyState` frozen dataclass (або додати `__post_init__` валідацію).
2. Переписати `on_bar` щоб повертати **новий** об'єкт через `dataclasses.replace`:

```python
def on_bar(self, regime_snapshot: RegimeSnapshot) -> StrategyState:
    state = self.state
    updated = replace(
        state,
        bars_in_state=state.bars_in_state + 1,
        cooldown_remaining=max(state.cooldown_remaining - 1, 0),
        volatility_cooldown_remaining=max(state.volatility_cooldown_remaining - 1, 0),
    )
    # ... решта логіки через replace(updated, ...)
    self.state = updated
    return updated
```

**Тест:** переконатись що `analyze()` не змінює `runtime.strategy_state` після виклику.

---

### P1-2. `order_diff.py` — нестабільний price matching для дешевих символів

**Файл:** `domain/order_diff.py`, рядки 51–54

**Проблема.**
```python
def _price_matches(live_price, target_price, price_diff_bps):
    reference = max(abs(live_price), abs(target_price), 1e-9)
    diff_bps = abs(live_price - target_price) / reference * 10_000
    return diff_bps <= price_diff_bps
```
Для DOGEUSDT ($0.18) при `price_diff_bps=5`:  
допуск = `0.18 * 5 / 10_000 = $0.000009` — менше ніж tick_size ($0.0001).  
Це означає, що ордер на $0.1800 і $0.1801 (різниця 1 tick) не матчиться, хоча це один і той самий рівень з точки зору біржі. Результат: зайві rebuild-и через хибний `diff_count > 0`.

**Зміна.**
Передати `tick_size` у `target_orders_diff_count` і враховувати його у `_price_matches`:
```python
def _price_matches(live_price, target_price, price_diff_bps, *, tick_size=0.0):
    reference = max(abs(live_price), abs(target_price), tick_size, 1e-9)
    diff_bps = abs(live_price - target_price) / reference * 10_000
    return diff_bps <= price_diff_bps
```
Відповідно передавати `tick_size` із `SpotGridPlanner.plan_from_analysis` → `target_orders_diff_count`.

**Тест:** для DOGE ціни $0.1800/$0.1801 при tick=0.0001 → `diff_count == 0`.

---

### P1-3. `rebuild_policy.py` — фіксований ціновий поріг без урахування волатильності

**Файл:** `domain/rebuild_policy.py`, рядки 44–47

**Проблема.**
```python
deviation = abs(price - state.last_rebuild_price) / state.last_rebuild_price
if deviation >= rebuild_price_deviation_pct:  # завжди 0.3%
```
Для BTC ($100k) 0.3% = $300 — це нормально. Для XRP ($0.60) 0.3% = $0.0018, що менше за 2 tick-и. Результат: XRP/DOGE ребілдяться майже щогодини просто через шум, BTC може не ребілдитись навіть при значному русі.

**Зміна.**
Додати в `should_rebuild` параметр `atr14` і обчислювати adaptive threshold:
```python
# Мінімальний поріг: більше з config і 20% від нормалізованого ATR
atr_threshold = (atr14 / price) * 0.20 if price > 0 else 0.0
effective_threshold = max(rebuild_price_deviation_pct, atr_threshold)
if deviation >= effective_threshold:
    reasons.append("price_deviation")
    return True, reasons
```
`atr14` вже доступний у `SpotGridPlanner.plan_from_analysis` через `analysis.indicators`.

**Тест:** для volatile символу (ATR = 5% ціни) threshold повинен бути щонайменше 1%, а не 0.3%.

---

### P1-4. `uptrend_policy.py` — порівняння рядків замість enum

**Файл:** `domain/uptrend_policy.py` (функція `limit_buy_levels`)

**Проблема.**
```python
buy_levels = [level for level in grid.levels if level.side.value == "BUY"]
```
Порівнюється `.value` рядкового представлення enum замість самого enum. Якщо значення `OrderSide.BUY` зміниться (наприклад, рефакторинг на `"buy"`) — фільтрація тихо зламається і бот перестане обмежувати buy-рівні при underwater recovery.

**Зміна:**
```python
buy_levels = [level for level in grid.levels if level.side == OrderSide.BUY]
sell_levels = [level for level in grid.levels if level.side == OrderSide.SELL]
```
Перевірити всі аналогічні місця у `range_entry_policy.py` і `de_risk.py`.

---

### P1-5. Відсутні реальні сповіщення про критичні події

**Файл:** `infrastructure/notifications.py`, `application/ports.py`

**Проблема.**
`LoggingSignalNotifier` тільки пише в лог:
```python
class LoggingSignalNotifier:
    async def notify(self, symbol, decision):
        logger.info("signal symbol=%s regime=%s rebuild=%s", ...)
```
При спрацюванні kill switch, переході в PANIC de-risk, або daily drawdown pause — оператор дізнається про це тільки якщо моніторить логи. Якщо бот працює без нагляду (live mode ночами) — капітал може знаходитись у PANIC режимі кілька годин без реакції.

**Зміна.**
Реалізувати `TelegramSignalNotifier` (або webhook) як окремий клас, що імплементує `SignalNotifier`:
- Надсилати повідомлення при: `rebuild_required=True` + `de_risk_mode in {HARD, PANIC}`, `force_risk_off=True`, `kill_switch_count` збільшився, `daily_drawdown_pause` активний.
- Додати `TELEGRAM_BOT_TOKEN` і `TELEGRAM_CHAT_ID` до `.env` / `utils/config.py`.
- В `bootstrap.py` будувати `TelegramSignalNotifier` якщо токен заданий, інакше fallback на `LoggingSignalNotifier`.

Бібліотека `python-telegram-bot` вже у `requirements.txt`.

---

### P1-6. `state_store.py` — нове підключення на кожну операцію

**Файл:** `infrastructure/state_store.py`

**Проблема.**
Кожен `load_symbol_state` і `save_symbol_state` відкриває і закриває окреме asyncpg-підключення. При 10 символах і кожному торговому циклі — 20 connect/disconnect операцій. Це:
- Збільшує латентність циклу на ~50–200 мс.
- Збільшує навантаження на PostgreSQL (handshake, auth).

`infrastructure/db.py` вже має `get_db_pool()`, але `PostgresStateStore` не використовує його.

**Зміна.**
Впровадити пул підключень у `PostgresStateStore`:
```python
class PostgresStateStore:
    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await get_db_pool()
        return self._pool

    async def load_symbol_state(self, symbol: str) -> SymbolRuntimeState | None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(...)
        ...
```
`async with pool.acquire()` автоматично повертає з'єднання у пул — без try/finally.

---

### P1-7. `analysis_batch_service.py` — надто широкий `except Exception`

**Файл:** `application/analysis_batch_service.py`, рядок 32

**Проблема.**
```python
except Exception:
    logger.exception("trading_cycle_failed symbol=%s phase=analysis", symbol)
    results[symbol] = None
    continue
```
`Exception` не перехоплює `KeyboardInterrupt` і `SystemExit` (вони `BaseException`), але перехоплює `MemoryError`, `RecursionError`, та інші фатальні помилки. Інфраструктурна проблема (OOM, корупція heap) маскується як "символ не вдалось проаналізувати".

**Зміна.**
Замінити на конкретніші типи або логувати severity:
```python
except Exception as exc:
    is_infra_error = isinstance(exc, (OSError, MemoryError, asyncpg.PostgresError))
    log_fn = logger.critical if is_infra_error else logger.exception
    log_fn("trading_cycle_failed symbol=%s phase=analysis error_type=%s", symbol, type(exc).__name__)
    results[symbol] = None
    continue
```
Аналогічно переглянути `trading_cycle_service.py` рядок 64–66.

---

## P2 — Середня пріоритетність

Покращують якість торгових рішень і зручність операційного обслуговування.

---

### P2-1. `range_entry_policy.py` — breakdown signal не розрізняє напрям руху

**Файл:** `domain/range_entry_policy.py`, функція `evaluate_range_entry_profile`

**Проблема.**
`directional_move` в `indicators.py` рахується як `abs(close[-1] - close[-lookback])`. Тому різкий рух вгору і різкий рух вниз дають однакове значення. В `range_entry_policy.py` це значення використовується для оцінки breakdown:
```python
indicators.directional_move >= config.regime.range_directional_threshold * ...
```
Сильний upswing у RANGE (ціна йде вгору) помилково тригерить "breakdown score" і пригальмовує нові BUY — тобто бот пропускає саме той момент, коли мав би накопичувати на pullback.

**Зміна.**
Додати в `IndicatorSnapshot` поле `directional_sign: float` (від -1.0 до +1.0) — відношення кінцевого руху до його абсолютної величини. У `range_entry_policy.py` враховувати знак:
```python
# breakdown — тільки при русі вниз
is_downside_breakout = (
    indicators.directional_move >= threshold
    and indicators.directional_sign < 0
)
```

---

### P2-2. Backtest не моделює slippage та комісію на fills

**Файл:** `backtesting/engine.py`

**Проблема.**
Fills симулюються за точною `target_price` без spread і slippage. Для малоліквідних символів (SUI, DOGE) реальний fill може бути на 1–5 bps гірше. Оскільки bot має maker_fee_bps=10 у конфізі, але backtest engine не застосовує fee до PnL — результати завищені систематично.

**Зміна.**
Додати в `BacktestConfig` (або `StrategyConfig.execution`) параметри:
- `simulated_slippage_bps: float = 2.0`
- `apply_maker_fee: bool = True`

Застосовувати при симуляції fill:
```python
effective_buy_price = fill_price * (1 + slippage_bps / 10_000 + maker_fee_bps / 10_000)
effective_sell_price = fill_price * (1 - slippage_bps / 10_000 - maker_fee_bps / 10_000)
```

---

### P2-3. Перевірка coherence конфігурації на старті

**Файл:** `application/bootstrap.py` або `domain/strategy_config.py`

**Проблема.**
Три незалежні обмеження для symbol inventory:
- `max_symbol_inventory_pct_of_equity * equity`
- `max_symbol_notional_cap`
- `max_inventory_notional`

Беруться через `min()`. Якщо `max_inventory_notional=$500`, а `max_symbol_notional_cap=$400` — перший cap завжди перемагає і `max_inventory_notional` — мертвий код. Оператор може помилково вважати, що `max_inventory_notional` активний.

**Зміна.**
Додати `validate_strategy_config(config: StrategyConfig)` що викликається при старті:
```python
def validate_strategy_config(config: StrategyConfig) -> None:
    risk = config.risk
    if risk.max_inventory_notional < risk.max_symbol_notional_cap:
        logger.warning(
            "config_warn: max_symbol_notional_cap=%.0f > max_inventory_notional=%.0f — "
            "max_inventory_notional ніколи не буде активним",
            risk.max_symbol_notional_cap, risk.max_inventory_notional,
        )
    if risk.min_symbol_entry_notional >= risk.max_symbol_notional_cap:
        raise ValueError("min_symbol_entry_notional >= max_symbol_notional_cap — нові ордери неможливі")
```

---

### P2-4. `rebuild_policy.py` — `diff_count > 0` тригерить rebuild навіть при 1 рівні різниці

**Файл:** `domain/rebuild_policy.py`, рядок 41–43

**Проблема.**
```python
if diff_count > 0:
    reasons.append(f"target_diff={diff_count}")
    return True, reasons
```
Навіть 1 неспівпадіння (1 fill виконався, grid змістився на 1 рівень) призводить до повного rebuild. При активному ринку це може давати rebuild майже щогодини навіть без реального drift.

**Зміна.**
Ввести `diff_count_threshold` (наприклад, `2`) щоб ігнорувати мінімальні відхилення:
```python
diff_threshold = max(1, len(target_orders) // 4)  # 25% рівнів або мінімум 1
if diff_count > diff_threshold:
    reasons.append(f"target_diff={diff_count}")
    return True, reasons
```
Або додати окремий параметр у `ExecutionConfig.rebuild_diff_threshold: int = 2`.

---

### P2-5. `cost_basis.py` — логування коли no-loss floor відкидає sell

**Файл:** `domain/target_order_builder.py`, рядок 172

**Проблема.**
```python
if level.side == OrderSide.SELL and not _is_sell_order_allowed(inventory, target_price, config):
    continue
```
Sell відкидається мовчки. Оператор не знає:
- Скільки sell-рівнів заблоковано.
- На якому рівні cost basis знаходиться inventory.
- Коли очікується, що no-loss floor опуститься нижче ринкової ціни.

**Зміна.**
Додати `logger.debug` або `logger.info` при першому blocked sell:
```python
blocked_sell_count = 0
for level in grid.levels:
    ...
    if level.side == OrderSide.SELL and not _is_sell_order_allowed(inventory, target_price, config):
        blocked_sell_count += 1
        continue
    ...

if blocked_sell_count > 0:
    logger.info(
        "no_loss_sell_blocked symbol=%s blocked_levels=%s min_exit_price=%s mark_price=%s cost_basis=%s",
        symbol.upper(), blocked_sell_count,
        minimum_exit_price(inventory, config),
        inventory.mark_price,
        inventory.cost_basis_price,
    )
```

---

### P2-6. `grid_viability.py` — втрата `client_order_id` при merge

**Файл:** `domain/grid_viability.py`

**Проблема.**
Коли два ордери нормалізуються до однієї tick-price, один з них відкидається і `client_order_id` другого теж губиться. Merged ордер має тільки один ID. У логах і при діагностиці неможливо визначити, який grid level породив цей ордер.

**Зміна.**
При merge зберігати обидва IDs у тегу:
```python
merged_tag = f"{existing.tag}+{order.tag}"
merged = replace(existing, size=round(existing.size + order.size, 8), tag=merged_tag)
```

---

## P3 — Довгострокові покращення

Значне покращення можливостей, але не критичні для поточного live-режиму.

---

### P3-1. WebSocket підписка на live price між циклами

**Поточно:** ціна береться раз на цикл через REST API Bybit.

**Проблема.** Між плановими циклами (кожну годину) ціна може різко змінитись на 3–5%. Бот реагує тільки на початку наступного циклу. Якщо відбувся flash crash — buy-ордери нижче ринку можуть виконатись масово, а бот дізнається про це тільки через годину.

**Зміна.**
Додати Bybit WebSocket subscriber (публічний channel `publicTrade` або `tickers`) у `infrastructure/`. При значному відхиленні від cached price (наприклад, > 2×ATR) — тригерити позаплановий цикл або принаймні скасовувати buy-ордери через emergency cancel.

---

### P3-2. Multi-timeframe аналіз режиму

**Поточно:** тільки 1h свічки, lookback 2400 барів (~100 днів).

**Проблема.** Режим на 1h може бути RANGE, а на 4h — DOWNTREND. Бот відкриває нові BUY у локальному RANGE всередині ведмедячого тренду.

**Зміна.**
Завантажувати додатково 4h або daily свічки для regime confirmation. Якщо вищий таймфрейм — DOWNTREND, знижувати allocation weight або блокувати нові BUY незалежно від 1h сигналу. Логіка може бути в `symbol_analyzer.py` як додаткова перевірка перед `regime_detector.detect()`.

---

### P3-3. Backtest sound report (HTML або Jupyter)

**Поточно:** `backtesting/reporting.py` повертає текстовий summary.

**Зміна.**
Додати `export_html_report(result, path)` або Jupyter notebook template із:
- Equity curve (matplotlib/plotly).
- Regime timeline (кольоровий графік).
- Rebuild heatmap (де відбувались rebuild по ціні/часу).
- Distribution of realized PnL per trade.
- Risk reason frequency chart.

---

### P3-4. Health check endpoint

**Поточно:** немає способу перевірити, чи бот живий, без доступу до логів.

**Зміна.**
Додати мінімальний HTTP health-check сервер (стандартна бібліотека `http.server` або `aiohttp`):
- `GET /health` → `{"status": "ok", "last_cycle": "2026-05-01T00:01:03", "symbols": ["ETHUSDT"]}`
- `GET /state` → поточний regime і kill_switch_count по символах.

Запускати як окремий asyncio task поряд зі scheduler. Дозволяє моніторинг через UptimeRobot або Prometheus.

---

### P3-5. Dry-run mode із детальним diff-звітом

**Поточно:** є `paper` mode, але він не показує що ЗМІНИЛОСЬ порівняно з live.

**Зміна.**
Додати `--dry-run` CLI flag: бот проводить повний цикл планування і виводить structured diff:
```
[ETHUSDT] Режим: RANGE → RANGE (без змін)
[ETHUSDT] Новий BUY @ 1800.0 x 0.055  ← буде поставлено
[ETHUSDT] Скасування BUY @ 1820.0    ← буде скасовано
[ETHUSDT] SELL @ 1950.0 x 0.03       ← без змін (вже є)
```
Корисно для відлагодження без ризику реальних ордерів.

---

## Зведена таблиця

| ID | Пріоритет | Файли | Суть | Ризик без виправлення |
|---|---|---|---|---|
| P0-0 | P0 | `bybit_account_client.py`, `execution_gateway.py`, `state_store.py`, `runtime_models.py`, `spot_grid_planner.py`, `market_data_provider.py`, `analysis_batch_service.py` | Хибний cost basis з execution history → sell нижче собівартості | **Прямий фінансовий збиток**, no-loss правило порушено |
| P0-1 | P0 | `execution_guardrails.py` | Неправильний порядок фільтрів | Sell-ордери відкидаються після dedupe |
| P0-2 | P0 | `state_store.py` | `AttributeError` маскує збій БД | Реальна причина збою невидима оператору |
| P0-3 | P0 | `scheduler.py` | Виняток зупиняє всі символи | Бот зупиняється без автовідновлення |
| P1-1 | P1 | `state_machine.py` | Мутація стану замість іммутабельних переходів | Прихована регресія при рефакторингу |
| P1-2 | P1 | `order_diff.py` | Нестабільний matching для дешевих монет | Зайві rebuild у XRP/DOGE/SUI |
| P1-3 | P1 | `rebuild_policy.py` | Фіксований ціновий поріг без ATR | Надмірні/недостатні rebuild залежно від активу |
| P1-4 | P1 | `uptrend_policy.py` | Порівняння рядків замість enum | Фільтрація buy-рівнів тихо зламається |
| P1-5 | P1 | `notifications.py` | Немає реальних сповіщень | PANIC/kill switch без оповіщення оператора |
| P1-6 | P1 | `state_store.py` | Нове з'єднання БД на кожну операцію | Зайва латентність, вичерпання підключень |
| P1-7 | P1 | `analysis_batch_service.py` | Надто широкий `except Exception` | Фатальні помилки маскуються як trading помилки |
| P2-1 | P2 | `range_entry_policy.py` | Breakdown signal без напряму | Пропуск BUY під час upswing у RANGE |
| P2-2 | P2 | `backtesting/engine.py` | Backtest без slippage і комісії | Завищені backtest-результати |
| P2-3 | P2 | `bootstrap.py` | Відсутня валідація конфігу | Мертві config-параметри без попередження |
| P2-4 | P2 | `rebuild_policy.py` | `diff_count > 0` завжди тригерить rebuild | Надмірні rebuild при мінімальних змінах |
| P2-5 | P2 | `target_order_builder.py` | No-loss block мовчить | Не видно чому sell-рівні відсутні |
| P2-6 | P2 | `grid_viability.py` | Втрата ID при merge ордерів | Складна діагностика злитих рівнів |
| P3-1 | P3 | `infrastructure/` | WebSocket між циклами | Реакція на flash crash тільки через годину |
| P3-2 | P3 | `symbol_analyzer.py` | Multi-timeframe аналіз | Купівлі у локальному RANGE всередині downtrend |
| P3-3 | P3 | `backtesting/` | HTML backtest report | Складна діагностика backtest без графіків |
| P3-4 | P3 | новий файл | Health check HTTP endpoint | Немає способу перевірити живість без логів |
| P3-5 | P3 | `main.py` | Dry-run CLI mode | Ручне відлагодження потребує реальних ордерів |

---

## Виявлені неузгодженості між задачами

Перед реалізацією важливо врахувати взаємозалежності між задачами, що торкаються одних і тих самих файлів.

---

### Неузгодженість 1. P0-0, P0-2 і P1-6 — три зміни в `state_store.py`

**Проблема.**
- **P0-0** додає нову колонку `cost_basis_price` у таблицю та нову логіку читання/запису.
- **P0-2** виправляє `conn = None` паттерн у тих самих методах.
- **P1-6** повністю замінює цей паттерн на connection pool (`async with pool.acquire()`), після чого P0-2 стає зайвим.

Якщо реалізовувати окремо і в неправильному порядку:
- P0-0 вноситься поверх сирого `conn` паттерну → AttributeError все ще можливий.
- P0-2 виправляє паттерн → потім P1-6 все переписує → P0-2 був зайвою роботою.

**Рекомендація.** Реалізувати всі три за **один PR** у такому порядку коду:
1. Мігрувати `state_store.py` на connection pool (суть P1-6) — це автоматично усуває AttributeError (P0-2).
2. Додати колонку і логіку `cost_basis_price` (суть P0-0) вже поверх чистого pool-коду.
3. P0-2 як окрема задача закривається автоматично.

---

### Неузгодженість 2. P1-3 і P2-4 — обидві змінюють `rebuild_policy.py`

**Проблема.**
- **P1-3** додає ATR-based adaptive threshold у функцію `should_rebuild`.
- **P2-4** додає `diff_count_threshold` у ту ж функцію.

Реалізовані окремо і в різний час — конфлікт при merge, або одна зміна затирає іншу.

**Рекомендація.** Реалізувати **разом в одному PR**. Сигнатура `should_rebuild` зміниться один раз: додати `atr14: float` і `diff_threshold: int` одночасно.

---

### Неузгодженість 3. P0-0 змінює `MarketDataProvider` protocol у `ports.py`

**Проблема.**
P0-0 додає параметр `persisted_cost_basis` до `get_market_context`. Це ламає протокол `MarketDataProvider` у `application/ports.py`:
```python
class MarketDataProvider(Protocol):
    async def get_market_context(self, symbol: str) -> MarketContext:  # ← стара сигнатура
```
Будь-який клас, що реалізує цей протокол (включаючи тести, mock-и, `NoOpMarketDataSynchronizer`), потребує оновлення.

**Рекомендація.** При реалізації P0-0 одразу оновити `ports.py`:
```python
async def get_market_context(self, symbol: str, *, persisted_cost_basis: float | None = None) -> MarketContext:
```
Перевірити всі реалізації протоколу в тестах.

---

### Неузгодженість 4. P0-0 прибирає `resolve_cost_basis_price` з `BybitSpotExchange`, але P2-5 покладається на `inventory.cost_basis_price`

**Проблема.**
P2-5 додає логування `minimum_exit_price(inventory, config)` у `target_order_builder.py`. Якщо P0-0 ще не реалізований — `cost_basis_price` може бути хибним. Логування хибного значення може ввести оператора в оману.

**Рекомендація.** P2-5 реалізовувати тільки після P0-0. Або додати у лог позначку джерела: `cost_basis_source=avgPrice|persisted|unavailable`.

---

### Неузгодженість 5. P0-1 (порядок guardrails) і P0-0 (cost basis)

**Проблема.**
P0-1 виправляє порядок `filter_no_loss_sells` у `execution_guardrails.py`. Але `filter_no_loss_sells` викликає `minimum_exit_price(inventory, ...)`, яка залежить від `inventory.cost_basis_price`. Якщо P0-0 ще не реалізований — `cost_basis_price` може бути хибним або відсутнім, і no-loss filter або пропускає поганий sell, або блокує всі sells.

**Рекомендація.** P0-1 корисний незалежно, але повний захист no-loss дає тільки комбінація P0-0 + P0-1. Реалізувати обидва в одному релізі.

---

### Неузгодженість 6. `cost_basis_qty` у старому описі — більше не потрібна

**Проблема.**
Попередня версія P0-0 передбачала зберігання `cost_basis_qty` у PostgreSQL для incremental calculation. Після переходу на `avgPrice` від Bybit (який вже є готовим weighted average) — `cost_basis_qty` непотрібна: Bybit веде цей розрахунок на своєму боці.

**Рекомендація.** У `SymbolRuntimeState` додати **тільки** `cost_basis_price: float | None`. Поле `cost_basis_qty` не додавати.

---

## Рекомендований порядок роботи

```
Тиждень 1 (один PR):  P0-0 + P0-2 + P1-6  — cost basis виправлення + state_store pool
                       P0-1                  — guardrail order fix (незалежний PR)
                       P0-3                  — scheduler safety (незалежний PR)

Тиждень 2:  P1-1 → P1-2 → P1-4 → P1-7  (~8–10 год)

Тиждень 3:  P1-3 + P2-4  (разом, один PR для rebuild_policy.py)
            P1-5          (Telegram notifier)

Тиждень 4:  P2-1 → P2-2 → P2-3 → P2-5* → P2-6  (*тільки після P0-0)

Далі:       P3-* за потребою
```

> **Увага:** поки P0-0 не реалізований — у live-режимі вручну перевірити, чи не є жодна з поточних позицій настільки давньою, щоб execution history вже "прокрутилась" за 100 записів. Якщо так — `cost_basis_price` для цього символу ненадійний, SELL ордери можуть бути виставлені нижче реальної собівартості.

Кожне виправлення P0/P1 повинно мати unit-тест перед мерджем. Для P0-0 — обов'язково перевірити на paper mode і переконатись що `avgPrice` коректно повертається для тестового акаунту Bybit demo.

