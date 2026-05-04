# Spot Grid Bot

## Огляд

`spot_grid_bot` це адаптивний спотовий grid-бот для Bybit spot/demo trading із власним шляхом ingest свічок, portfolio-aware planning, persistence runtime state, діагностикою та operator tooling.

Поточна точка входу: [main.py](./main.py)

Зараз бот включає:

- PostgreSQL-backed збереження свічок і runtime state
- Binance Spot синхронізацію свічок
- Bybit spot баланси, live orders і execution sync
- regime detection за індикаторами та market structure
- multi-timeframe підтвердження режиму
- risk-aware planning і staged de-risking
- portfolio-level budget allocation між символами
- persisted cost basis із `avgPrice` як primary source і no-loss логікою для `SELL`
- tick-aware diff target/live orders і venue-aware normalization grid-рівнів
- Telegram notifications для критичних подій
- live price WebSocket monitoring для позапланової реакції
- HTTP health/state endpoints
- `dry-run` режим зі structured diff
- backtesting із діагностикою та HTML report export

## Структура проєкту

- [main.py](./main.py): CLI entrypoint і dispatch режимів процесу
- [application](./application): bootstrap, scheduler, health, dry-run, orchestration services
- [domain](./domain): strategy, regime, risk, allocation, grid, cost basis, state machine
- [infrastructure](./infrastructure): Binance sync, Bybit adapters, PostgreSQL repositories, notifier, live price monitor
- [backtesting](./backtesting): backtest engine і reporting
- [tests](./tests): unit і scenario tests
- [utils](./utils): environment loading, runtime config, logging

## Швидкий старт

### 1. Створи локальне середовище

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

### 2. Налаштуй середовище

Створи або онови `.env` мінімум такими значеннями:

```env
DB_HOST=localhost
DB_PORT=5432
DATABASE=pompilo_db
DB_USER=admin
DB_PASS=admin_pass

BYBIT_API_KEY=...
BYBIT_API_SECRET=...
BYBIT_API_ENDPOINT=https://api-demo.bybit.com

SPOT_TRADING_SYMBOLS=ETHUSDT,SOLUSDT
EXECUTION_MODE=bybit_spot_demo
```

Опційні operator features:

```env
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...

HEALTHCHECK_HOST=0.0.0.0
HEALTHCHECK_PORT=8080

ENABLE_LIVE_PRICE_MONITOR=true
LIVE_PRICE_DEVIATION_ATR_MULTIPLIER=2.0
LIVE_PRICE_MONITOR_COOLDOWN_SECONDS=60
```

### 3. Синхронізуй свічки

```bash
./.venv/bin/python main.py sync --period 365 --timeframe 1h
```
```bash
./.venv/bin/python main.py sync --period 365 --timeframe 4h
```

### 4. Запусти один planning/execution cycle

```bash
./.venv/bin/python main.py once
```

### 5. Переглянь зміни без торкання live orders

```bash
./.venv/bin/python main.py dry-run
```

### 6. Запусти recurring live scheduler

```bash
./.venv/bin/python main.py live
```

### 7. Запусти повний набір тестів

```bash
./.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

## CLI Команди

### `sync`

Завантажує свічки з Binance Spot і зберігає їх у PostgreSQL.

```bash
./.venv/bin/python main.py sync --period 365 --timeframe 1h
./.venv/bin/python main.py sync --period 30 --timeframe 4h
```

### `once`

Запускає один повний analysis/planning/execution cycle для всіх налаштованих символів.

```bash
./.venv/bin/python main.py once
```

### `dry-run`

Запускає повний analysis/planning cycle і виводить structured diff відносно поточних live orders, не синхронізуючи жоден ордер на біржу.

```bash
./.venv/bin/python main.py dry-run
```

Приклад форми виводу:

```text
[ETHUSDT] Regime: RANGE
[ETHUSDT] New BUY @ 1800.0 x 0.055
[ETHUSDT] Cancel BUY @ 1820.0 x 0.055
[ETHUSDT] Keep SELL @ 1950.0 x 0.03
```

### `live`

Запускає recurring scheduler. У `live` режимі процес:

- чекає до заданого часу циклу
- синхронізує свічки перед кожним scheduled cycle
- виконує trading cycle для всіх configured symbols
- піднімає health endpoints
- за потреби запускає WebSocket live-price monitor у background

```bash
./.venv/bin/python main.py live
```

## Runtime Flow

Для кожного символу торговий процес тепер працює так:

1. Переконується, що candle schema і per-symbol candle tables існують.
2. Відновлює persisted per-symbol runtime state із PostgreSQL.
3. Перед scheduled cycle синхронізує свічки з Binance.
4. Завантажує recent 1h candles із PostgreSQL.
5. Будує агрегований higher-timeframe candle stream із тієї ж базової історії.
6. Завантажує live balances, mark price, instrument filters і open orders із Bybit.
7. Визначає `cost_basis_price` через Bybit `avgPrice`, а якщо треба, fallback-иться на persisted runtime state.
8. Рахує indicator snapshot.
9. Визначає regime за індикаторами та market structure.
10. Застосовує higher-timeframe downtrend confirmation до фінальних entry-рішень.
11. Оцінює risk, projected exposure і de-risk mode.
12. Робить preliminary analysis для всіх символів.
13. Будує portfolio snapshot і розподіляє symbol budgets.
14. Просуває state machine.
15. Будує target orders.
16. Застосовує no-loss, venue і execution guardrails.
17. Порівнює target orders із поточними live orders.
18. Виконує sync тільки якщо rebuild справді потрібен.
19. Зберігає оновлений runtime state назад у PostgreSQL.
20. Оновлює health/state snapshot для operator endpoints.

Основна orchestration логіка:

- [application/trading_cycle_service.py](./application/trading_cycle_service.py)
- [application/scheduler.py](./application/scheduler.py)
- [application/bootstrap.py](./application/bootstrap.py)

## Детальний опис фіч

### 1. Ingest і збереження свічок

Бот сам повністю відповідає за candle-ingestion path.

- Свічки беруться з Binance Spot для обох таймфреймів (1h та 4h).
- Свічки зберігаються в PostgreSQL у `_candles_trading_data.<symbol>_1h` та `_candles_trading_data.<symbol>_4h`.
- Candle schema/tables створюються автоматично.
- Scheduler оновлює обидва набори market data (1h та 4h) перед кожним scheduled cycle.
- Planner працює з історією із PostgreSQL, а не напряму з біржовими klines.

Код:

- [infrastructure/db.py](./infrastructure/db.py)
- [infrastructure/binance_api.py](./infrastructure/binance_api.py)
- [infrastructure/binance_market_data_synchronizer.py](./infrastructure/binance_market_data_synchronizer.py)
- [infrastructure/market_data_provider.py](./infrastructure/market_data_provider.py)

### 2. Market Regime Detection

Стратегія працює з режимами:

- `RANGE`
- `UPTREND`
- `DOWNTREND`
- `HIGH_VOLATILITY`
- `RISK_OFF`

Regime detection поєднує:

- `ema20`, `ema50`, `ema200`
- slope `ema50`
- `atr14`
- realized volatility
- range width
- directional move і directional sign
- abnormal candle та ATR spike flags
- swing-high / swing-low market structure

Детектор тепер також використовує higher-timeframe confirmation:

- market-data provider будує грубіший candle stream із базової 1h історії
- якщо higher timeframe у `DOWNTREND`, fresh entries ставляться на паузу, навіть коли lower timeframe ще виглядає як `RANGE`

Код:

- [domain/indicators.py](./domain/indicators.py)
- [domain/market_structure.py](./domain/market_structure.py)
- [domain/regime_detector.py](./domain/regime_detector.py)
- [domain/symbol_analyzer.py](./domain/symbol_analyzer.py)

### 3. State Machine і Runtime Memory

Бот тримає persistent per-symbol runtime state:

- поточний strategy state
- hysteresis/pending regime transition state
- volatility cooldown
- kill-switch history
- persisted `cost_basis_price`

State machine тепер робить immutable transitions, тому preview analysis безпечний і не псує runtime state під час non-committing pass.

Код:

- [domain/state_machine.py](./domain/state_machine.py)
- [domain/runtime_models.py](./domain/runtime_models.py)
- [infrastructure/state_store.py](./infrastructure/state_store.py)

### 4. Risk Logic

Risk layer оцінює:

- breakout kill switch
- abnormal volatility
- emergency volatility
- daily drawdown pause
- quote reserve pressure
- symbol-level і portfolio-level inventory limits
- projected outstanding buy exposure
- state-based restrictions у `DOWNTREND`, `HIGH_VOLATILITY` і `RISK_OFF`

Risk output включає:

- `pause_entries`
- `force_risk_off`
- `cancel_entries`
- `allow_exit_only`
- `de_risk_mode = NONE | SOFT | HARD | PANIC`

Код:

- [domain/risk_manager.py](./domain/risk_manager.py)
- [domain/exposure.py](./domain/exposure.py)

### 5. Portfolio-Level Allocation

Planner тепер portfolio-aware ще до фінального per-symbol order generation.

Він:

- запускає preliminary symbol analysis по всіх configured symbols
- будує один shared portfolio snapshot
- рахує один global new-entry budget із free quote і current outstanding exposure
- розподіляє цей budget тільки між eligible symbols
- штрафує underwater inventory і exposure-heavy symbols ще до local grid sizing

Код:

- [domain/portfolio_allocator.py](./domain/portfolio_allocator.py)
- [domain/allocation.py](./domain/allocation.py)
- [application/analysis_batch_service.py](./application/analysis_batch_service.py)

### 6. Cost Basis і No-Loss Sell Enforcement

Бот тепер трактує Bybit `avgPrice` як primary source для spot cost basis.

Поведінка:

- primary source: Bybit wallet `avgPrice`
- fallback source: persisted `cost_basis_price` із PostgreSQL/runtime state
- якщо обидва недоступні й позиція відкрита, `SELL` planning блокується
- коли inventory стає нульовим, persisted `cost_basis_price` очищається

No-loss логіка:

- sell targets rebased вище minimum exit floor
- sell orders нижче no-loss floor блокуються
- planning логує випадки, коли sell levels прибираються через відсутній cost basis або через no-loss floor

Код:

- [infrastructure/bybit_account_client.py](./infrastructure/bybit_account_client.py)
- [domain/cost_basis.py](./domain/cost_basis.py)
- [domain/target_order_builder.py](./domain/target_order_builder.py)
- [infrastructure/execution_guardrails.py](./infrastructure/execution_guardrails.py)

### 7. Grid Planning і Venue Awareness

Grid planning тепер venue-aware.

Бот:

- будує ATR-based range і trend grids
- поважає symbol-specific tick size і size step
- зливає duplicate normalized levels після tick rounding
- зберігає merged source tags для діагностики
- не створює хибних target/live mismatch на дешевих монетах, бо one-tick drift обробляється безпечно

Код:

- [domain/grid_builder.py](./domain/grid_builder.py)
- [domain/grid_viability.py](./domain/grid_viability.py)
- [domain/order_diff.py](./domain/order_diff.py)

### 8. Underwater Recovery Logic

Коли inventory underwater, стратегія може перейти в controlled recovery behavior.

Бот може:

- зменшувати або обмежувати кількість нових buy levels
- обмежувати recovery budget часткою від free quote
- використовувати різні recovery budgets для `RANGE` і `UPTREND`
- робити sell behavior агресивнішим навколо recovery exits

Код:

- [domain/uptrend_policy.py](./domain/uptrend_policy.py)
- [domain/target_order_builder.py](./domain/target_order_builder.py)

### 9. Execution Guardrails

Перед sync target orders на біржу бот застосовує:

- no-loss sell filtering
- marketable-order filtering
- dedupe близьких рівнів
- per-cycle order throttles
- venue minimum notional / minimum quantity normalization

Guardrails навмисно застосовуються в domain-first порядку:

1. no-loss restrictions
2. marketability checks
3. dedupe / cleanup
4. execution throttles

Код:

- [infrastructure/execution_guardrails.py](./infrastructure/execution_guardrails.py)
- [infrastructure/execution_gateway.py](./infrastructure/execution_gateway.py)

### 10. Notifications

Notifier layer зараз підтримує:

- logging-only notifications за замовчуванням
- Telegram notifications для critical rebuild/risk events

Telegram alerts надсилаються лише для high-severity випадків, таких як:

- `force_risk_off`
- `de_risk_mode = HARD | PANIC`
- kill-switch-triggered behavior
- critical risk reasons, наприклад `daily_drawdown_pause` або `emergency_volatility`

Код:

- [infrastructure/notifications.py](./infrastructure/notifications.py)

### 11. Live Price Monitoring

У `live` режимі scheduler може запускати Bybit public WebSocket monitor між регулярними циклами.

Поведінка:

- підписується на ticker updates
- порівнює live price з cached cycle reference
- використовує ATR-based deviation threshold
- при сильному price shock запускає off-cycle trading pass для конкретного символу
- throttle-ить повторні спрацювання через cooldown

Код:

- [infrastructure/live_price_monitor.py](./infrastructure/live_price_monitor.py)
- [application/scheduler.py](./application/scheduler.py)

### 12. Health і State Endpoints

Коли `HEALTHCHECK_PORT > 0`, live process піднімає:

- `GET /health`
- `GET /state`

`/health` повертає process-level liveness information:

```json
{
  "status": "ok",
  "last_cycle": "2026-05-01T15:00:01+00:00",
  "symbols": ["ETHUSDT", "SOLUSDT"]
}
```

`/state` повертає latest in-memory runtime summary по символах, включно з current regime і kill-switch count.

Код:

- [application/health.py](./application/health.py)

### 13. Dry-Run Preview Mode

`dry-run` виконує повний planning path, але ніколи не sync-ить ордери.

Це корисно для:

- безпечного дебагу проти реального venue state
- перегляду того, що буде створено, залишено або скасовано
- аналізу regime/risk-driven diff перед увімкненням live execution

Код:

- [application/dry_run.py](./application/dry_run.py)
- [main.py](./main.py)

### 14. Backtesting і Reporting

Backtest layer тепер включає:

- historical order simulation
- slippage-aware fill accounting
- optional maker-fee accounting
- no-loss sell diagnostics
- rebuild count і de-risk diagnostics
- risk-reason frequency tracking
- final inventory і unrealized/realized PnL reporting
- HTML report export

Programmatic usage:

```python
from backtesting import BacktestEngine, export_html_report
from domain.strategy_config import DEFAULT_STRATEGY_CONFIG

engine = BacktestEngine(DEFAULT_STRATEGY_CONFIG)
result = engine.run("ETHUSDT", candles)
export_html_report(result, "backtest_report.html")
```

Код:

- [backtesting/engine.py](./backtesting/engine.py)
- [backtesting/reporting.py](./backtesting/reporting.py)

## Ключова Runtime Конфігурація

Важливі environment variables:

- `SPOT_TRADING_SYMBOLS`: список символів через кому
- `EXECUTION_MODE`: режим execution adapter
- `BYBIT_API_KEY`
- `BYBIT_API_SECRET`
- `BYBIT_API_ENDPOINT`
- `DB_HOST`
- `DB_PORT`
- `DATABASE`
- `DB_USER`
- `DB_PASS`
- `CANDLE_LOOKBACK`
- `MAX_NEW_ORDERS_PER_CYCLE`
- `MIN_SYMBOL_ENTRY_NOTIONAL`
- `MAX_SYMBOL_INVENTORY_PCT_OF_EQUITY`
- `MAX_SYMBOL_NEW_ENTRY_PCT_OF_FREE_QUOTE`
- `UNDERWATER_AVERAGING_ENABLED`
- `UNDERWATER_AVERAGING_TRIGGER_PCT`
- `UNDERWATER_RECOVERY_BUDGET_PCT`
- `UNDERWATER_RANGE_BUDGET_MULTIPLIER`
- `UNDERWATER_UPTREND_BUDGET_MULTIPLIER`
- `UNDERWATER_MAX_RECOVERY_LEVELS`
- `UNDERWATER_DEEP_STOP_PCT`
- `RUN_TARGET_MINUTE`
- `RUN_TARGET_SECOND`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `HEALTHCHECK_HOST`
- `HEALTHCHECK_PORT`
- `ENABLE_LIVE_PRICE_MONITOR`
- `LIVE_PRICE_DEVIATION_ATR_MULTIPLIER`
- `LIVE_PRICE_MONITOR_COOLDOWN_SECONDS`

Source of truth:

- [utils/config.py](./utils/config.py)

## Тестування

Test suite покриває:

- regime detection і state transitions
- runtime state persistence
- cost basis handling і no-loss sell logic
- execution guardrails
- portfolio allocation
- underwater recovery logic
- cheap-symbol tick handling
- scheduler resilience
- Telegram notifier behavior
- health endpoint behavior
- live price monitor deviation handling
- dry-run formatting
- backtest diagnostics і reporting

Запуск:

```bash
./.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Поточний стан suite:

- `94` тести проходять локально в `.venv`

