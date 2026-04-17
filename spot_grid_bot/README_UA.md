# Spot Grid Bot

## Огляд

`spot_grid_bot` це адаптивний спотовий grid-бот для Bybit.

Зараз він:

- сам створює і підтримує схему/таблиці свічок у PostgreSQL
- сам синхронізує свічки напряму з Binance Spot
- читає свічки з PostgreSQL
- читає баланси, відкриті ордери, тікери та виконання з Bybit spot
- визначає ринковий режим за індикаторами та структурою ринку
- оцінює ризик і projected exposure
- рахує budget для нових покупок у `USDT` через percentage-based caps
- розподіляє budget через portfolio-level allocator
- підтримує trigger-based underwater averaging від вільного `USDT`
- будує regime-aware grid або de-risk ордери
- дотримується no-loss логіки для `SELL`
- зберігає runtime state по кожному символу в PostgreSQL
- має шар backtesting/evaluation з діагностикою

Точка входу: [main.py](./main.py)

## Структура проєкту

- [main.py](./main.py): точка входу
- [application](./application): bootstrap, orchestration, scheduler, ports
- [domain](./domain): strategy, regime, risk, grid, inventory, cost basis, market structure
- [infrastructure](./infrastructure): Binance candle sync, Bybit exchange adapter, PostgreSQL candles, runtime state store
- [backtesting](./backtesting): backtest engine і reporting helpers
- [tests](./tests): unit і scenario tests
- [utils](./utils): config, env loading, logging

## Runtime Flow

Для кожного торгового символу один цикл працює так:

1. Переконується, що схема свічок і таблиці для символів існують.
2. У `live` режимі оновлює свіжі свічки з Binance Spot.
3. Завантажує свічки з PostgreSQL.
4. Завантажує live balances і open orders з Bybit.
5. Відновлює збережений runtime state для символу.
6. Рахує snapshot індикаторів.
7. Визначає ринковий режим.
8. Оцінює ризик і projected exposure.
9. Робить preliminary analysis для всіх символів.
10. Будує один portfolio snapshot на весь цикл.
11. Розподіляє portfolio-level entry budgets між eligible symbols.
12. Просуває state machine.
13. Будує target orders.
14. Порівнює target orders із live orders.
15. Якщо потрібен rebuild, синхронізує ордери з біржею.
16. Зберігає оновлений runtime state.

Основна orchestration логіка: [application/trading_cycle_service.py](./application/trading_cycle_service.py)  
Planner: [domain/spot_grid_planner.py](./domain/spot_grid_planner.py)  
Portfolio allocator: [domain/portfolio_allocator.py](./domain/portfolio_allocator.py)

## Джерела даних

### Свічки

`spot_grid_bot` тепер сам повністю відповідає за отримання й збереження свічок.

Свічки беруться з Binance Spot і зберігаються в PostgreSQL у таблицях виду:

- `_candles_trading_data.<symbol>_p_candles`

Схема свічок повністю збігається з `futures-trading-bot`:

- `open_time`
- `close_time`
- `symbol`
- `open`
- `close`
- `high`
- `low`
- `cvd`
- `volume`
- `candle_id`

Бот:

- створює candle schema на старті
- створює candle tables для символів на старті
- може окремо запускати historical/refresh sync через CLI
- оновлює свічки перед кожним scheduled live cycle
- за замовчуванням аналізує останні `2400` годинних свічок по кожному символу

Код:

- [infrastructure/db.py](./infrastructure/db.py)
- [infrastructure/binance_api.py](./infrastructure/binance_api.py)
- [infrastructure/binance_market_data_synchronizer.py](./infrastructure/binance_market_data_synchronizer.py)
- [infrastructure/market_data_provider.py](./infrastructure/market_data_provider.py)

### Live Exchange State

Баланси, тікер, відкриті ордери та історія виконань беруться з Bybit V5 spot API.

Код:

- [infrastructure/execution_gateway.py](./infrastructure/execution_gateway.py)

## Ринкові режими

Бот працює з такими режимами:

- `RANGE`
- `UPTREND`
- `DOWNTREND`
- `HIGH_VOLATILITY`
- `RISK_OFF`

Детектор режиму використовує:

- `ema20`, `ema50`, `ema200`
- `ema50 slope`
- `atr14`
- realized volatility
- range width
- directional move
- abnormal candle і ATR spike flags
- market structure зі swing highs і swing lows

Шар market structure:

- знаходить локальні swings
- переходить на segmented extrema, якщо тренд надто гладкий для класичних pivot points
- класифікує структуру як:
  - `BULLISH`
  - `BEARISH`
  - `RANGE`
  - `MIXED`
  - `NEUTRAL`

Трендові режими зараз підтверджуються і індикаторами, і структурою:

- `UPTREND`: bullish EMA context + bullish structure
- `DOWNTREND`: bearish EMA context + bearish structure
- `RANGE`: flat/contained context або mixed/neutral structure
- зламана або суперечлива структура тренду деградує в `RANGE`, а не лишає хибне продовження тренду

Код:

- [domain/indicators.py](./domain/indicators.py)
- [domain/market_structure.py](./domain/market_structure.py)
- [domain/regime_detector.py](./domain/regime_detector.py)

## State Machine

Бот використовує state machine, щоб не перемикати режим на кожній шумовій свічці.

Вона включає:

- hysteresis confirmation bars
- cooldown після зміни стану
- `volatility_cooldown_remaining`

Код:

- [domain/state_machine.py](./domain/state_machine.py)

На практиці це означає:

- одна шумова свічка не повинна миттєво перевернути режим
- після `HIGH_VOLATILITY` entries можуть лишатися заблокованими ще кілька барів

## Логіка ризику

Risk layer оцінює:

- breakout kill switch
- abnormal volatility
- emergency volatility
- symbol inventory cap як `% від equity`
- symbol new-entry cap як `% від free quote`
- absolute symbol notional cap
- low quote reserve
- daily drawdown pause
- projected exposure від outstanding buy orders
- обмеження для `DOWNTREND` і `RISK_OFF`

Код:

- [domain/risk_manager.py](./domain/risk_manager.py)
- [domain/exposure.py](./domain/exposure.py)

Результат risk layer включає:

- `pause_entries`
- `force_risk_off`
- `cancel_entries`
- `allow_exit_only`
- `de_risk_mode = NONE | SOFT | HARD | PANIC`
- projected exposure diagnostics

Поточна модель sizing:

- budget для нових покупок рахується в `USDT`, а не в “штуках монети”
- planner бере мінімум із:
  - залишку room по inventory як `% від total equity`
  - symbol new-entry cap як `% від free quote`
  - global new-entry cap як `% від free quote`
  - absolute symbol notional cap
- якщо результат нижчий за `min_symbol_entry_notional`, бот пропускає нові `BUY` по цьому символу

Код:

- [domain/allocation.py](./domain/allocation.py)
- [domain/inventory_manager.py](./domain/inventory_manager.py)
- [domain/risk_manager.py](./domain/risk_manager.py)

## Portfolio-Level Allocation

Бот тепер запускає portfolio-aware allocation pass перед фінальним order planning.

Поточний portfolio flow:

- збирає preliminary analysis для всіх символів
- будує один спільний portfolio snapshot
- рахує один global new-entry budget із free quote і поточного outstanding exposure
- розподіляє budget тільки між eligible symbols
- передає portfolio budget вниз у symbol-local allocation layer

Eligible symbols зараз:

- `RANGE`
- `UPTREND`

Символ виключається з нових входів, якщо:

- `pause_entries=True`
- `force_risk_off=True`
- режим `DOWNTREND`
- режим `HIGH_VOLATILITY`
- режим `RISK_OFF`

Розподіл budget зараз враховує:

- regime weight
- regime confidence
- inventory pressure
- outstanding buy pressure
- projected quote-usage pressure
- underwater inventory penalty
- max concurrent entry symbols

Це означає:

- portfolio capital більше не залежить тільки від порядку обходу символів
- лише обмежена кількість символів може отримати fresh entry budget за один цикл
- underwater inventory і далі штрафується на рівні portfolio allocation ще до того, як розглядається recovery логіка

## Venue-Aware Grid Handling

Сітка лишається ATR-based, але більше не будується так, ніби всі символи мають однаковий `price_tick`.

Поточна логіка:

- raw grid geometry будується від ATR
- planner отримує symbol-specific venue constraints із Bybit:
  - `tick_size`
  - `qty_step`
  - `min_order_qty`
  - `min_order_amt`
- planner застосовує venue-viability pass до execution:
  - prices нормалізуються до symbol tick size
  - duplicate normalized levels об’єднуються
  - sell ladders скорочуються, якщо поточний inventory не проходить мінімальні біржові вимоги
- execution потім застосовує live-price safety offset для buys

Для дешевих символів, як `ENA`, `SUI`, `XRP`, `DOGE`, це важливо, бо:

- глобальне округлення до `0.1` злипало рівні
- exchange minima могли робити multi-level sell ladders нереалістичними
- зсув верхнього buy level від ринку має зберігати форму ladder, а не зводити всі buys в одну ціну

Поточна buy-offset поведінка:

- якщо верхній `BUY` занадто близько до live price, зсувається вся buy ladder
- форма ladder зберігається, замість того щоб репрайсити всі `BUY` в одну точку

Код:

- [domain/grid_builder.py](./domain/grid_builder.py)
- [domain/grid_viability.py](./domain/grid_viability.py)
- [domain/inventory_manager.py](./domain/inventory_manager.py)
- [infrastructure/market_data_provider.py](./infrastructure/market_data_provider.py)
- [infrastructure/execution_gateway.py](./infrastructure/execution_gateway.py)

## Торгова логіка

### RANGE

Бот будує симетричну сітку навколо поточної ціни:

- `BUY` нижче ціни
- `SELL` вище ціни
- ширина і крок залежать від ATR
- якщо quality у боковику слабка, бот може пом’якшувати нові `BUY`, а не повністю зупиняти торгівлю
- якщо inventory underwater і просадка перевищила recovery trigger, бот може перейти на обмежений recovery buy budget

### UPTREND

Бот використовує окрему pullback-and-scale-out логіку:

- `BUY` рівні ставляться нижче ринку як ATR-based pullbacks
- buy budget зміщується в бік глибших pullback levels
- якщо ціна занадто сильно відірвалась від `ema20`, нові trend buys блокуються
- якщо inventory underwater і recovery trigger уже виконаний, бот може використати більший recovery budget, ніж у `RANGE`
- `SELL` рівні ширші, ніж у `RANGE`
- чим вищий take-profit level, тим більший sell size
- take-profit у тренді може адаптуватися до сили тренду, а не бути повністю статичним

### DOWNTREND

Нові покупки блокуються. Бот може перейти в staged de-risk behavior замість відкриття нових entries.

Якщо по символу ще висять старі buy-entry orders з попередньої фази, planner примусово робить rebuild, щоб вони були прибрані на наступному sync.

### HIGH_VOLATILITY

Нові entries пригнічуються, бот використовує explicit de-risk logic.

Відкриті buy-entry orders вважаються stale для цієї protective phase і прибираються при rebuild.

### RISK_OFF

Protective mode:

- entries заблоковані
- тільки de-risk path
- залишкові entry buys скасовуються при rebuild

Код:

- [domain/spot_grid_planner.py](./domain/spot_grid_planner.py)
- [domain/grid_builder.py](./domain/grid_builder.py)
- [domain/inventory_manager.py](./domain/inventory_manager.py)
- [domain/allocation.py](./domain/allocation.py)
- [domain/de_risk.py](./domain/de_risk.py)

Це означає:

- `BTC`, `SOL` і `XRP` now sizing іде через одну `USDT` risk model
- бот більше не залежить від одного глобального `max_inventory_units` для всіх активів
- дешеві й дорогі монети масштабуються через budget у quote currency, а не через “кількість монет”
- symbol-local sizing тепер працює під portfolio-level budget cap, а не повністю окремо

## No-Loss Sell Rule

Бот спеціально обмежений так, щоб не продавати spot inventory у збиток.

Поточна поведінка:

- він не повинен ставити `SELL` нижче ніж `cost_basis_price + 1%`
- якщо `cost_basis_price` недоступна, використовується консервативний reference
- de-risk sells теж блокуються, якщо ціна нижча за дозволений exit floor

Код:

- [domain/spot_grid_planner.py](./domain/spot_grid_planner.py)
- [domain/de_risk.py](./domain/de_risk.py)
- [infrastructure/execution_gateway.py](./infrastructure/execution_gateway.py)

Це означає:

- якщо inventory underwater, бот не форсує exit зі збитком
- він або тримає inventory, або діє згідно з поточною фазою
- якщо ринок уже перейшов у protective regime, нові buy entries лишаються заблокованими до повернення коректної recovery phase

## Underwater Averaging

У бота тепер є окремий underwater averaging path.

Поточна логіка:

- averaging не вмикається, поки просадка від `cost_basis_price` не перевищить `UNDERWATER_AVERAGING_TRIGGER_PCT`
- averaging дозволений тільки в:
  - `RANGE`
  - `UPTREND`
- averaging заблокований у:
  - `DOWNTREND`
  - `HIGH_VOLATILITY`
  - `RISK_OFF`
- якщо просадка перевищує `UNDERWATER_DEEP_STOP_PCT`, нові underwater buys знову вимикаються

Recovery budget зараз рахується від вільного `USDT`:

- база = поточний вільний `USDT`
- recovery budget = `free_quote * UNDERWATER_RECOVERY_BUDGET_PCT`
- потім він коригується за режимом:
  - `RANGE` використовує `UNDERWATER_RANGE_BUDGET_MULTIPLIER`
  - `UPTREND` використовує `UNDERWATER_UPTREND_BUDGET_MULTIPLIER`

Recovery sizing все ще обмежується:

- залишком symbol inventory room
- фактичним вільним quote balance
- venue minimum order constraints

Це означає:

- underwater averaging більше не прив’язаний до `% від total equity`
- на малому рахунку recovery budget масштабується від вільного `USDT`, і це простіше контролювати
- recovery у `RANGE` свідомо менший, ніж у `UPTREND`
- бот не усереднюється безкінечно в поганих режимах

## Cost Basis And Take-Profit Planning

`cost_basis_price` рахується з реальних Bybit spot executions:

- бот читає recent fills
- рахує середню собівартість залишкового inventory
- коротко кешує результат

Код:

- [infrastructure/execution_gateway.py](./infrastructure/execution_gateway.py)

Phase 4 додала cost-basis-aware sell planning:

- sell take-profit orders тепер не лише фільтруються після planning
- вони активно rebasing-яться від effective inventory reference price
- minimum take-profit price будується з:
  - `cost_basis_price`
  - `min_sell_markup_bps`
  - ATR-based profit floor

Код:

- [domain/cost_basis.py](./domain/cost_basis.py)
- [domain/spot_grid_planner.py](./domain/spot_grid_planner.py)

Це означає:

- sell ladders економічно узгоджені з фактичним spot inventory
- бот більше не покладається тільки на post-filtering, щоб уникати поганих exits
- underwater inventory також штрафується на рівні portfolio allocation, коли бот вирішує, чи відновлювати accumulation
- якщо underwater averaging знизив `cost_basis_price`, досягти no-loss exit надалі стає легше

## Rebuild Logic

Бот робить rebuild не тільки через рух ціни, а й через material differences між:

- `target_orders`
- `live_orders`

Rebuild також може статись, якщо:

- немає live orders
- щойно відбувся state transition
- `last_rebuild_price` відсутня
- price deviation перевищує threshold
- target orders порожні, а live orders ще існують
- символ уже в protective regime, але ще має live buy-entry orders, які треба скасувати

Код:

- [domain/order_diff.py](./domain/order_diff.py)
- [domain/spot_grid_planner.py](./domain/spot_grid_planner.py)

## Execution Logic

Bybit spot executor:

- завантажує open orders
- порівнює їх з target orders
- скасовує зайві ордери
- ставить відсутні ордери
- поважає exchange filters
- уникає duplicate `clientOrderId`

Guardrails включають:

- `max_new_orders_per_cycle`
- `max_cancel_replace_per_cycle`
- `max_total_open_orders`
- `min_level_distance_bps`
- marketable order filter
- no-loss sell filter

Код:

- [infrastructure/execution_gateway.py](./infrastructure/execution_gateway.py)

Практичний наслідок:

- стратегія може запланувати більше ордерів, ніж executor поставить за один цикл
- execution навмисно throttled заради безпеки
- коли planner прибирає entry buys у protective regime, executor sync скасовує ці stale buy orders на біржі

## Candle Ingestion And Independence

`spot_grid_bot` більше не залежить від `futures-trading-bot` для заповнення candle history.

Тепер у нього є власні:

- створення candle schema/table
- Binance Spot candle sync adapter
- scheduler hook для періодичного candle refresh
- окремий `sync` CLI mode

Це не змінює торгову стратегію, але робить проєкт операційно незалежним.

## Persistence

Бот зберігає runtime state по кожному символу в PostgreSQL:

- `regime`
- `bars_in_state`
- `cooldown_remaining`
- `volatility_cooldown_remaining`
- `pending_regime`
- `pending_count`
- `last_rebuild_price`
- `kill_switch_count`
- `recent_equity`

Код:

- [infrastructure/state_store.py](./infrastructure/state_store.py)
- [application/trading_cycle_service.py](./application/trading_cycle_service.py)

Це означає:

- після рестарту бот не втрачає runtime memory
- risk state і state machine продовжують із збережених значень
- один символ не ділить runtime state з іншим

## Backtesting And Evaluation

У проєкті є не просто мінімальний backtest loop, а окремий evaluation layer.

Backtest engine:

- симулює fills
- оновлює inventory balances
- підтримує simulated `cost_basis_price`
- відстежує realized PnL на sells
- відстежує unrealized PnL по залишковому inventory
- рахує rebuilds і de-risk events
- відстежує blocked no-loss sell situations
- відстежує частоту risk reasons
- міряє inventory utilization відносно поточної symbol-level notional cap model

Код:

- [backtesting/engine.py](./backtesting/engine.py)

Backtest result metrics включають:

- `pnl`
- `realized_pnl`
- `unrealized_pnl`
- `max_drawdown`
- `trade_count`
- `rebuild_count`
- `average_inventory_utilization`
- `de_risk_event_count`
- `blocked_no_loss_sell_count`
- `risk_reason_counts`
- `regime_statistics`
- `kill_switch_count`

Reporting helpers:

- [backtesting/reporting.py](./backtesting/reporting.py)

Доступні helpers:

- `build_backtest_summary(...)`
- `format_backtest_summary(...)`

Це дає проєкту research/evaluation loop, а не тільки live execution loop.

## Testing

У проєкті є behavior-focused тести на:

- rebuild idempotency
- execution guardrails
- no-loss de-risk behavior
- protective-regime cancellation of remaining buy-entry orders
- outstanding order exposure accounting
- percentage-based symbol allocation для дешевих і дорожчих активів
- portfolio-level budget distribution і concurrency limits
- underwater allocation penalty
- runtime state persistence і isolation
- cost-basis sell planning
- market-structure regime detection
- backtest diagnostics і reporting

Тести:

- [tests](./tests)

Запуск:

```bash
./.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

## Поточний стан

На цей момент бот:

- працює як live spot bot на Bybit
- має власний Binance Spot candle synchronization path
- створює і підтримує candle tables у `_candles_trading_data`
- читає свічки з PostgreSQL
- дотримується no-loss exit behavior
- використовує explicit `HIGH_VOLATILITY` і staged de-risk handling
- враховує outstanding buy-order exposure
- рахує нові entries через percentage-based `USDT` budgets
- розподіляє fresh entry capital через portfolio-level allocator
- скасовує stale buy-entry orders, коли символ уже в protective regime
- зберігає per-symbol runtime state
- будує cost-basis-aware take-profit sells
- використовує окрему `UPTREND` policy:
  - ATR pullback buys
  - weighted deeper buy sizing
  - overextension entry block
  - staged sell-out із більшим size на вищих рівнях
  - adaptive take-profit залежно від сили тренду
- використовує trigger-based underwater averaging:
  - recovery стартує тільки після заданої просадки від `cost_basis_price`
  - recovery budget рахується від вільного `USDT`
  - у `RANGE` recovery менший, ніж у `UPTREND`
  - у `DOWNTREND` / `RISK_OFF` / `HIGH_VOLATILITY` averaging вимкнений
- визначає режим через indicators + market structure
- має diagnostics-oriented backtesting і reporting layer

## Команди запуску

Із директорії `spot_grid_bot`:

```bash
./.venv/bin/python main.py sync --period 365 --timeframe 1h
./.venv/bin/python main.py once
./.venv/bin/python main.py live
```
