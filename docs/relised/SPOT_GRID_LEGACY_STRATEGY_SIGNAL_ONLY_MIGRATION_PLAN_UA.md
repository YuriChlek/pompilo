# План перенесення legacy стратегії `spot_grid_bot` у platform-native `spot_grid`

## Мета

Перенести торгову логіку з legacy `spot_grid_bot` у platform-native модуль:

```text
bot-platform-service/src/bot_platform_service/trading_bots/spot_grid
```

Новий `spot_grid` не має відкривати позиції, створювати біржові ордери, скасовувати ордери або працювати з private exchange API. Його відповідальність - аналізувати market snapshot, runtime state і portfolio context, після чого створювати стандартизовані сигнали/події. Окремий execution service має читати ці події та відповідати за відкриття/закриття позицій.

## Поточний стан

Поточний platform-native `spot_grid` вже інтегрований з Bot Platform, але стратегія мінімальна:

- бере останній `close` як `reference_price`;
- рахує `range_low` і `range_high` по snapshot candles;
- будує симетричні buy/sell рівні навколо `reference_price`;
- конвертує рівні у `BotSignal`;
- працює в `signal_only`;
- не враховує regime, RSI, ATR, balance, позиції, cost basis, no-loss, portfolio allocation і de-risk.

Legacy `spot_grid_bot` має багатшу стратегію:

- market regime: `RANGE`, `UPTREND`, `DOWNTREND`, `HIGH_VOLATILITY`, `RISK_OFF`;
- indicators: EMA20/EMA50/EMA200, ATR14, RSI14, realized volatility, volume, market structure;
- range/uptrend grid construction;
- RSI-фільтри входу/виходу;
- no-loss sell policy;
- underwater averaging;
- de-risk;
- portfolio allocation;
- exchange constraints;
- rebuild policy;
- runtime state.

## Головний архітектурний принцип

Bot Platform залишається signal-only.

Заборонено переносити в `bot-platform-service` такі legacy частини:

- Bybit private execution clients;
- створення, скасування або синхронізацію біржових ордерів;
- пряме відкриття/закриття позицій;
- standalone scheduler;
- candle sync з legacy bot;
- Telegram/notification side effects поза platform ports;
- root-level imports із `spot_grid_bot`.

Дозволено переносити тільки pure strategy logic і platform-safe application orchestration.

## Додаткові правила міграції

### Не переносити legacy файли 1-в-1

Міграція має переносити поведінку стратегії, а не механічно копіювати legacy файли.

Кожен legacy модуль треба адаптувати до platform-native контрактів:

- використовувати `BotMarketSnapshot` замість legacy candle tables;
- використовувати `BotSignal` і `payload_json` замість target exchange orders;
- використовувати `Decimal` для price, quantity, notional, confidence і volume;
- нормалізувати symbols у формат `BTCUSDT`;
- прибрати `os.getenv` із domain logic;
- винести bot-owned strategy defaults у package-local `bot_config.py`;
- використовувати готові analytical libraries для індикаторів, якщо це надійніше за власну реалізацію;
- прибрати залежність від root-level `spot_grid_bot`;
- відокремити strategy decision від execution details.

Legacy назви можна зберігати тільки тоді, коли вони не спотворюють нову відповідальність. Наприклад, `grid_builder.py` доречний, але `target_order_builder.py` у platform-native модулі має стати `target_intent_builder.py`, бо результатом є execution-neutral intent, а не біржовий order.

### Конфігурація бота

Кожен platform-native бот, який має strategy parameters, повинен мати власний Python-файл:

```text
trading_bots/<module_id>/bot_config.py
```

Для `spot_grid` це:

```text
bot-platform-service/src/bot_platform_service/trading_bots/spot_grid/bot_config.py
```

`bot_config.py` має містити typed defaults, dataclasses, bounds і parser/merge helpers для strategy config. Він не має читати `.env`, process env, Docker env, файли, БД, Redis або exchange API. Runtime значення для конкретного instance беруться з persisted `BotInstanceConfig.config`, валідуються через `config_schema.py`, після чого adapter/application layer merge-ить їх із defaults із `bot_config.py` і передає resolved config у domain planner.

Секрети не належать у `bot_config.py`. Якщо майбутній execution service потребує account credentials, bot config/schema може містити тільки `secret_ref`, а raw secret має читатися execution service через власний secret boundary.

### Бібліотеки для індикаторів і time-series

Не потрібно писати власні EMA/ATR/RSI/volatility реалізації з нуля, якщо є придатна готова бібліотека. `spot_grid` може використовувати `pandas`, `pandas-ta`, `ta`, NumPy-based helpers або інші focused analytical packages, якщо це спрощує код і робить поведінку більш перевірюваною.

Правила:

- залежність має бути явно додана в dependency metadata сервісу;
- heavy analytical libraries не можна імпортувати з `manifest.py` або `config_schema.py`;
- provider, DB, Redis, network і exchange SDK не мають потрапляти в pure strategy code;
- outputs бібліотеки треба нормалізувати на boundary назад у platform/domain models;
- trading values у domain-facing моделях залишаються `Decimal`;
- для кожного перенесеного індикатора потрібні fixture tests, щоб оновлення бібліотеки не змінило strategy behavior непомітно.

### Не створювати execution service всередині `spot_grid`

Навіть якщо legacy `spot_grid_bot` має `execution_service.py`, у новому `spot_grid` не можна переносити execution orchestration як частину bot module.

У platform-native модулі application layer має відповідати за один planning cycle:

- зібрати platform context;
- викликати domain planner;
- сформувати strategy intents;
- конвертувати intents у `BotSignal`;
- повернути `BotRunResult`.

Відкриття/закриття позицій, order sizing з live balance, private exchange calls, fill reconciliation, cancel/replace і execution audit мають жити в окремому execution service за межами `trading_bots/spot_grid`.

## Цільовий потік

```text
market_data candles_collected event
  -> bot-platform event consumer
  -> enabled spot_grid instance
  -> platform market data context + persisted instance config
  -> spot_grid strategy planner
  -> BotSignal / position intent payload
  -> bot_signal.persisted.v1 event
  -> execution service
  -> exchange/order/position lifecycle
```

`spot_grid` відповідає тільки за рішення:

```text
що зробити, по якому символу, чому, з якою впевненістю, на якому рівні ціни, з якими ризиковими обмеженнями
```

Execution service відповідає за дію:

```text
чи відкривати позицію, яким ордером, яким розміром, на якій біржі, як обробити fills, retries, cancel/replace і закриття
```

## Цільовий контракт сигналу

На першому етапі варто використовувати існуючий `BotSignal` і розширити `payload_json`, не змінюючи базову таблицю `_bot_platform.bot_signals`.

Рекомендований `payload_schema`:

```text
spot_grid.position_intent
```

Рекомендована версія:

```text
payload_schema_version = 1
```

Приклад payload для входу:

```json
{
  "intent_type": "open_position",
  "execution_intent": "limit_entry_candidate",
  "strategy": "spot_grid",
  "regime": "range",
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "reference_price": "64500.00",
  "target_price": "63210.00",
  "price_band": {
    "range_low": "62000.00",
    "range_high": "66000.00"
  },
  "risk": {
    "max_position_fraction": "0.10",
    "suggested_quote_notional": "25.00",
    "max_quote_notional": "50.00"
  },
  "guards": {
    "rsi14": "31.5",
    "atr14": "420.0",
    "buy_allowed": true,
    "sell_allowed": false,
    "no_loss_required": true
  },
  "reason_codes": [
    "range_buy",
    "rsi_oversold",
    "portfolio_budget_available"
  ]
}
```

Приклад payload для виходу:

```json
{
  "intent_type": "close_position",
  "execution_intent": "limit_exit_candidate",
  "strategy": "spot_grid",
  "regime": "range",
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "reference_price": "64500.00",
  "target_price": "66800.00",
  "position": {
    "cost_basis": "63000.00",
    "min_no_loss_exit_price": "63378.00"
  },
  "guards": {
    "rsi14": "68.0",
    "sell_allowed": true,
    "no_loss_passed": true
  },
  "reason_codes": [
    "range_take_profit",
    "rsi_overbought",
    "no_loss_passed"
  ]
}
```

Мапінг на `BotSignal`:

| Strategy decision | `signal_type` | `side` | Execution meaning |
|---|---|---|---|
| відкрити/докупити spot позицію | `entry` | `buy` | execution service може створити buy intent/order |
| закрити/скоротити spot позицію | `exit` | `sell` | execution service може створити sell intent/order |
| скоротити ризик без звичайного take-profit | `rebalance` або `exit` | `sell` | execution service застосовує de-risk policy |
| нічого не робити | `hold` | `null` | execution service ігнорує для execution |
| ризикове попередження | `alert` | `null` | execution service не відкриває позицію |

## Межа між `spot_grid` і execution service

`spot_grid` має:

- аналізувати market snapshot;
- читати resolved bot config, переданий платформою або adapter application layer;
- визначати regime;
- рахувати індикатори;
- будувати target price levels;
- рахувати рекомендований notional/weight;
- перевіряти strategy guardrails;
- створювати deterministic `BotSignal`;
- пояснювати рішення через `reason`, `payload_json`, diagnostics і state changes.

Execution service має:

- читати persisted signal event `bot_signal.persisted.v1`;
- використовувати event як lightweight notification, а full payload дочитувати з `_bot_platform.bot_signals.payload_json` по `signal_id`;
- перевіряти актуальність сигналу;
- перевіряти permissions/secrets;
- отримувати live balances, positions, open orders і venue constraints;
- остаточно рахувати quantity;
- відкривати/закривати позицію;
- обробляти idempotency, retries, fills, cancel/replace;
- вести execution audit;
- оновлювати позиційний state.

Якщо execution service не може виконати сигнал, це не має робити run `spot_grid` failed. Це окремий execution result.

## Цільова структура модуля

```text
bot-platform-service/src/bot_platform_service/trading_bots/spot_grid/
├── adapter.py
├── bot_config.py
├── config_schema.py
├── manifest.py
├── domain/
│   ├── models.py
│   ├── indicators.py
│   ├── market_structure.py
│   ├── regime_detector.py
│   ├── state_machine.py
│   ├── grid_builder.py
│   ├── range_entry_policy.py
│   ├── uptrend_policy.py
│   ├── risk_manager.py
│   ├── portfolio_allocator.py
│   ├── inventory_manager.py
│   ├── de_risk.py
│   ├── cost_basis.py
│   ├── target_intent_builder.py
│   └── grid_planner.py
├── application/
│   ├── ports.py
│   ├── context_builder_service.py
│   └── trading_cycle_service.py
└── infrastructure/
    └── platform_snapshot_adapter.py
```

Канонічна назва planner-файлу для platform-native модуля - `grid_planner.py`.
Не створювати паралельний `spot_grid_planner.py`, щоб не розділити ownership planner logic між двома файлами.

`target_order_builder.py` з legacy краще переносити як `target_intent_builder.py`, бо platform-native модуль не будує біржові ордери. Він будує execution-neutral target intents.

## Фази реалізації

### Фаза 0. Передати instance config у run contract

Завдання:

- додати до run contract доступ до persisted `BotInstanceConfig.config` або еквівалентного immutable config snapshot;
- передавати цей config snapshot у `dry_run` і `run_once`;
- оновити contract tests для run request.

Критерії готовності:

- bot adapter може отримати config конкретного instance під час run;
- backward-compatible tests для існуючих bot modules проходять;
- strategy defaults більше не є єдиним джерелом runtime config.

### Фаза 1. Додати bot-owned config файл

Завдання:

- додати `spot_grid/bot_config.py` з typed defaults, bounds і parser/merge helpers;
- заборонити `os.getenv` і env-derived strategy defaults у bot-owned config;
- синхронізувати `config_schema.py` з defaults/bounds із `bot_config.py`.

Критерії готовності:

- resolved config будується як `bot_config.py` defaults плюс persisted instance overrides;
- тести падають, якщо `bot_config.py` читає env, створює clients або імпортує legacy root package;
- `config_schema.py` залишається lightweight-import safe.

### Фаза 2. Передати повний market data context у `spot_grid`

Завдання:

- змінити `SpotGridTradingCycleService`, щоб він приймав `BotMarketDataContext` або module-local context із primary/supporting snapshots;
- оновити adapter, щоб він не відкидав `supporting_snapshots`;
- додати tests для primary `1h` і supporting `4h`.

Критерії готовності:

- planner input містить primary snapshot і supporting snapshots;
- поточний single-timeframe сценарій лишається робочим;
- tests фіксують, що supporting timeframe доходить до application layer.

### Фаза 3. Зафіксувати dry-run і preview semantics

Завдання:

- оновити `BotModule` contract/docstrings так, щоб `dry_run` не обіцяв zero-persistence behavior на рівні bot module;
- зафіксувати, що platform `dry_run` є orchestration-controlled режимом і може мати persistence side effects, якщо orchestration layer так налаштований;
- додати окремий zero-persistence preview path для operator validation або явно описати існуючий contract, якщо такий path уже є;
- додати contract tests і documentation guard проти обіцянки “dry_run завжди без persistence”.

Критерії готовності:

- rollout-перевірка має чіткий спосіб запуску без persistence;
- `dry_run` semantics узгоджені між `BotModule` contract/docstrings, документацією, tests і orchestration;
- bot module не відповідає за platform persistence policy.

### Фаза 4. Зафіксувати source і execution safety boundaries

Завдання:

- заборонити imports із legacy `spot_grid_bot` у platform-native `spot_grid`;
- заборонити private exchange clients і order creation terms у `trading_bots/spot_grid`;
- зафіксувати, що `start()` не запускає long-running execution.

Критерії готовності:

- source-boundary tests падають на legacy imports;
- source-boundary tests падають на private exchange/order execution references;
- `run_once` створює тільки signals/state_changes.

### Фаза 5. Описати `spot_grid.position_intent` contract

Завдання:

- описати JSON-safe payload schema `spot_grid.position_intent`;
- зафіксувати `payload_schema_version = 1`;
- описати, що `bot_signal.persisted.v1` містить metadata і `signal_id`, а не full payload.

Критерії готовності:

- contract tests падають, якщо payload schema змінено без bump версії;
- execution service contract знає, що full payload треба читати з `_bot_platform.bot_signals.payload_json`;
- приклади entry/exit payload відповідають `BotSignal` constraints.

### Фаза 6. Ввести platform-safe strategy models

Завдання:

- перенести або перепроєктувати platform-safe моделі зі `spot_grid_bot/domain/*_models.py`;
- відділити execution/order-specific legacy моделі від strategy intent моделей;
- нормалізувати symbols у формат `BTCUSDT`.

Критерії готовності:

- domain models не імпортують DB, exchange або platform infrastructure;
- execution/order-specific поля не потрапляють у strategy models;
- unit tests покривають базову серіалізацію моделей.

### Фаза 7. Перевести trading values на `Decimal`

Завдання:

- замінити `float` на `Decimal` у price/quantity/notional/confidence/volume полях;
- додати boundary conversion для JSON-safe payload;
- додати tests для Decimal serialization.

Критерії готовності:

- всі domain-facing trading values використовують `Decimal`;
- payload serialization не втрачає precision через float;
- tests падають на float у критичних strategy DTOs.

### Фаза 8. Вибрати indicator library

Завдання:

- оцінити `pandas`, `pandas-ta`, `ta` або аналог для EMA/ATR/RSI/volatility;
- додати обрану бібліотеку в dependency metadata, якщо вона потрібна;
- зафіксувати boundary rule для нормалізації library outputs у domain DTOs.

Критерії готовності:

- вибір бібліотеки задокументований;
- dependency явно оголошена або custom implementation обґрунтовано залишена;
- heavy analytical libraries не імпортуються з `manifest.py` або `config_schema.py`.

### Фаза 9. Додати candle-to-indicator adapter

Завдання:

- адаптувати `BotMarketSnapshot.candles` до input формату обраної indicator library або internal calculator;
- нормалізувати timestamps, OHLCV і symbol/timeframe metadata;
- додати fixture tests для adapter output.

Критерії готовності:

- adapter не читає legacy candle tables;
- ordered closed candles з platform snapshot коректно перетворюються в indicator input;
- tests покривають empty/short candle history.

### Фаза 10. Реалізувати EMA/ATR/RSI

Завдання:

- додати EMA20/EMA50/EMA200;
- додати ATR14;
- додати RSI14.

Критерії готовності:

- fixture candles дають стабільні expected indicator values;
- outputs нормалізуються в `Decimal`/domain DTOs на boundary;
- dependency upgrade не може непомітно змінити indicator output без test failure.

### Фаза 11. Додати volatility і volume metrics

Завдання:

- додати realized volatility;
- додати volume metrics;
- включити metrics у indicator snapshot.

Критерії готовності:

- metrics deterministic для fixture candles;
- short history має documented fallback;
- diagnostics показують достатність/недостатність даних.

### Фаза 12. Додати market structure

Завдання:

- додати swing high/low;
- додати range position;
- додати local support/resistance candidates.

Критерії готовності:

- market structure не залежить від legacy candle tables;
- fixture scenarios покривають range і breakout shape;
- output придатний для grid builder.

### Фаза 13. Додати single-timeframe regime detection

Завдання:

- перенести або адаптувати `regime_detector`;
- визначати `RANGE`, `UPTREND`, `DOWNTREND`, `HIGH_VOLATILITY`, `RISK_OFF`;
- додати regime diagnostics.

Критерії готовності:

- regime визначається deterministically;
- `DOWNTREND` не створює entry signals;
- diagnostics пояснюють regime decision.

### Фаза 14. Додати regime state machine

Завдання:

- перенести або адаптувати `state_machine`;
- зберігати regime state per `symbol/timeframe`;
- зменшити noise між близькими regime states через hysteresis/cooldown.

Критерії готовності:

- state changes містять regime state;
- repeated snapshots не спричиняють нестабільне перемикання regime;
- stale state не блокує нові snapshots.

### Фаза 15. Додати multi-timeframe confirmation

Завдання:

- використовувати supporting `4h` snapshot для підтвердження primary `1h`;
- блокувати new buy intent, якщо supporting `4h` показує downtrend;
- додати reason codes для multi-timeframe block.

Критерії готовності:

- supporting timeframe може заблокувати buy intent;
- відсутній supporting snapshot має documented conservative fallback;
- diagnostics показують primary і supporting regime.

### Фаза 16. Ввести `TargetIntent`

Завдання:

- замінити legacy `TargetOrder` concept на execution-neutral `TargetIntent`;
- описати intent fields для entry, exit, rebalance, hold і alert;
- заборонити exchange order id і private execution details в intent.

Критерії готовності:

- intent не містить venue order IDs або fill state;
- intent має `intent_type`, `execution_intent`, target price і reason codes;
- intent можна конвертувати в `BotSignal` без execution API.

### Фаза 17. Додати range grid builder

Завдання:

- перенести range grid construction;
- створювати buy intents нижче reference price;
- створювати sell intents вище reference price.

Критерії готовності:

- `RANGE` scenario створює очікувані buy/sell intents;
- кожен intent має reason codes;
- grid levels не порушують basic price sanity checks.

### Фаза 18. Додати uptrend grid builder

Завдання:

- перенести uptrend pullback grid construction;
- використовувати ATR step;
- використовувати local support/resistance alignment.

Критерії готовності:

- `UPTREND` створює buy intents тільки на pullback-рівнях;
- ATR step впливає на spacing;
- support/resistance alignment має fixture coverage.

### Фаза 19. Додати RSI guardrails

Завдання:

- додати RSI buy rule: entry тільки якщо `RSI <= 35`;
- додати RSI sell rule: звичайний exit тільки якщо `RSI > 65`;
- додати diagnostics для blocked RSI decisions.

Критерії готовності:

- RSI buy/sell filters мають unit tests;
- заблоковані intents пояснюються reason codes;
- de-risk logic ще не обходить ці правила до окремої фази.

### Фаза 20. Додати no-loss sell policy

Завдання:

- додати cost basis input у strategy context;
- додати min exit price з fee/slippage/profit markup;
- блокувати звичайні sell intents, якщо cost basis невідомий.

Критерії готовності:

- sell intent нижче no-loss threshold не створюється;
- unknown cost basis блокує звичайні sell intents;
- diagnostics пояснюють no-loss decision.

### Фаза 21. Додати price distance і volatility pause

Завдання:

- додати захист від buy/sell рівнів занадто близько до reference price;
- додати cooldown/high-volatility entry pause;
- додати state fields для cooldown.

Критерії готовності:

- занадто близькі рівні блокуються;
- high-volatility pause блокує new entry intents;
- cooldown state зберігається через `BotStateChange`.

### Фаза 22. Описати portfolio і position input contract

Завдання:

- описати `PortfolioContext` і `PositionContext` як explicit platform input/port;
- визначити conservative empty context fallback;
- заборонити direct exchange query всередині bot module.

Критерії готовності:

- bot може працювати без portfolio context у conservative mode;
- portfolio/position provider не є private exchange client всередині `spot_grid`;
- tests покривають empty context.

### Фаза 23. Додати exposure calculations

Завдання:

- перенести exposure calculations;
- рахувати per-symbol exposure;
- рахувати portfolio-level exposure.

Критерії готовності:

- exposure deterministic для fixture portfolio;
- exposure не читає exchange напряму;
- diagnostics містять per-symbol і portfolio exposure.

### Фаза 24. Додати portfolio allocator

Завдання:

- перенести portfolio allocator;
- застосувати max position fraction;
- додати per-symbol і portfolio-level budget caps.

Критерії готовності:

- `max_position_fraction` реально впливає на suggested notional;
- intent payload містить `suggested_quote_notional` і `max_quote_notional`;
- execution service може зменшити/відхилити notional без зміни signal contract.

### Фаза 25. Додати underwater detection

Завдання:

- перенести underwater position detection;
- визначати recovery eligibility;
- додати reason codes для underwater/recovery state.

Критерії готовності:

- underwater state визначається deterministically;
- recovery eligibility не створює intent без guardrails;
- diagnostics пояснюють underwater decision.

### Фаза 26. Додати recovery averaging intents

Завдання:

- перенести recovery budget limits;
- створювати recovery buy intents тільки при виконанні guardrails;
- блокувати averaging у downtrend/high-volatility/risk-off без явного дозволу.

Критерії готовності:

- recovery buy intent має reason codes і budget fields;
- averaging блокується в заборонених regimes;
- intent не містить private execution details.

### Фаза 27. Додати de-risk intents

Завдання:

- перенести de-risk order builder як de-risk intent builder;
- дозволити de-risk sell intent у risk-off;
- явно відділити de-risk від звичайного take-profit exit.

Критерії готовності:

- de-risk sell може обходити RSI sell rule;
- de-risk не обходить execution safety, бо execution service повторно перевіряє risk;
- de-risk intent має окремі reason codes.

### Фаза 28. Конвертувати `TargetIntent` у `BotSignal`

Завдання:

- замінити поточний `_signal_from_level` на conversion з `TargetIntent`;
- мапити entry/exit/rebalance/hold/alert у `BotSignalType`;
- використовувати `payload_schema = "spot_grid.position_intent"` і `payload_schema_version = 1`.

Критерії готовності:

- signal payload достатній для execution service;
- `BotSignal.build` створює deterministic `signal_key`;
- старий `spot_grid.grid_level` лишається тільки rollback compatibility, якщо це потрібно.

### Фаза 29. Перевірити persisted signal event contract

Завдання:

- перевірити, що `bot_signal.persisted.v1` містить persisted `signal_id`, schema/hash і routing metadata;
- перевірити, що full payload дочитується з `_bot_platform.bot_signals.payload_json`;
- додати integration tests з persisted signals.

Критерії готовності:

- повторна доставка того самого snapshot не створює дублікати persisted signal;
- execution service contract не очікує full payload у Redis event;
- persisted signal можна знайти по `signal_id`.

### Фаза 30. Додати strategy state persistence

Завдання:

- зберігати last strategy plan у `BotStateChange`;
- зберігати regime state per `symbol/timeframe`;
- зберігати cooldown counters.

Критерії готовності:

- state changes застосовуються idempotently;
- state keys deterministic;
- failed execution service не ламає strategy run.

### Фаза 31. Додати semantic intent idempotency

Завдання:

- зберігати last emitted intent hash;
- не дублювати semantic intents між різними snapshots, якщо strategy state не змінився;
- не покладатися на `BotSignal.signal_key` для cross-snapshot semantic dedupe.

Критерії готовності:

- новий snapshot із тим самим semantic intent блокується через explicit `intent_hash`/state gate;
- зміна payload schema не ламає старі persisted signals;
- stale state не блокує нові materially different intents.

### Фаза 32. Додати legacy fixture comparison harness

Завдання:

- зробити fixture runner, який подає однакові candles/context у legacy і platform planner;
- порівнювати regime, indicators, target prices і reason codes;
- не порівнювати live exchange side effects.

Критерії готовності:

- fixture harness не імпортує legacy runtime у platform production code;
- мінімум базові range/uptrend/downtrend scenarios проходять;
- відхилення documented.

### Фаза 33. Розширити legacy comparison scenarios

Завдання:

- додати high volatility scenario;
- додати underwater scenario;
- додати no-loss block scenario.

Критерії готовності:

- мінімум 10 fixture scenarios проходять;
- кожне відхилення має documented reason;
- platform planner не залежить від legacy imports.

### Фаза 34. Cleanup `spot_grid`

Завдання:

- прибрати dead code, тимчасові helpers і невикористані compatibility paths у `trading_bots/spot_grid`;
- видалити або явно позначити rollback-only support для старого `spot_grid.grid_level`;
- перевірити відповідність фактичної структури цільовій структурі модуля;
- оновити module-local docs/comments, якщо вони не відповідають новому intent flow.

Критерії готовності:

- у `spot_grid` немає unused legacy imports, exchange/private client references, env reads або order execution code;
- rollback-only code, якщо лишається, має явний тест і documented removal condition;
- source-boundary tests проходять по всій папці `trading_bots/spot_grid`;
- public schema/config/docs відповідають фактичному коду.

### Фаза 35. Rollout у zero-persistence preview

Завдання:

- запустити planning через zero-persistence preview path;
- перевірити signals/state diagnostics без persisted signal side effects;
- перевірити subscribed symbols.

Критерії готовності:

- preview показує очікувані intents по configured symbols;
- preview не створює persisted signals;
- rollback не потрібен, бо production state не змінюється.

### Фаза 36. Rollout у `dry_run`

Завдання:

- увімкнути `dry_run` instance;
- перевірити contract-defined dry-run persistence behavior;
- перевірити diagnostics і state changes.

Критерії готовності:

- `spot_grid` dry-run instance працює з real platform context;
- persistence behavior відповідає `BotModule` contract/docstrings, orchestration tests і документації;
- Bot Platform не відкриває позиції.

### Фаза 37. Rollout у `notification_only`

Завдання:

- увімкнути `notification_only`;
- перевірити operator-visible diagnostics/notifications;
- не підключати execution service до signals на цьому кроці.

Критерії готовності:

- notifications не містять secrets або private execution data;
- signals/diagnostics створюються по subscribed symbols;
- rollback = disable platform instance.

### Фаза 38. Rollout у `signal_only`

Завдання:

- увімкнути `signal_only`;
- перевірити persisted signals і `bot_signal.persisted.v1`;
- залишити execution service disconnected.

Критерії готовності:

- persisted signals створюються по всіх subscribed symbols;
- signal event stream стабільний;
- Bot Platform не відкриває позиції.

### Фаза 39. Підключити execution service

Завдання:

- підключити execution service тільки до stable persisted signal stream;
- додати kill switch на рівні execution service;
- залишити legacy bot як rollback reference до завершення rollout.

Критерії готовності:

- execution service читає signal event і full payload із `_bot_platform.bot_signals`;
- global kill switch і per-symbol/bot pause працюють;
- rollback = disable platform instance або відключити execution consumer.

## Потрібні зміни в execution service

Цей план передбачає окремий сервіс виконання позицій. Для нього потрібні такі можливості:

- consumer для `bot_signal.persisted.v1`;
- читання `_bot_platform.bot_signals.payload_json`, бо Redis event не містить full payload;
- idempotency по `signal_id` або `signal_key`;
- permission check для bot instance;
- secret refs для exchange account;
- live balance/position/open orders provider;
- venue constraints provider;
- order sizing;
- order placement/cancel/replace;
- fill reconciliation;
- execution audit tables;
- status events: accepted, rejected, placed, filled, cancelled, failed;
- global kill switch;
- per-symbol/bot execution pause.

Execution service не має довіряти сигналу повністю. Він повинен повторно перевіряти live risk, баланс, venue constraints і stale snapshot.

## Тестовий план

Мінімальний набір тестів:

- unit tests для indicators;
- fixture tests, які фіксують outputs обраної indicator library;
- unit tests для regime detector;
- unit tests для range/uptrend grid builder;
- unit tests для RSI buy/sell filters;
- unit tests для no-loss policy;
- unit tests для de-risk/recovery intents;
- unit tests для portfolio allocation;
- contract tests для `spot_grid.position_intent`;
- adapter tests для `BotSignal`;
- idempotency tests для duplicate snapshot event;
- semantic intent hash tests для різних snapshots з однаковим strategy intent;
- bot_config tests проти env reads і side effects;
- source-boundary tests проти legacy imports і exchange clients;
- integration tests з persisted signals;
- smoke test: market-data event -> spot_grid signal -> signal event published.

Команди перевірки після кожної фази:

```bash
bot-platform-service/.venv/bin/python -m pytest bot-platform-service/tests/unit
bot-platform-service/.venv/bin/python -m pytest bot-platform-service/tests
```

Якщо змінюється market-data context або event flow:

```bash
market-data-service/.venv/bin/python -m pytest market-data-service/tests
```

## Ризики

- Legacy стратегія частково використовує `float`; під час перенесення треба перейти на `Decimal`.
- Legacy planner змішував strategy target orders і execution-specific order details; у platform-native модулі це треба розділити.
- No-loss logic потребує достовірного cost basis, якого може не бути у market snapshot.
- Portfolio allocation потребує позиційного контексту, який зараз не є частиною `BotMarketDataContext`.
- Поточний run contract має отримати persisted instance config, інакше bot-owned defaults/overrides не впливатимуть на фактичний run.
- Execution service має мати власний risk gate, бо signal може застаріти між створенням і виконанням.
- Надто часті snapshots можуть створювати багато близьких intent signals; потрібна semantic idempotency/rebuild policy.

## Рекомендований порядок роботи

Виконувати фази 0-39 у наведеному порядку. Кожна фаза має завершуватися working tests і не повинна починатися, якщо попередня фаза залишила відкритий contract або boundary gap.

Це дозволяє поступово покращувати стратегію, не порушуючи головну межу: `spot_grid` генерує подію, а позиції відкриває і закриває окремий execution service.
