# Spot Grid Bot — Покращення торгових результатів

> Версія: 2026-05-01  
> Статус: чернетка для обговорення  
> Код не змінювати до узгодження пріоритетів

---

## Зміст

- [T1 — Якість вибору рівнів сітки](#t1--якість-вибору-рівнів-сітки)
- [T2 — Покращення визначення режиму](#t2--покращення-визначення-режиму)
- [T3 — Динамічний розмір позиції](#t3--динамічний-розмір-позиції)
- [T4 — Underwater averaging](#t4--underwater-averaging)
- [T5 — Sell-side логіка](#t5--sell-side-логіка)
- [T6 — Портфельна алокація](#t6--портфельна-алокація)
- [T7 — Часові паттерни](#t7--часові-паттерни)
- [T8 — Адаптивна геометрія сітки](#t8--адаптивна-геометрія-сітки)
- [Зведена таблиця пріоритетів](#зведена-таблиця-пріоритетів)
- [Рекомендований порядок роботи](#рекомендований-порядок-роботи)

---

## T1 — Якість вибору рівнів сітки

Поточна сітка будується рівномірно від ATR-геометрії без урахування того, де реально торгується ринок. Рівні "в повітрі" виконуються рідше і дають гірший середній fill price.

---

### T1-1. Volume Profile: рівні на POC / VAH / VAL

**Задіяні файли:** `domain/indicators.py`, `domain/grid_builder.py`, `domain/market_models.py`

**Суть ідеї.**
Point of Control (POC) — ціновий рівень із найбільшим накопиченим об'ємом за lookback-вікно. Value Area High (VAH) і Value Area Low (VAL) — верхня і нижня межі зони, де відбулось 70% торгів. Ціна природно тяжіє до POC і відштовхується від VAH/VAL.

Якщо розмістити:
- BUY рівні поблизу VAL і POC — buy fills будуть частіші, бо ціна там "зависає".
- SELL рівні поблизу VAH — fills при поверненні до верхньої зони.

Дані вже є: свічки з полем `volume` зберігаються в `_candles_trading_data`. POC рахується без нових API-запитів.

**Як рахувати POC з наявних свічок:**
1. Розбити ціновий діапазон lookback-вікна на N бакетів (наприклад, 50).
2. Для кожної свічки додати `volume` у бакет, що відповідає `(high + low) / 2`.
3. POC = центр бакета з максимальним накопиченим volume.
4. VAH, VAL — межі зони, що містить 70% total volume.

**Де інтегрувати:**
- Додати `VolumeProfile` dataclass у `domain/market_models.py`.
- Рахувати в `domain/indicators.py` поряд з EMA і ATR.
- Передавати в `GridBuilder.build_range_grid` і `build_trend_pullback_grid` як опціональний параметр.
- Якщо VAL доступний — зміщувати нижній BUY anchor до VAL замість `price - width/2`.

**Очікуваний ефект:** збільшення fill rate BUY ордерів у RANGE на 15–25%, менше "мертвих" рівнів, що ніколи не виконуються.

---

### T1-2. Swing levels у GridBuilder замість рівномірного ATR-кроку

**Задіяні файли:** `domain/grid_builder.py`, `domain/market_structure.py`, `domain/symbol_analyzer.py`

**Суть ідеї.**
`market_structure.py` вже знаходить swing highs і swing lows у кожному циклі (функції `_find_swing_highs`, `_find_swing_lows`). Але ці рівні ніде не передаються в `GridBuilder` — той продовжує будувати рівномірну ATR-решітку.

Swing lows — природна підтримка, де ймовірність відскоку вища. Swing highs — природний спротив, де ймовірність sell-fill вища. Якщо сітка будується навколо цих рівнів, ордери виконуються в більш значущих точках.

**Логіка змін:**
- У `symbol_analyzer.py` або `market_data_provider.py` — витягати список recent swing highs/lows із `detect_market_structure`.
- Передавати їх у `build_range_grid` як `support_levels: list[float]` і `resistance_levels: list[float]`.
- У `GridBuilder`: якщо swing рівень потрапляє у BUY-зону grid (`range_low` до `mid`) — "прив'язати" найближчий BUY level до нього (у межах ±0.5×ATR).
- Аналогічно для SELL у sell-зоні.
- Якщо swing рівнів немає або вони поза зоною — fallback до поточної ATR-рівномірності.

**Очікуваний ефект:** рівні сітки відповідають реальній ринковій структурі. Менше rebuild через те, що ордери "чіпляються" за рівні, які ринок поважає.

---

## T2 — Покращення визначення режиму

Поточний детектор використовує EMA-стек і market structure bias. Сильна база, але є сліпі зони: перекупленість/перепроданість у RANGE, volume-якість при переходах, внутрішньоденний контекст.

---

### T2-1. RSI як фільтр якості входу в RANGE

**Задіяні файли:** `domain/indicators.py`, `domain/range_entry_policy.py`, `domain/strategy_config.py`

**Суть ідеї.**
У RANGE-режимі бот входить у BUY рівномірно незалежно від поточної позиції ціни всередині range. RSI дає додатковий сигнал:
- RSI < 35 → oversold, ціна ближче до дна range → вища ймовірність відскоку, добрий час для BUY.
- RSI > 65 → overbought, ціна ближче до вершини range → новий BUY менш доцільний.
- 35–65 → нейтральна зона, поточна логіка без змін.

**Де інтегрувати:**
- Додати `rsi14: float` у `IndicatorSnapshot` (в `domain/market_models.py`).
- Рахувати в `domain/indicators.py` через `pandas_ta.rsi()` — бібліотека вже у `requirements.txt`.
- У `range_entry_policy.py` у функції `evaluate_range_entry_profile`:
  - RSI < 35 → `budget_penalty = 1.0` (без штрафу), можливо +10% budget bonus.
  - RSI 35–50 → `budget_penalty = 1.0` (нейтрально).
  - RSI 50–65 → `budget_penalty = 0.85` (легкий штраф).
  - RSI > 65 → `budget_penalty = range_poor_entry_budget_penalty` або `block_new_buys = True`.
- Додати `rsi_overbought_threshold: float = 65.0` і `rsi_oversold_threshold: float = 35.0` у `GridConfig`.

**Очікуваний ефект:** бот перестає купувати на вершині range. BUY відбуваються у більш вигідних точках, cost basis знижується в середньому на 1–3%.

---

### T2-2. Volume confirmation при зміні режиму

**Задіяні файли:** `domain/state_machine.py`, `domain/indicators.py`, `domain/strategy_config.py`

**Суть ідеї.**
Зараз перехід між режимами підтверджується тільки кількістю барів (`hysteresis_confirm_bars`). Low-volume breakouts часто є помилковими — ціна пробиває рівень без реального попиту і повертається назад. Якщо додатково вимагати підтвердження об'ємом, false transitions зменшаться.

**Логіка:**
- Рахувати `volume_ma20` (середній об'єм за 20 барів) у `indicators.py`.
- У `state_machine.py` при підрахунку `pending_count`: якщо поточний bar's volume < `volume_ma20 * volume_confirmation_multiplier` (наприклад, 1.2) — не збільшувати `pending_count`, навіть якщо режим відповідає pending.
- Перехід відбувається тільки коли накопичилось достатньо барів **з** нормальним об'ємом.
- Додати `volume_confirmation_multiplier: float = 1.2` у `RegimeConfig`. При `0.0` — поведінка як зараз (backward compatible).

**Очікуваний ефект:** менше хибних переходів у UPTREND/DOWNTREND, менше непотрібних rebuild після short-lived spike.

---

### T2-3. VWAP як динамічний рівень для BUY у UPTREND

**Задіяні файли:** `domain/indicators.py`, `domain/uptrend_policy.py`, `domain/market_models.py`

**Суть ідеї.**
У UPTREND бот ставить BUY на ATR pullbacks від поточної ціни (0.5×ATR, 1.0×ATR, 1.5×ATR). Але ціна у тренді часто відскакує від VWAP (денного або тижневого), а не від фіксованого ATR-рівня. Якщо VWAP знаходиться в BUY-зоні — це сильніша точка для входу.

**Логіка:**
- Рахувати `vwap_daily: float` у `indicators.py` з `(close * volume).sum() / volume.sum()` за останні 24 бари (1h × 24 = 1 день).
- Опціонально: `vwap_weekly` за 168 барів.
- У `uptrend_policy.py`: якщо VWAP потрапляє між нижнім і верхнім BUY pullback рівнями — "прив'язати" один BUY level до VWAP замість ATR-рівня.
- Якщо VWAP нижче всіх ATR pullbacks — не змінювати нічого (fallback до поточної логіки).

**Очікуваний ефект:** BUY у UPTREND виконуються у більш значущих точках. При ринку, що поважає VWAP, fill rate зростає.

---

## T3 — Динамічний розмір позиції

Поточна модель: `budget = free_quote * max_entry_pct`. Однаковий відсоток для DOGE ($0.18) і ETH ($2000) — це різний ризик при однаковому USDT.

---

### T3-1. ATR-normalized position sizing між символами

**Задіяні файли:** `domain/portfolio_allocator.py`, `domain/allocation.py`, `domain/strategy_config.py`

**Суть ідеї.**
ATR/Price (normalized ATR) відображає відносну волатильність символу. DOGE з ATR/Price = 4% несе вдвічі більший ризик на $1 ніж BTC з ATR/Price = 2%. Поточна модель цього не враховує.

Формула normalization:
```
target_atr_pct = глобальний цільовий рівень волатильності (наприклад, 2%)
scale_factor = target_atr_pct / (atr14 / price)
adjusted_budget = base_budget * scale_factor
```
Символи з вищою волатильністю отримують менший budget, з нижчою — більший. Загальний ризик на USDT вирівнюється.

**Де інтегрувати:**
- Додати `normalized_atr: float` у `SymbolAllocationInput` (в `domain/portfolio_models.py`).
- У `PortfolioAllocator._score_symbol` використовувати `scale_factor` як додатковий множник.
- Додати `target_symbol_atr_pct: float = 0.02` у `RiskConfig`. При значенні `0.0` — нормалізація вимкнена.

**Очікуваний ефект:** рівномірний ризик-на-долар між символами. Зменшення просадки при раптовому спайку волатильності у одному з символів.

---

### T3-2. Correlation-aware allocation

**Задіяні файли:** `domain/portfolio_allocator.py`, `domain/portfolio_models.py`, `domain/indicators.py`

**Суть ідеї.**
ETH, SOL, BNB у ведмедячому ринку часто ходять разом (correlation > 0.85). Якщо бот тримає всі три у рівних частках — при спільному падінні просадка потроюється. Зменшення allocation для highly correlated пар знижує фактичний ризик портфеля без зміни кількості активних символів.

**Логіка:**
- Рахувати correlation matrix між символами на основі 20-денних returns (з наявних свічок в БД).
- У `PortfolioAllocator.allocate`: якщо два символи мають `corr > 0.8` і обидва eligible — зменшити allocation слабшого (за confidence або momentum) на 30–40%.
- Correlation matrix рахується раз на цикл і зберігається у `PortfolioSnapshot`.
- Додати `correlation_penalty_threshold: float = 0.8` і `correlation_penalty_factor: float = 0.65` у `RiskConfig`.

**Очікуваний ефект:** при системному ринковому падінні портфель не "складається" одночасно по всіх позиціях. Зменшення max drawdown портфеля.

---

## T4 — Underwater averaging

Поточна логіка: фіксований бюджет від `free_quote * recovery_budget_pct`, однаковий розмір на всіх recovery рівнях. Є простір для розумнішого усереднення.

---

### T4-1. Geometric DCA — більший розмір при глибшому drawdown

**Задіяні файли:** `domain/uptrend_policy.py`, `domain/strategy_config.py`

**Суть ідеї.**
Поточна recovery логіка розподіляє budget рівномірно між `max_recovery_levels`. Але при deep drawdown (-20%) агресивніше усереднення знижує cost basis швидше і зменшує breakeven price. При shallow drawdown (-11%) агресивний DCA зайвий — ринок може відновитись сам.

**Логіка:**
- Розбити `underwater_averaging_trigger_pct` до `underwater_deep_stop_pct` на 3 зони:
  - Зона 1 (`10%–15%` drawdown): `size_multiplier = 1.0` (як зараз).
  - Зона 2 (`15%–20%` drawdown): `size_multiplier = 1.5`.
  - Зона 3 (`20%–25%` drawdown): `size_multiplier = 2.0`.
- Multiplier застосовується до базового recovery budget.
- Загальний розмір все одно обмежується `remaining_inventory_room` і `available_quote`.
- Додати `underwater_dca_zone_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0)` у `GridConfig`.

**Очікуваний ефект:** при deep drawdown cost basis знижується агресивніше → breakeven price досягається швидше при recovery.

---

### T4-2. Conditional averaging — чекати стабілізації режиму

**Задіяні файли:** `domain/uptrend_policy.py`, `domain/runtime_models.py`

**Суть ідеї.**
Зараз averaging стартує одразу при переході в RANGE або UPTREND і досягненні trigger drawdown. Але `bars_in_state = 1` (щойно перейшли) — ще невідомо, чи цей перехід не хибний. При хибному переході (швидке повернення в DOWNTREND) averaging витратить капітал у найгірший момент.

**Логіка:**
- Перевіряти `strategy_state.bars_in_state` у `build_underwater_recovery_profile`.
- Якщо `bars_in_state < min_bars_before_recovery` (наприклад, 3) → `recovery_profile.active = False`, `reason = "regime_too_new"`.
- Averaging активується тільки після кількох підтверджених барів у новому режимі.
- Додати `underwater_min_bars_in_regime: int = 3` у `GridConfig`.

**Очікуваний ефект:** менше averaging при хибних переходах між режимами. Менше випадків "усередниться під час downtrend замаскованого під range".

---

## T5 — Sell-side логіка

Поточна логіка: статична ATR-сходинка SELL від cost basis вгору. Не адаптується до поточного unrealized PnL і не trailing-ться при сильному тренді.

---

### T5-1. Trailing take-profit у UPTREND

**Задіяні файли:** `domain/uptrend_policy.py`, `domain/target_order_builder.py`, `domain/runtime_models.py`

**Суть ідеї.**
Якщо UPTREND продовжується і ціна вийшла за верхній SELL рівень — бот пропускає подальший ріст. Поточні SELL рівні статичні відносно ціни на момент rebuild. При сильному тренді верхній sell може виконатись занадто рано.

**Логіка:**
- Зберігати `last_sell_upper_price: float | None` у `StrategyState` (в `domain/runtime_models.py`).
- При rebuild у UPTREND, якщо поточна ціна > `last_sell_upper_price` — це означає, що верхній рівень вже виконався і ціна пішла далі.
- Нижній SELL anchor зсувати до `max(поточна ema20 + 1×ATR, cost_basis * 1.01)` замість фіксованого `price + 1.0×ATR`.
- Верхній SELL — `ema20 + 2.5×ATR` для сильного тренду (замість `price + 2.5×ATR`).
- EMA20 як anchor природно "trailing-ться" разом із ціною.

**Очікуваний ефект:** у сильному UPTREND бот більше фіксує прибуток на вищих рівнях замість раннього виходу.

---

### T5-2. Адаптивний sell size залежно від unrealized PnL

**Задіяні файли:** `domain/inventory_manager.py`, `domain/cost_basis.py`

**Суть ідеї.**
Поточні SELL всі мають size пропорційно до ваг (`uptrend_sell_size_weights`). При inventory в +20% unrealized profit — є сенс більш агресивно фіксувати верхні рівні. При inventory в -5% — верхній SELL все одно заблокований no-loss правилом, але нижні SELL (якщо є вільна від no-loss зона) можна зменшити.

**Логіка:**
- Рахувати `unrealized_pnl_pct = (mark_price - cost_basis_price) / cost_basis_price`.
- При `unrealized_pnl_pct > 0.15` → збільшити upper sell size weight на 20–30%.
- При `unrealized_pnl_pct < 0` → зменшити sell size (більше тримати для потенційного recovery).
- Додати `sell_size_pnl_scale_threshold: float = 0.15` і `sell_size_pnl_bonus: float = 0.25` у `GridConfig`.

**Очікуваний ефект:** при хороших позиціях бот фіксує більше прибутку. При збиткових — не розпродує inventory за безцінь.

---

## T6 — Портфельна алокація

Поточний allocator будує `weight` з confidence, inventory pressure, outstanding pressure. Не враховує відносну силу символів між собою.

---

### T6-1. Momentum ranking між символами

**Задіяні файли:** `domain/portfolio_allocator.py`, `domain/portfolio_models.py`, `domain/indicators.py`

**Суть ідеї.**
Spot grid стратегія краще працює на mean-reversion: купувати слабкі символи (відносно weak momentum) у розрахунку на відновлення, а не переплачувати за символи, що вже виросли.

**Логіка:**
- Рахувати `momentum_20d: float` для кожного символу = `(close[-1] - close[-20]) / close[-20]` із наявних свічок.
- Додати `momentum_20d` у `SymbolAllocationInput`.
- У `_score_symbol`: якщо `momentum_20d > 0.15` (символ вже зріс на 15%+) → знижувати weight на `momentum_penalty_factor` (наприклад, 0.7). Якщо `momentum_20d < -0.10` → підвищувати weight (більше можливостей для accumulation).
- Логіка inverting: більший allocation для слабших (але ще eligible) символів.
- Додати `momentum_penalty_threshold: float = 0.15` і `momentum_penalty_factor: float = 0.70` у `RiskConfig`.

**Очікуваний ефект:** capital перетікає до символів з кращим risk/reward, де ціна вже скоригувалась. Менше покупок на піках.

---

### T6-2. Розділення бюджету: нові входи vs recovery

**Задіяні файли:** `domain/portfolio_allocator.py`, `domain/strategy_config.py`

**Суть ідеї.**
Зараз `total_allocatable_quote` — один пул для всіх цілей: нові входи в clean символи і recovery для underwater. Вони змагаються за одні гроші. При великих underwater позиціях нові перспективні символи не отримують капіталу.

**Логіка:**
- Розбити `total_allocatable_quote` на два ring-fenced пула:
  - `new_entry_budget = total_allocatable_quote * new_entry_budget_fraction` (наприклад, 70%).
  - `recovery_budget = total_allocatable_quote * (1 - new_entry_budget_fraction)` (30%).
- Symbols без underwater inventory отримують із `new_entry_budget`.
- Symbols з underwater inventory → recovery budget.
- Якщо recovery символів немає — невикористаний recovery budget переходить в new_entry_budget.
- Додати `new_entry_budget_fraction: float = 0.70` у `RiskConfig`.

**Очікуваний ефект:** нові хороші можливості не блокуються через underwater averaging. Capitail allocation більш передбачуваний.

---

## T7 — Часові паттерни

---

### T7-1. Знижена активність у низьколіквідні години

**Задіяні файли:** `application/scheduler.py`, `domain/strategy_config.py`, `utils/config.py`

**Суть ідеї.**
Spot ринки значно менш ліквідні між 22:00–06:00 UTC. Spread ширший, маніпуляції частіші, fill quality гірша. Нові BUY у цей час несуть вищий ризик без відповідної компенсації.

**Логіка:**
- Додати в `StrategyConfig` або `utils/config.py`:
  ```
  LOW_LIQUIDITY_START_UTC: int = 22  # година
  LOW_LIQUIDITY_END_UTC: int = 6
  LOW_LIQUIDITY_BUDGET_FACTOR: float = 0.5
  ```
- У `TradingCycleAnalysisBatchService.analyze` або `PortfolioAllocator.allocate` перевіряти поточний UTC час.
- Якщо `low_liquidity_hours` → множити `total_allocatable_quote` на `LOW_LIQUIDITY_BUDGET_FACTOR`.
- Sell-ордери не обмежувати (виходити можна завжди).

**Очікуваний ефект:** менше poor fills у нічні години. Менший USDT-ризик під час маніпулятивних рухів із низьким об'ємом.

---

### T7-2. Calendar блокування перед макроподіями

**Задіяні файли:** `utils/config.py`, `application/trading_cycle_service.py`

**Суть ідеї.**
CPI, FOMC засідання, великі ETF flow дати — відомі заздалегідь і часто дають різкі рухи. У ці дні нові BUY мають вищий шанс виконатись саме перед spike вниз.

**Логіка:**
- Підтримувати простий файл `events.json` зі списком дат і часів заморожування:
  ```json
  [
    {"date": "2026-05-07", "start_utc": "18:00", "end_utc": "22:00", "reason": "FOMC"},
    {"date": "2026-05-13", "start_utc": "12:00", "end_utc": "16:00", "reason": "CPI"}
  ]
  ```
- У `TradingCycleAnalysisBatchService` читати файл і якщо поточний час у заблокованому вікні → `pause_entries=True` для всіх символів.
- Sell-ордери не блокувати.
- Файл оновлюється вручну або через простий скрипт-парсер economic calendar.

**Очікуваний ефект:** уникнення входів безпосередньо перед відомими high-volatility подіями.

---

## T8 — Адаптивна геометрія сітки

---

### T8-1. Адаптивний крок сітки від realized volatility

**Задіяні файли:** `domain/grid_builder.py`, `domain/indicators.py`, `domain/strategy_config.py`

**Суть ідеї.**
Поточний крок сітки: `atr14 * atr_grid_step_multiplier` (фіксований multiplier). ATR за 14 барів може відставати від поточної реальної волатильності. Якщо ринок заспокоївся (realized vol знизилась) — вузький крок дає більше fills. Якщо ринок розширився — широкий крок менше шансів перевиконатись.

**Логіка:**
- Рахувати `realized_vol_7d` як std dev повернень за 7 днів (168 барів) — вже є в `indicators.py`.
- Порівнювати з `realized_vol_30d` (720 барів, рахувати додатково).
- Обчислювати `vol_ratio = realized_vol_7d / realized_vol_30d`.
- Адаптувати крок: `effective_step_multiplier = atr_grid_step_multiplier * clamp(vol_ratio, 0.6, 1.8)`.
- При `vol_ratio < 1.0` (ринок заспокоївся) → крок зменшується → більше рівнів у tight range.
- При `vol_ratio > 1.0` (ринок розширився) → крок збільшується → рівні не зливаються.
- Додати `adaptive_step_vol_ratio_min: float = 0.6` і `adaptive_step_vol_ratio_max: float = 1.8` у `GridConfig`.

**Очікуваний ефект:** сітка автоматично адаптується до поточної volatility environment. Менше dedupe злиттів у volatile ринку, більше fills у спокійному.

---

### T8-2. Асиметрична сітка залежно від позиції ціни у range

**Задіяні файли:** `domain/grid_builder.py`, `domain/range_entry_policy.py`

**Суть ідеї.**
Поточна RANGE сітка симетрична: однакова кількість BUY і SELL рівнів. Але якщо ціна зараз у верхній половині range — BUY рівні далеко, а SELL рівні близько. В такому разі більше SELL рівнів і менше BUY рівнів логічніше (більше take-profit можливостей).

**Логіка:**
- Рахувати `price_position_in_range = (price - range_low) / (range_high - range_low)`, 0.0 = bottom, 1.0 = top.
- Якщо `price_position_in_range > 0.65` → збільшити кількість SELL рівнів на 1, зменшити BUY на 1.
- Якщо `price_position_in_range < 0.35` → збільшити BUY на 1, зменшити SELL на 1.
- Зона 0.35–0.65 → стандартна симетрична сітка.
- `range_low` і `range_high` вже доступні у `GridSpec` з `build_range_grid`.

**Очікуваний ефект:** сітка завжди відповідає актуальній позиції ціни. Менше "порожніх" рівнів, що ніколи не виконуються.

---

## Зведена таблиця пріоритетів

| ID | Назва | Impact | Складність | Нові дані потрібні? |
|---|---|---|---|---|
| T1-1 | Volume Profile (POC/VAH/VAL) | Високий | Висока | Ні (volume є в БД) |
| T1-2 | Swing levels у GridBuilder | Високий | Середня | Ні (вже рахується) |
| T2-1 | RSI фільтр у RANGE | Середній–Високий | Низька | Ні (pandas_ta є) |
| T2-2 | Volume confirmation переходів | Середній | Низька | Ні |
| T2-3 | VWAP anchor у UPTREND | Середній | Середня | Ні |
| T3-1 | ATR-normalized sizing | Високий | Середня | Ні |
| T3-2 | Correlation-aware allocation | Середній | Висока | Ні |
| T4-1 | Geometric DCA | Середній | Низька | Ні |
| T4-2 | Conditional averaging | Середній | Низька | Ні |
| T5-1 | Trailing take-profit UPTREND | Середній–Високий | Середня | Ні |
| T5-2 | Adaptive sell size від PnL | Середній | Низька | Ні |
| T6-1 | Momentum ranking | Середній | Середня | Ні |
| T6-2 | Split бюджету new vs recovery | Середній | Середня | Ні |
| T7-1 | Low-liquidity hours | Низький–Середній | Низька | Ні |
| T7-2 | Calendar блокування | Низький–Середній | Низька | Файл events.json |
| T8-1 | Adaptive step від vol ratio | Середній | Середня | Ні |
| T8-2 | Асиметрична сітка | Середній | Низька | Ні |

---

## Рекомендований порядок роботи

### Фаза 1 — Quick wins (2–3 тижні)
Мінімальна складність, максимальний impact. Жодних нових структур даних.

```
T2-1  RSI фільтр у RANGE               (~4 год)
T2-2  Volume confirmation переходів     (~3 год)
T4-1  Geometric DCA                     (~3 год)
T4-2  Conditional averaging             (~2 год)
T5-2  Adaptive sell size від PnL        (~3 год)
T8-2  Асиметрична сітка                 (~3 год)
T7-1  Low-liquidity hours               (~2 год)
```

### Фаза 2 — Structural improvements (3–4 тижні)
Потребують змін у кількох файлах і нових полів у моделях.

```
T1-2  Swing levels у GridBuilder        (~6 год)
T3-1  ATR-normalized sizing             (~5 год)
T5-1  Trailing take-profit UPTREND      (~6 год)
T6-2  Split бюджету new vs recovery     (~4 год)
T8-1  Adaptive step від vol ratio       (~4 год)
T2-3  VWAP anchor у UPTREND            (~5 год)
```

### Фаза 3 — Advanced features (4–6 тижнів)
Складніші в реалізації і потребують extensive backtesting перед live.

```
T1-1  Volume Profile (POC/VAH/VAL)      (~10 год)
T6-1  Momentum ranking                  (~6 год)
T3-2  Correlation-aware allocation      (~8 год)
T7-2  Calendar блокування               (~3 год)
```

### Правило для кожної зміни
1. Реалізувати з feature flag (параметр у `StrategyConfig`, default = вимкнено).
2. Провести backtest: порівняти `pnl`, `max_drawdown`, `rebuild_count`, `trade_count`.
3. Запустити на paper mode ≥ 7 днів.
4. Тільки потім вмикати в live з мінімальним розміром.
