# План покращення прибутковості futures-trading-bot

---

## Попереднє: Baseline бектест (обов'язково перед будь-якими змінами)

Перш ніж щось міняти, потрібен baseline — без нього неможливо виміряти ефект кожної фази.

```bash
mkdir -p artifacts

PYTHONPATH=. ./.venv/bin/python backtesting/run_backtest.py \
  --symbol SOLUSDT ETHUSDT AVAXUSDT DOGEUSDT LINKUSDT \
  --from 2025-11-01 \
  --to 2026-05-01 \
  --output-json artifacts/baseline.json
```

### Ключові метрики для аналізу після кожної фази

| Метрика | Що означає |
|---|---|
| `exit_reason_counts.stop_loss / total_trades` | > 70% = проблема зі стопами або якістю входу |
| `net_pnl_pct` | Загальний результат |
| `expectancy_by_setup` | Порівняти `breakout_close` vs `breakout_reclaim` окремо |
| `pnl_by_regime` | Перевірити чи `bull_trend` прибутковий |
| `skipped_signal_counts` | Що блокує більшість входів |

---

## Фаза 1 — Виправлення entry-фільтрів

### 1.1 Відокремити volume-фільтр для `breakout_reclaim`

**Файл:** `trading/domain/signal_generation.py`

**Проблема:**
В `_build_breakout_context()` перевірка `spike_ratio < _required_volume_ratio(...)` блокує обидва
сетапи разом. Але природа reclaim-свічки — повернення на низькому об'ємі після імпульсного пробою.
Вимагати volume spike на reclaim-свічці = відрізати переважну більшість валідних reclaim-входів.

**Рішення:**

1. Видалити volume check з `_build_breakout_context()`.
2. Додати volume check всередині `_build_breakout_close_signal()` — тільки для close breakout.
3. Для `_build_breakout_reclaim_signal()` додати тільки м'який поріг (уникати "мертвих" годин).

```python
# _build_breakout_context() — ВИДАЛИТИ цей блок:
# spike_ratio = ...
# if spike_ratio < _required_volume_ratio(...):
#     return None

# _build_breakout_close_signal() — ДОДАТИ:
volume_analysis = trend_data.volume_analysis or {}
spike_ratio = _as_decimal(volume_analysis.get("spike_ratio"), default=Decimal("0"))
if spike_ratio < _required_volume_ratio(breakout_context["trend_strength"], strategy_config):
    return None

# _build_breakout_reclaim_signal() — ДОДАТИ м'який поріг:
volume_analysis = trend_data.volume_analysis or {}
spike_ratio = _as_decimal(volume_analysis.get("spike_ratio"), default=Decimal("0"))
if spike_ratio < Decimal("0.70"):  # уникати "мертвого" часу без ліквідності
    return None
```

---

### 1.2 Збільшити `lookback_candles`: 20 → 40

**Файл:** `trading/domain/strategy_config.py`

**Проблема:**
`lookback_candles = 20` на H1 = 20 годин. Рівень, сформований за 20 годин — це просто
"вчорашній high/low". Він постійно ламається і відразу reverses.
40 H1 свічок ≈ 2 бізнес-дні — більш значуща структура.

**Рішення:**
```python
lookback_candles: int = 40  # було 20
```

Ефект: менша кількість сигналів (рідші пробої значущіших рівнів), але вищий win rate.

---

### 1.3 Оновити `indicator_history_period` в BacktestConfig

**Файл:** `backtesting/models.py`

**Проблема:**
Поточне `indicator_history_period = 25`. В `_build_breakout_context()` перевіряється:
```python
if len(history_rows) < strategy_config.breakout.lookback_candles:
    return None
```
При `lookback_candles = 40` і `indicator_history_period = 25` — сигналів у бектесті не буде
взагалі, бо history завжди коротша ніж потрібно.

**Рішення:**
```python
indicator_history_period: int = 50  # було 25, мінімум = lookback_candles + 10
```

---

### 1.4 Додати MFI-фільтр (уникати входів в перекупленість/перепроданість)

**Файл:** `trading/domain/signal_generation.py`

**Проблема:**
`MFIAnalyzer` рахується кожен цикл і передається в `TrendResult.mfi_signal` як
`'overbought'` / `'oversold'` / `'neutral'`, але в `signal_generation.py` він ніде
не використовується. Входити в long коли MFI > 70 = купувати на виснаженні.
Це класичний тип false breakout.

**Рішення:**
Додати в `_build_breakout_context()` після визначення `regime`:

```python
# MFI exhaustion filter
mfi_signal = str(getattr(trend_data, "mfi_signal", "neutral"))
if regime.direction == BUY_DIRECTION and mfi_signal == "overbought":
    return None  # не купувати в перекупленість
if regime.direction == SELL_DIRECTION and mfi_signal == "oversold":
    return None  # не продавати в перепроданість
```

---

### 1.5 Додати CVD-підтвердження напрямку

**Файл:** `trading/domain/signal_generation.py` + `trading/domain/strategy_config.py`

**Проблема:**
`CVDAnalyzer` рахує trend, strength і confidence кожен цикл, але вони ніколи не
перевіряються в signal generation. CVD-дивергенція (ціна робить new high, але CVD —
ні) — один з найнадійніших сигналів слабкого пробою.

**Рішення:**
Додати конфіг-параметр в `BreakoutTrendStrategyConfig`:
```python
require_cvd_alignment: bool = True
```

Додати перевірку в `_build_breakout_context()` після MFI-фільтру:
```python
# CVD directional confirmation
if strategy_config.breakout.require_cvd_alignment:
    cvd_analysis = getattr(trend_data, "cvd_analysis", {}) or {}
    cvd_trend = str(cvd_analysis.get("trend", "neutral"))
    if regime.direction == BUY_DIRECTION and cvd_trend == "bearish":
        return None  # ціна пробиває вгору, але CVD падає — слабкий пробій
    if regime.direction == SELL_DIRECTION and cvd_trend == "bullish":
        return None
```

Якщо CVD = `'neutral'` — вхід дозволений. Блокуємо тільки явну дивергенцію.

---

### Бектест після Фази 1

```bash
PYTHONPATH=. ./.venv/bin/python backtesting/run_backtest.py \
  --symbol SOLUSDT ETHUSDT AVAXUSDT \
  --from 2025-11-01 --to 2026-05-01 \
  --output-json artifacts/phase1.json
```

**Очікуємо:** менша кількість сигналів, але краще `expectancy_by_setup` для `breakout_reclaim`.

---

## Фаза 2 — Виправлення стоп-логіки

### 2.1 Збільшити `stop_atr_multiplier`: 1.30 → 1.80

**Файл:** `trading/domain/strategy_config.py`

**Проблема:**
1.3 ATR на H1 — дуже тісний стоп для трендових входів. Типова корекція після H1-пробою
= 0.8–1.5 ATR. Стоп на 1.3 ATR систематично спрацьовує на нормальному шумі до того,
як позиція розвивається.

**Рішення:**
```python
stop_atr_multiplier: Decimal = Decimal("1.80")  # було 1.30
```

Ефект: позиція буде меншою за розміром (більша відстань до стопу = менший qty при
тому самому % ризику), але значно рідше стопуватиметься на шумі.

Також збільшити для reclaim:
```python
reclaim_stop_atr_multiplier: Decimal = Decimal("1.40")  # було 1.00
```

---

### 2.2 Вимагати D1-alignment

**Файл:** `trading/domain/strategy_config.py`

**Проблема:**
Зараз `require_d1_alignment = False`. Входи проти денного тренду значно гірші за
якістю. Торгувати H1 breakout проти D1 тренду = входити в контртрендовий рух
проти більших гравців.

**Рішення:**
```python
# RegimeStrategyConfig
require_d1_alignment: bool = True  # було False
```

> **Примітка:** Якщо кількість угод після цієї зміни стає занадто малою (менше
> 1-2 сигналів на символ в місяць) — розглянути як soft filter через додатковий
> параметр `d1_alignment_weight` замість hard block.

---

### 2.3 Підвищити поріг `min_trend_strength`

**Контекст:**
GMMA вважає тренд `"medium"` якщо **або** short EMAs впорядковані **або** long EMAs
впорядковані (тільки одна група). Це досить м'яка умова.

**Рекомендація:** Залишити `"medium"`, але після бектесту порівняти результати при
`"strong"`. Якщо `pnl_by_regime` для `medium` суттєво гірший — підняти поріг.

---

### Бектест після Фази 2

Порівняти `exit_reason_counts.stop_loss` — відсоток стоп-лосів має знизитись відносно
baseline.

---

## Фаза 3 — Перебудова exit-логіки

> **Важливо:** Фази 3 і 4 треба робити разом. Немає сенсу тюнити exit-параметри
> без симуляції stop-management в бектесті (Фаза 4).

### 3.1 Зменшити `tp1_close_fraction`: 50% → 25%

**Файл:** `trading/domain/strategy_config.py`

**Проблема:**
Закривати 50% на TP1 = виходити з половиною позиції дуже рано. Якщо ціна
повертається до breakeven після partial close — net profit ≈ 0. З комісіями — мінус.
Результат: 3R потенційний трейд перетворюється на ~0.5R фактичний.

**Рішення:**
```python
tp1_close_fraction: Decimal = Decimal("25")  # було 50
```

25% закривається при TP1 — достатньо для "зняття прибутку", але залишає 75%
позиції для продовження тренду.

---

### 3.2 Відсунути `breakeven_trigger_r`: 1.5R → 2.0R

**Файл:** `trading/domain/strategy_config.py`

**Проблема:**
Переводити стоп в breakeven при 1.5R = вбивати угоди на нормальних корекціях.
При розширенні на 1.5R корекція до breakeven на H1 — звичайна справа.
Угода закривається в нуль замість продовження до 3R.

**Рішення:**
```python
breakeven_trigger_r: Decimal = Decimal("2.0")  # було 1.5
```

---

### 3.3 Відсунути `trail_activation_r`: 2.0R → 2.5R

**Файл:** `trading/domain/strategy_config.py`

**Рішення:**
```python
trail_activation_r: Decimal = Decimal("2.5")  # було 2.0
```

Дати позиції більше простору до того, як починається trailing.

---

### 3.4 Використовувати реальний ATR для trailing замість апроксимації

**Файли:** `trading/domain/execution.py`, `trading/infrastructure/execution_service.py`

**Проблема:**
Поточне trailing розраховує відстань як апроксимацію:
```python
# execution.py
return initial_risk / stop_atr_multiplier * trail_atr_multiple
# = initial_risk / 1.80 * 1.0 ≈ 0.56 * initial_risk  (приблизний ATR)
```
Реальний ATR в момент trailing може бути зовсім іншим ніж на момент входу.
Якщо ринок заспокоївся — trailing надто широкий (не захищає прибуток).
Якщо прискорився — надто вузький (стопується на шумі).

**Рішення:**

Крок 1 — `build_stop_loss_update()` в `execution.py` отримує опціональний `current_atr`:
```python
def build_stop_loss_update(
    symbol,
    opened_positions,
    current_price_raw,
    exit_config,
    breakout_config,
    current_atr: Optional[Decimal] = None,  # ← новий параметр
) -> Optional[Dict[str, Any]]:
```

Крок 2 — У trailing-логіці замінити апроксимацію на реальний ATR:
```python
# Замість:
trail_distance = _estimated_atr_distance(initial_risk, breakout_config, exit_config)

# Стати:
if current_atr and current_atr > 0:
    trail_distance = current_atr * exit_config.trail_atr_multiple
else:
    trail_distance = _estimated_atr_distance(initial_risk, breakout_config, exit_config)
```

Крок 3 — `manage_open_position()` в `execution_service.py` витягує ATR через
`market_data_provider`:
```python
# У manage_open_position(), після fetch_current_price:
current_atr = None
if self.market_data_provider is not None:
    try:
        trend_data, _ = self.market_data_provider.get_market_context(normalized_symbol, False)
        current_atr = Decimal(str(trend_data.atr)) if trend_data.atr else None
    except Exception:
        pass  # fallback to approximation

update = build_stop_loss_update(
    symbol, opened_positions, current_price,
    self.strategy_config.exit, self.strategy_config.breakout,
    current_atr=current_atr,
)
```

---

## Фаза 4 — Симуляція stop-management в бектесті

**Файл:** `backtesting/execution.py`, `backtesting/models.py`

**Проблема — критична для розуміння стратегії:**
`ExecutionSimulator._try_close_position()` знає тільки про фіксований TP і SL.
Він **не симулює:**
- Partial close на TP1 (після досягнення `breakeven_trigger_r`)
- Переведення стопу в breakeven після partial close
- Trailing stop активацію після `trail_activation_r`

Через це бектест принципово відрізняється від live поведінки:
- Трейди, що в live завершуються на breakeven, в бектесті падають в повний SL або
  ідуть в повний TP
- Неможливо валідувати exit-параметри (breakeven_trigger_r, trail_activation_r)
  без цієї симуляції

**Рішення — розширити `Position` і `ExecutionSimulator`:**

Крок 1 — Додати поля в `Position` (`backtesting/models.py`):
```python
@dataclass
class Position:
    # ... існуючі поля ...
    breakeven_triggered: bool = False
    trailing_active: bool = False
    peak_price: Decimal = Decimal("0")  # highest for long / lowest for short
```

Крок 2 — Додати метод `_apply_stop_management()` в `ExecutionSimulator`:

```python
def _apply_stop_management(
    self,
    candle: Dict[str, Any],
    exit_config,
    breakout_config,
) -> None:
    """Update stop-loss level based on breakeven and trailing rules."""
    if not self.position:
        return

    position = self.position
    candle_high = Decimal(str(candle["high"]))
    candle_low = Decimal(str(candle["low"]))

    initial_risk = position.signal_payload.get("risk_distance")
    if not initial_risk:
        return
    initial_risk = Decimal(str(initial_risk))
    if initial_risk <= 0:
        return

    # Оновити peak price
    if position.direction == "buy":
        position.peak_price = max(position.peak_price, candle_high)
        move = position.peak_price - position.entry_price
    else:
        if position.peak_price == 0:
            position.peak_price = position.entry_price
        position.peak_price = min(position.peak_price, candle_low)
        move = position.entry_price - position.peak_price

    # Breakeven trigger
    if not position.breakeven_triggered:
        if move >= initial_risk * exit_config.breakeven_trigger_r:
            fee_buffer = position.entry_price * Decimal("0.0011")
            if position.direction == "buy":
                new_stop = position.entry_price + fee_buffer
            else:
                new_stop = position.entry_price - fee_buffer
            position.stop_loss = new_stop
            position.breakeven_triggered = True

    # Trailing activation
    if position.breakeven_triggered and move >= initial_risk * exit_config.trail_activation_r:
        trail_distance = (
            initial_risk / Decimal(str(breakout_config.stop_atr_multiplier))
            * Decimal(str(exit_config.trail_atr_multiple))
        )
        if position.direction == "buy":
            trailed = position.peak_price - trail_distance
            if trailed > position.stop_loss:
                position.stop_loss = trailed
        else:
            trailed = position.peak_price + trail_distance
            if trailed < position.stop_loss:
                position.stop_loss = trailed
```

Крок 3 — Викликати `_apply_stop_management()` в `process_candle()` перед перевіркою виходу:
```python
def process_candle(self, candle, bar_index):
    self._try_fill_pending_order(candle, bar_index)
    self._apply_stop_management(candle, exit_config, breakout_config)  # ← додати
    self._try_close_position(candle, bar_index)
```

> **Примітка:** `ExecutionSimulator` потребує отримати `exit_config` і `breakout_config`
> з `BacktestConfig`. Їх треба передати при ініціалізації або через `replay_symbol()`.

---

### Бектест після Фаз 3+4

```bash
PYTHONPATH=. ./.venv/bin/python backtesting/run_backtest.py \
  --symbol SOLUSDT ETHUSDT AVAXUSDT \
  --from 2025-11-01 --to 2026-05-01 \
  --output-json artifacts/phase3_4.json
```

Порівняти `average_profit_pct` і `net_pnl_pct` з baseline і попередніми фазами.

---

## Фаза 5 — Персистентний DailyLossTracker

**Файли:** `utils/db_actions.py`, `trading/infrastructure/execution_service.py`

**Проблема:**
`InMemoryDailyLossTracker` скидається при кожному рестарті процесу. Якщо бот впав
в середині поганого дня і перезапустився — daily_loss_stop не спрацює. На
волатильному ринку це може означати серію збиткових угод без захисту.

**Рішення:**

Крок 1 — Додати таблицю в schema `_live_trading_state`:
```sql
CREATE TABLE IF NOT EXISTS _live_trading_state.daily_loss_tracking (
    trading_date DATE NOT NULL,
    realized_loss_r NUMERIC NOT NULL DEFAULT 0,
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (trading_date)
);
```

Крок 2 — Додати async функції в `utils/db_actions.py`:
```python
async def record_daily_loss_r(loss_r: Decimal, for_date: date | None = None) -> Decimal:
    """Persist realized daily loss and return the new accumulated total."""
    ...

async def get_daily_loss_r(for_date: date | None = None) -> Decimal:
    """Read today's accumulated realized loss from DB."""
    ...
```

Крок 3 — `BybitPositionExecutor.record_daily_realized_loss_r()` в
`execution_service.py` → persists до DB плюс оновлює in-memory tracker.

Крок 4 — При старті `TradingScheduler.run_forever()` перед першим циклом
завантажити поточний `daily_loss_r` з DB і встановити в tracker.

---

## Фаза 6 — Фокус на ліквідних символах

**Файл:** `utils/config.py`

**Проблема:**
24 символи обробляються послідовно кожну годину. Символи з малою ліквідністю
(`PENGUUSDT`, `WIFUSDT`, `VIRTUALUSDT`, `TAOUSDT`) дають маніпулятивні пробої,
ширший spread і гірші fills. Вони займають слоти `max_open_positions = 3`, блокуючи
якісніші можливості на ліквідних символах.

**Рішення:**
Скоротити `TRADING_SYMBOLS` до топ-12:

```python
TRADING_SYMBOLS = [
    # majors
    'ETHUSDT',
    'XRPUSDT',
    'LTCUSDT',
    # l1_l2_beta
    'SOLUSDT',
    'AVAXUSDT',
    'ADAUSDT',
    'ARBUSDT',
    'NEARUSDT',
    # infra
    'LINKUSDT',
    'AAVEUSDT',
    'UNIUSDT',
    # meme_beta (найліквідніший)
    'DOGEUSDT',
]
```

Символи що виключаються: `APTUSDT`, `SUIUSDT`, `DOTUSDT`, `TONUSDT`, `TAOUSDT`,
`WIFUSDT`, `PENGUUSDT`, `VIRTUALUSDT`, `WLDUSDT`, `JUPUSDT`, `ENAUSDT`, `ZECUSDT`.

---

## Зведена таблиця змін параметрів

| Параметр | Поточне | Нове | Файл |
|---|---|---|---|
| `lookback_candles` | 20 | 40 | `trading/domain/strategy_config.py` |
| `stop_atr_multiplier` | 1.30 | 1.80 | `trading/domain/strategy_config.py` |
| `reclaim_stop_atr_multiplier` | 1.00 | 1.40 | `trading/domain/strategy_config.py` |
| `tp1_close_fraction` | 50 | 25 | `trading/domain/strategy_config.py` |
| `breakeven_trigger_r` | 1.5 | 2.0 | `trading/domain/strategy_config.py` |
| `trail_activation_r` | 2.0 | 2.5 | `trading/domain/strategy_config.py` |
| `require_d1_alignment` | False | True | `trading/domain/strategy_config.py` |
| `require_cvd_alignment` (новий) | — | True | `trading/domain/strategy_config.py` |
| `indicator_history_period` | 25 | 50 | `backtesting/models.py` |

---

## Порядок виконання

```
Фаза 0 — Baseline бектест
         (обов'язково перед усім)
          ↓
Фаза 1 — Entry фільтри
         volume filter для reclaim,
         lookback 20→40,
         MFI filter,
         CVD confirmation
          ↓ бектест після Фази 1
Фаза 2 — Stop-loss якість
         stop_atr_multiplier 1.30→1.80,
         reclaim_stop 1.00→1.40,
         require_d1_alignment=True
          ↓ бектест після Фази 2
Фази 3+4 — Exit параметри + симуляція в бектесті
           tp1_close 50→25, breakeven 1.5→2.0,
           trail 2.0→2.5, реальний ATR для trailing,
           симуляція breakeven/trailing в ExecutionSimulator
          ↓ бектест після Фаз 3+4
Фаза 5 — DailyLossTracker в БД
         (risk management, не впливає на P&L)
          ↓
Фаза 6 — Скорочення символів до топ-12
         (після валідації через бектест)
```

> **Фази 3 і 4 треба робити разом.** Тюнити exit-параметри без симуляції
> stop-management в бектесті не має сенсу.

---

## Що НЕ потрібно міняти

- Архітектуру `application / domain / infrastructure` — добре зроблено
- `PortfolioRiskConfig` (max_open_positions, heat, clusters) — логічна і правильна
- Загальну структуру `BreakoutTrendStrategyConfig` — тільки значення параметрів
- `detect_market_regime()` — логіка H1+H4+D1 алайнменту правильна
- Концепцію `reclaim_enabled` — ідея правильна, проблема тільки у volume-фільтрі
