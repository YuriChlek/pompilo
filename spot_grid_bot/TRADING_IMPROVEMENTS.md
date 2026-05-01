# Spot Grid Bot — Trading Roadmap

> Версія: 2026-05-01
> Статус: короткий roadmap, узгоджений з поточною кодовою базою

---

## Мета

Цей документ фіксує лише ті торгові покращення, які мають реалістичний потенціал дати користь у поточній архітектурі бота.

Що вже є в коді і не вважається "відсутньою функцією":

- portfolio-level allocation
- range entry quality filtering
- underwater recovery logic
- adaptive uptrend take-profit adjustment
- multi-timeframe regime confirmation

Тому roadmap нижче не дублює вже реалізовані механізми, а описує лише наступні доцільні кроки.

---

## Пріоритети

| ID | Ініціатива | Пріоритет | Очікувана користь | Складність |
|---|---|---|---|---|
| R1 | Volume confirmation для regime transitions | Високий | Менше false transitions | Низька |
| R2 | Conditional underwater averaging | Високий | Менше averaging у шумі | Низька |
| R3 | Budget split: new entries vs recovery | Високий | Recovery не блокує нові можливості | Середня |
| R4 | Asymmetric range grid | Високий | Краща відповідність grid до position-in-range | Низька |
| R5 | ATR-normalized portfolio sizing | Середній | Рівномірніший risk-per-dollar між символами | Середня |
| R6 | Adaptive sell sizing by unrealized PnL | Середній | Краще керування profit-taking | Низька |
| R7 | RSI filter для RANGE entries | Середній | Додатковий entry-quality filter | Низька |
| R8 | Swing-aware range grid levels | Середній | Рівні ближчі до реальної market structure | Середня |
| R9 | Adaptive grid step by volatility regime | Середній | Grid краще підлаштовується під volatility | Середня |
| R10 | VWAP anchor у UPTREND | Низький–Середній | Кращі pullback entry levels | Середня |
| R11 | Trailing sell anchor for strong UPTREND | Низький–Середній | Пізніша фіксація прибутку в сильному тренді | Середня |

---

## Реальний стан поточної реалізації

Перед початком будь-якої роботи важливо враховувати:

- `range_entry_policy.py` уже знижує budget і buy-level count у слабкому RANGE.
- `uptrend_policy.py` уже має underwater recovery profile.
- `uptrend_policy.py` уже має adaptive take-profit adjustment для UPTREND.
- `portfolio_allocator.py` уже робить portfolio-level scoring і budget distribution.
- `grid_builder.py` уже розділяє логіку RANGE і UPTREND grids.

Тому roadmap не слід трактувати як "боту бракує базових торгових механізмів". Йдеться про подальший тюнінг і refinement.

---

## Фаза 1 — Високий ROI

### R1. Volume confirmation для regime transitions

**Навіщо.**
Поточний regime detection не враховує підтвердження об'ємом при переходах. Це може давати false transitions на слабких breakout candles.

**Що змінювати.**

- Додати `current_volume` і `volume_ma20` у indicator/regime snapshot.
- У state transition logic збільшувати `pending_count` лише якщо breakout підтриманий об'ємом.
- Додати feature flag у `StrategyConfig`.

**Очікуваний ефект.**
Менше помилкових переходів між `RANGE`, `UPTREND`, `DOWNTREND`.

---

### R2. Conditional underwater averaging

**Навіщо.**
Поточне underwater averaging активується без перевірки, наскільки "свіжим" є новий regime. Це може призводити до averaging одразу після шумного transition.

**Що змінювати.**

- Передавати `bars_in_state` у recovery logic.
- Не вмикати averaging, доки regime не прожив мінімальну кількість барів.
- Додати параметр `underwater_min_bars_in_regime` у `GridConfig`.

**Очікуваний ефект.**
Менше помилкового усереднення під час нестабільних transitions.

---

### R3. Budget split: new entries vs recovery

**Навіщо.**
Поточний allocator працює з єдиним пулом allocatable quote. При активному recovery нові символи можуть не отримати budget.

**Що змінювати.**

- Розділити allocatable quote на `new_entry_quota` і `recovery_quota`.
- Recovery symbols фінансувати з recovery pool.
- Невикористаний recovery pool повертати в new-entry pool.

**Очікуваний ефект.**
Портфель не "залипає" лише в recovery сценаріях.

---

### R4. Asymmetric range grid

**Навіщо.**
Поточна RANGE grid симетрична. Якщо ціна вже близько до верхньої або нижньої межі діапазону, симетричний розподіл рівнів не оптимальний.

**Що змінювати.**

- Визначати позицію ціни всередині range.
- Якщо ціна у верхній частині range: більше SELL levels, менше BUY levels.
- Якщо ціна у нижній частині range: більше BUY levels, менше SELL levels.

**Очікуваний ефект.**
Краща відповідність grid поточному положенню ціни в діапазоні.

---

## Фаза 2 — Середній ROI

### R5. ATR-normalized portfolio sizing

**Навіщо.**
Поточний allocator не нормалізує allocation між символами за relative volatility. Це може завищувати exposure до більш "нервових" інструментів.

**Що змінювати.**

- Передавати `atr_pct = atr14 / mark_price` у portfolio allocation input.
- Додати volatility-based multiplier у symbol scoring.
- Обмежити multiplier clamp-ом, щоб не ламати allocation extremes.

**Очікуваний ефект.**
Більш однорідний risk-per-dollar між символами.

---

### R6. Adaptive sell sizing by unrealized PnL

**Навіщо.**
Поточні sell weights статичні. Немає відмінності між inventory у сильному плюсі та inventory біля breakeven.

**Що змінювати.**

- У sell sizing враховувати unrealized PnL відносно `cost_basis_price`.
- При сильному плюсі збільшувати вагу верхніх sell levels.
- При underwater inventory не посилювати sell aggressiveness.

**Очікуваний ефект.**
Гнучкіший profit-taking без зміни базової grid-архітектури.

---

### R7. RSI filter для RANGE entries

**Навіщо.**
У боті вже є range quality scoring, але немає окремого oversold/overbought filter для RANGE entry quality.

**Що змінювати.**

- Додати `rsi14` до `IndicatorSnapshot`.
- У RANGE entry policy штрафувати entries при overbought і трохи пом'якшувати при oversold.
- Тримати зміну за feature flag.

**Очікуваний ефект.**
Додатковий фільтр якості BUY у mean-reversion сценаріях.

---

### R8. Swing-aware range grid levels

**Навіщо.**
Поточний regime detection уже використовує swing structure, але самі swing levels не доходять до grid construction.

**Що змінювати.**

- Передати `swing_highs` і `swing_lows` із structure analysis у planning layer.
- У RANGE grid дозволити зміщення частини рівнів до близьких support/resistance points.
- Лишити fallback на поточну ATR-based ladder.

**Очікуваний ефект.**
BUY/SELL levels ближчі до реально значущих ринкових зон.

---

### R9. Adaptive grid step by volatility regime

**Навіщо.**
Поточний step базується на ATR. Це вже непогано, але короткострокова volatility інколи змінюється швидше, ніж ATR встигає адаптуватися.

**Що змінювати.**

- Додати short-vs-long realized volatility ratio.
- Масштабувати `atr_grid_step_multiplier` цим ratio у безпечних межах.
- Використовувати однаковий підхід для RANGE та UPTREND grids.

**Очікуваний ефект.**
Менше надто щільних grids у stress-vol і менше надто широких grids у спокійному режимі.

---

## Фаза 3 — Опційні покращення після валідації

### R10. VWAP anchor у UPTREND

**Навіщо.**
VWAP може дати більш змістовний pullback anchor для uptrend entries, але це не базовий gap у поточній логіці.

**Що змінювати.**

- Додати `vwap_daily` до indicator snapshot.
- Якщо VWAP лежить у buy pullback zone, дозволяти підмінити один із ATR-based buy levels.

**Коли робити.**
Лише після того, як базові allocator/grid improvements уже перевірені.

---

### R11. Trailing sell anchor for strong UPTREND

**Навіщо.**
У боті вже є adaptive uptrend take-profit adjustment, але немає persisted trailing anchor між циклами.

**Що змінювати.**

- Додати persisted `last_sell_upper_price` у runtime state.
- У strong uptrend зсувати sell anchor вгору замість жорсткої прив'язки лише до поточної ціни.

**Коли робити.**
Лише якщо backtest покаже, що поточний adaptive take-profit усе ще занадто рано фіксує прибуток.

---

## Що свідомо виключено з roadmap

Ці ідеї не пріоритезуються для поточної версії:

- Volume Profile / POC / VAH / VAL
- Correlation-aware allocation
- Low-liquidity hours budget throttling
- Calendar blocking перед макроподіями
- Momentum ranking between symbols

Причина однакова: або високий implementation overhead при слабко доведеному edge, або сумнівний ROI для поточної архітектури бота.

---

## Порядок впровадження

### Етап 1

- R1 Volume confirmation
- R2 Conditional underwater averaging
- R3 Budget split: new entries vs recovery
- R4 Asymmetric range grid

### Етап 2

- R5 ATR-normalized sizing
- R6 Adaptive sell sizing
- R7 RSI filter
- R8 Swing-aware grid levels
- R9 Adaptive grid step

### Етап 3

- R10 VWAP anchor
- R11 Trailing sell anchor

---

## Правило для кожної зміни

1. Реалізувати через config flag або безпечний default.
2. Прогнати backtest і порівняти:
   - `pnl`
   - `max_drawdown`
   - `rebuild_count`
   - `trade_count`
   - `blocked_no_loss_sell_count`
3. Прогнати в `paper` mode перед live rollout.
4. Не вмикати в live за замовчуванням без локальної валідації.
