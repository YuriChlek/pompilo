# Plan покращення ML pipeline для прибуткових сигналів

> Дата складання: 2026-05-02  
> Поточний стан: Milestone 5 in progress  
> Мета: перейти від "статистично коректної" моделі до моделі що генерує прибуткові торгові сигнали

---

## Зміст

- [Діагноз: Чому поточна модель може не торгувати в плюс](#діагноз)
- [Milestone A — Швидкі виправлення (1-2 дні)](#milestone-a)
- [Milestone B — Покращення моделі (3-5 днів)](#milestone-b)
- [Milestone C — Feature Engineering Upgrade (3-5 днів)](#milestone-c)
- [Milestone D — Архітектурний апгрейд (2-3 тижні)](#milestone-d)
- [Послідовність виконання](#послідовність-виконання)
- [Метрики успіху](#метрики-успіху)

---

## Діагноз

### Проблема 1 — Невідповідність метрики та мети (критична)

**Файл:** `module_ai/evaluate_run.py:324-331`

Approval gate перевіряє тільки:
```python
approved = (
    validation_forecast["rmse_q50"] <= args.approval_max_validation_rmse        # ← статистика
    and final_forecast["rmse_q50"] <= args.approval_max_final_test_rmse          # ← статистика
    and validation_forecast["interval_coverage_rate"] >= ...                     # ← статистика
    and final_forecast["interval_coverage_rate"] >= ...                          # ← статистика
    and validation_per_symbol_ok                                                 # ← статистика
    and final_test_per_symbol_ok                                                 # ← статистика
)
```

Жодної метрики прибутковості. Модель з `rmse_q50=0.005` і `directional_accuracy=0.48` отримує `deployable=True`.

**Розрив між ціллю навчання та ціллю торгівлі:**
```
Навчання оптимізує:    min( |q50 - actual_return| )
Торгівля потребує:     max( direction_correct * |return| - transaction_cost - slippage )
```

Квантильна регресія навчається точно вгадувати *величину* повернення. Але для прибутку важливо вгадати *напрям* з достатнім запасом, щоб перекрити комісію. Це різні задачі.

### Проблема 2 — Фіктивний backtest у calibration (критична)

**Файл:** `module_ai/signal_calibration.py:46-48`

```python
parser.add_argument("--transaction-cost", type=float, default=0.0)
parser.add_argument("--slippage", type=float, default=0.0)
parser.add_argument("--execution-rule", default="close_to_close")
```

Default значення — `transaction_cost=0.0`, `slippage=0.0`, `execution_rule=close_to_close`.

**Наслідок:** `cumulative_return` в calibration report — без комісій. При реальній комісії Bybit/Binance taker 0.055% і `horizon_h=6` (6 годин), кожна угода коштує 0.11% (in+out). Якщо середній `average_trade_return` в backtest = 0.08%, то на реальному ринку кожна угода — збиток.

**Soft leakage в грідах:** функція `_derive_default_grids` (`signal_calibration.py:157-188`) будує grid thresholds на основі *розподілу validation forecasts*. Тобто пошуковий простір вже адаптований до validation даних ще до того як кандидати оцінюються.

### Проблема 3 — Flattened window: втрата часової структури

**Файл:** `module_ai/windowing.py:128`

```python
rows.append(window[features].to_numpy(dtype=np.float64).reshape(-1))
# encoder_len=48, features=25 → вектор [1200]
# [symbol_id__t_minus_47, hour_sin__t_minus_47, ..., volatility_24__t_minus_0]
```

GBM бачить 1200 незалежних чисел. Модель не знає що `close__t_minus_1` і `close__t_minus_0` — сусідні свічки у часі. Тимчасова послідовність закодована лише в іменах ознак.

**Практичний наслідок:** feature importance покаже що модель використовує ~15-30 з 1200 ознак, переважно найближчі лаги `t_minus_0` та `t_minus_1`. 46 свічок lookback — практично не використовується. `encoder_len=48` надмірний і уповільнює тренування.

### Проблема 4 — Жорсткий AND-фільтр heuristic gate

**Файл:** `module_signal/bot.py:233-243`

```python
buy_confirmed = (
    heuristic_direction == TradeSignal.BUY      # умова 1
    and int(round(of_data.indicators["rsi"])) < max_rsi_val  # умова 2
    and of_data.cvd["trend"] == "bullish"       # умова 3
    and of_data.cvd["strength"] in {"strong", "very_strong"}  # умова 4
)
```

Якщо кожна умова виконується в 70% випадків, AND із 4: `0.7^4 ≈ 0.24`. Фільтр відкидає ~76% ML BUY-сигналів. Sell-сторона ще жорсткіша: потребує `rsi > 70` і `confidence > 60.0`.

### Проблема 5 — Відсутність walk-forward у тренуванні

**Файл:** `module_ai/train.py:127-133`

Один фіксований split `70/15/15`. Гіперпараметри (`n_estimators=300`, `max_depth=3`, `learning_rate=0.05`) встановлені без будь-якої оптимізації. Walk-forward код існує в `splits.py:generate_walk_forward_folds` але не використовується під час тренування.

### Проблема 6 — Відсутність крос-символьних ознак

**Файл:** `module_ai/data_pipeline.py:29-76`

Всі 25 ознак — внутрішньосимвольні. Немає:
- `btc_return_1`, `btc_return_6` — BTC як ринковий драйвер для альтів
- `eth_return_1` — ETH-beta сигнал
- Кореляцій між символами в realtime

### Проблема 7 — Немає ранньої зупинки (early stopping)

**Файл:** `module_ai/modeling.py:163-204`

sklearn `GradientBoostingRegressor` тренується фіксовану кількість дерев (`n_estimators=300`) без early stopping. CatBoost підтримує early stopping але воно не налаштоване. Результат: гарантований overfitting до training split.

---

## Milestone A — Швидкі виправлення (1-2 дні)

> Не змінює архітектуру. Виправляє критичні методологічні помилки в evaluation та calibration.

### A1 — Додати торгові метрики в approval gate

**Файл для зміни:** `module_ai/evaluate_run.py`

**Що змінити:**

Додати нові CLI аргументи в `build_argparser()`:
```
--approval-min-directional-accuracy    default=0.52   (мін. точність напряму по не-HOLD сигналах)
--approval-min-hit-rate                default=0.50   (мін. частка прибуткових угод)
--approval-min-profit-factor           default=1.0    (мін. profit factor: сума прибутків / сума збитків)
--approval-max-drawdown                default=0.20   (макс. drawdown)
```

Змінити умову `approved` (рядок 324):
```python
approved = (
    # поточні статистичні критерії (залишаються)
    validation_forecast["rmse_q50"] <= args.approval_max_validation_rmse
    and final_forecast["rmse_q50"] <= args.approval_max_final_test_rmse
    and validation_forecast["interval_coverage_rate"] >= args.approval_min_validation_coverage
    and final_forecast["interval_coverage_rate"] >= args.approval_min_final_test_coverage
    and validation_per_symbol_ok
    and final_test_per_symbol_ok
    # НОВІ торгові критерії
    and validation_report["signal_metrics"]["directional_accuracy"] >= args.approval_min_directional_accuracy
    and validation_report["trading_metrics"]["hit_rate"] >= args.approval_min_hit_rate
    and validation_report["trading_metrics"]["profit_factor"] >= args.approval_min_profit_factor
    and validation_report["trading_metrics"]["max_drawdown"] <= args.approval_max_drawdown
)
```

Додати ці нові критерії у `approval_criteria` dict у `evaluation_summary` (рядок 358).

**Очікуваний ефект:** відфільтрує артефакти які добре передбачають returns але погано торгують.

---

### A2 — Реалістичний backtest у calibration

**Файл для зміни:** `module_ai/signal_calibration.py`

**Що змінити:**

Змінити дефолти в `build_argparser()`:
```python
# БУЛО:
parser.add_argument("--execution-rule", default="close_to_close")
parser.add_argument("--transaction-cost", type=float, default=0.0)
parser.add_argument("--slippage", type=float, default=0.0)

# СТАЛО:
parser.add_argument("--execution-rule", default="next_open_to_h_close")
parser.add_argument("--transaction-cost", type=float, default=0.0005)   # 0.05% taker Bybit
parser.add_argument("--slippage", type=float, default=0.0002)           # 0.02% slippage estimate
```

Змінити функцію `_score_candidate()` (рядок 285):
```python
# Додати penalty за від'ємний net cumulative return
if metrics["cumulative_return"] < 0:
    viable = False
    reasons.append("negative_net_cumulative_return")

# Додати вагу profit_factor у score
score = (
    metrics["cumulative_return"] * 100.0
    + metrics["sharpe_like"] * 10.0
    + metrics["hit_rate"] * 2.0
    + metrics["directional_accuracy"] * 1.0
    + np.log1p(max(metrics["profit_factor"], 0.0)) * 5.0   # НОВЕ
    - metrics["max_drawdown"] * 50.0
    - density_penalty * 5.0
)
```

Ізолювати grid від validation даних: в `_derive_default_grids()` замінити квантилі validation forecasts на **фіксовану сітку** або рахувати грід тільки на **train split** (щоб уникнути soft leakage):
```python
# ЗАМІСТЬ: q50_quantiles = np.quantile(abs_q50, [0.0, 0.25, 0.5, 0.75, 0.9])
# СТАЛО: фіксована сітка на основі типових crypto return distributions
cost_grid = [0.0, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02]
threshold_grid = [0.0, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01]
width_grid = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]
```

**Очікуваний ефект:** calibration report перестане показувати фіктивні прибутки. Відібрані thresholds будуть реалістично прибутковими після комісій.

---

### A3 — Пом'якшити heuristic confirmation gate

**Файл для зміни:** `module_signal/bot.py`

**Що змінити:**

Замінити жорсткий AND на зважений score для `_combine_with_heuristics()`:

```python
# БУЛО (AND-4 умови):
buy_confirmed = (
    heuristic_direction == TradeSignal.BUY
    and int(round(of_data.indicators["rsi"])) < max_rsi_val
    and of_data.cvd["trend"] == "bullish"
    and of_data.cvd["strength"] in {"strong", "very_strong"}
)

# СТАЛО (зважений score, поріг >= 4 з 8):
heuristic_buy_score = (
    (heuristic_direction == TradeSignal.BUY) * 3           # найсильніший сигнал
    + (of_data.cvd["trend"] == "bullish") * 2
    + (of_data.cvd["strength"] == "very_strong") * 2
    + (of_data.cvd["strength"] == "strong") * 1
    + (int(round(of_data.indicators["rsi"])) < 50) * 1
    + (of_data.market_trend != "bearish") * 1
)
buy_confirmed = heuristic_buy_score >= 4 and entry_price is not None
```

Аналогічно для sell:
```python
# БУЛО:
sell_confirmed = (
    heuristic_direction == TradeSignal.SELL
    and heuristic_signal.get("confidence", 0.0) > 60.0
    and of_data.indicators["rsi"] > 70
    and of_data.cvd["trend"] == "bearish"
    and of_data.cvd["strength"] in {"strong", "very_strong"}
    and of_data.market_trend != "neutral"
)

# СТАЛО:
heuristic_sell_score = (
    (heuristic_direction == TradeSignal.SELL) * 3
    + (of_data.cvd["trend"] == "bearish") * 2
    + (of_data.cvd["strength"] == "very_strong") * 2
    + (of_data.cvd["strength"] == "strong") * 1
    + (of_data.indicators["rsi"] > 60) * 1
    + (of_data.market_trend == "bearish") * 2
    + (heuristic_signal.get("confidence", 0.0) > 55.0) * 1
)
sell_confirmed = heuristic_sell_score >= 4 and entry_price is not None
```

**Очікуваний ефект:** recall сигналів виросте при збереженні достатнього рівня precision. Менше хороших сигналів буде відкинуто.

---

### A4 — Додати early stopping для CatBoost

**Файл для зміни:** `module_ai/modeling.py`

**Що змінити:**

В `_train_catboost_quantile_models()` (рядок 207):
```python
# Додати early_stopping_rounds
model_kwargs = {
    ...
    "early_stopping_rounds": 50,    # НОВЕ: зупинитись якщо 50 ітерацій без покращення
    "eval_fraction": 0.1,           # НОВЕ: 10% train даних → internal validation
}
```

В `_train_gradient_boosting_quantile_models()` (рядок 163):
```python
# sklearn GBM не підтримує early stopping нативно.
# Замінити на n_iter_no_change:
model = GradientBoostingRegressor(
    loss="quantile",
    alpha=quantile,
    n_iter_no_change=30,       # НОВЕ: early stopping
    validation_fraction=0.1,   # НОВЕ: internal val set
    tol=1e-5,
    ...
)
```

**Очікуваний ефект:** зменшення overfitting, краща генералізація на тест.

---

## Milestone B — Покращення моделі (3-5 днів)

> Зберігає поточний пайплайн, покращує якість моделі.

### B1 — Додати LightGBM як новий backend

**Нові файли:** додати `lightgbm` в `requirements.txt`  
**Файли для зміни:** `module_ai/modeling.py`, `module_ai/train.py`

LightGBM стабільно перевершує sklearn GBM на tabular даних:
- Швидший у 10-20x (histogram-based splits)
- `boosting_type='dart'` зменшує overfitting
- `num_leaves` замість `max_depth` — більша гнучкість

**Що додати в `modeling.py`:**

```python
MODEL_BACKEND_LIGHTGBM = "lightgbm"
SUPPORTED_MODEL_BACKENDS = (
    MODEL_BACKEND_GRADIENT_BOOSTING,
    MODEL_BACKEND_CATBOOST,
    MODEL_BACKEND_LIGHTGBM,       # НОВЕ
)

def _train_lightgbm_quantile_models(
    X: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int,
    n_estimators: int,
    num_leaves: int = 31,
    learning_rate: float,
    subsample: float,
    min_child_samples: int = 20,
    **kwargs,
) -> dict[str, Any]:
    import lightgbm as lgb
    models = {}
    for quantile in QUANTILES:
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            subsample=subsample,
            min_child_samples=min_child_samples,
            boosting_type="dart",           # менший overfitting ніж gbdt
            random_state=random_state,
            n_jobs=-1,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        # Split 10% для early stopping
        val_size = max(int(len(X) * 0.1), 1)
        X_tr, X_val = X[:-val_size], X[-val_size:]
        y_tr, y_val = y[:-val_size], y[-val_size:]
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        models[f"q{int(quantile * 100):02d}"] = model
    return {
        "quantiles": list(QUANTILES),
        "models": models,
        "model_type": "lightgbm_quantile",
        "model_backend": MODEL_BACKEND_LIGHTGBM,
        ...
    }
```

**Що додати в `train.py`:**

Нові CLI параметри:
```
--num-leaves        default=31   (LightGBM specific, замінює max_depth)
--min-child-samples default=20   (LightGBM min samples per leaf)
```

**Рекомендовані стартові параметри для LightGBM:**
```
--model-backend lightgbm
--n-estimators 1000          (early stopping спиниться раніше)
--num-leaves 31
--learning-rate 0.03
--subsample 0.8
--min-child-samples 30
```

---

### B2 — Додати класифікаційну ціль (двоетапний сигнал)

**Файли для зміни:** `module_ai/data_pipeline.py`, `module_ai/runtime.py`, `module_ai/modeling.py`, `module_ai/forecast.py`

**Ідея:** поряд з квантильними регресорами навчати бінарний класифікатор `P(return > threshold)`. Видавати сигнал тільки якщо обидві моделі погоджуються.

**Крок 1 — Нова ціль у `data_pipeline.py`:**

```python
# Додати нову функцію після add_target_return_h()
def add_direction_target(
    df: pd.DataFrame,
    horizon_h: int,
    *,
    buy_threshold: float = 0.003,    # мін. return щоб вважатись BUY
    sell_threshold: float = 0.003,   # мін. abs return щоб вважатись SELL
) -> pd.DataFrame:
    """
    Додає target_direction: 1=BUY, -1=SELL, 0=HOLD
    Використовується для тренування класифікатора паралельно з квантильним регресором.
    """
    out = add_target_return_h(df, horizon_h)
    conditions = [
        out[TARGET_NAME] > buy_threshold,
        out[TARGET_NAME] < -sell_threshold,
    ]
    out["target_direction"] = np.select(conditions, [1, -1], default=0)
    return out
```

**Крок 2 — Додати класифікатор у `modeling.py`:**

```python
def train_direction_classifier(
    X: np.ndarray,
    y_direction: np.ndarray,
    *,
    random_state: int = 1337,
    n_estimators: int = 200,
) -> dict[str, Any]:
    """
    Навчає класифікатор напряму: 1=BUY, -1=SELL, 0=HOLD.
    Використовує LightGBM multiclass або CalibratedClassifierCV(GBM).
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV

    base = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=3,
        random_state=random_state,
        subsample=0.8,
    )
    model = CalibratedClassifierCV(base, cv=3, method="isotonic")
    model.fit(X, y_direction)
    return {
        "direction_classifier": model,
        "classes": list(model.classes_),
        "classifier_type": "calibrated_gbm",
    }
```

**Крок 3 — Змінити `runtime.py:fit_runtime_bundle()`:**

```python
# Після train_quantile_models — тренувати класифікатор
direction_bundle = train_direction_classifier(
    train_samples.X,
    train_direction_samples.y,   # потрібно build_window_samples з target_direction
    random_state=model_params.get("random_state", 1337),
)
```

**Крок 4 — Змінити `forecast.py:get_return_forecast()`:**

```python
# Повертати P(buy) і P(sell) поряд з quantile forecasts
direction_probs = predict_direction_probs(direction_bundle, X_latest)
return ForecastResult(
    ...
    p_buy=float(direction_probs[0, buy_class_idx]),
    p_sell=float(direction_probs[0, sell_class_idx]),
)
```

**Крок 5 — Змінити `bot.py:_derive_ml_signal()`:**

```python
# Додати перевірку P(direction) як додатковий фільтр
if forecast.p_buy is not None and forecast.p_buy < 0.55:
    return MLSignalAssessment(signal=TradeSignal.HOLD, reason="classifier_confidence_low", ...)
```

**Очікуваний ефект:** двоетапний фільтр (квантильний бар'єр + класифікатор) підвищить precision сигналів. Частина сигналів із правильним q10>threshold але неправильним напрямом буде відфільтрована.

---

### B3 — Walk-forward validation при тренуванні

**Файл для зміни:** `module_ai/train.py`

**Що змінити:**

Додати CLI аргумент `--walk-forward-cv` (bool flag). Коли enabled — замість одного split, тренувати на кількох expanding фолдах і звітувати середнє:

```python
parser.add_argument("--walk-forward-cv", action="store_true",
    help="Use walk-forward CV for validation metrics (does not change final model training)")
```

В `main()` після `split_dataset()`:
```python
if args.walk_forward_cv:
    # Генерувати walk-forward фолди з train+val частини
    # Для кожного фолду: train → predict → evaluate
    # Зберігати середнє по фолдам як validation metrics
    # УВАГА: generate_walk_forward_folds працює тільки з single-symbol
    # Для multi-symbol потрібна окрема реалізація (per-symbol walk-forward + averaging)
    walk_forward_metrics = _run_walk_forward_metrics(feature_df, split_config, runtime_bundle)
    dataset_meta["walk_forward_cv_metrics"] = walk_forward_metrics
```

Це не змінює саме тренування (фінальна модель тренується на повному train split), але дає більш надійну оцінку справжньої generalization здатності.

**Увага:** `generate_walk_forward_folds` у `splits.py:414` викликає `_ensure_single_symbol` — не підтримує multi-symbol. Для multi-symbol walk-forward потрібно: запускати per-symbol, потім агрегувати weighted average metrics.

---

### B4 — Оптимізація гіперпараметрів через Optuna

**Новий файл:** `module_ai/hyperopt.py`  
**Нові залежності:** `optuna` в `requirements.txt`

```python
# Короткий опис нового модуля
"""
Запуск: cli.py optimize-hyperparams --symbol ALL --n-trials 50 --model-backend lightgbm
Оптимізує: learning_rate, num_leaves, subsample, min_child_samples
Ціль: max(sharpe_like на validation walk-forward фолдах)
Зберігає: artifacts/__multi__/hyperopt_report.json з best_params
"""
```

Ключовий момент: **ціль оптимізації** — не RMSE, а `sharpe_like` або `profit_factor` з реалістичними costs.

```python
def _objective(trial: optuna.Trial, ...) -> float:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
    }
    # Train на walk-forward фолдах
    # Повернути mean(sharpe_like) з realtime costs
    return mean_sharpe_across_folds
```

---

## Milestone C — Feature Engineering Upgrade (3-5 днів)

> Збагачує feature space без зміни архітектури моделі.

### C1 — Скоротити encoder_len, додати агреговані features

**Файл для зміни:** `module_ai/train.py`, `module_ai/data_pipeline.py`

**Проблема:** `encoder_len=48` × 25 features = 1200 ознак у flat vector. GBM використовує ~15-30. Це надмірно і сповільнює тренування.

**Що змінити:**

Змінити дефолт в `build_argparser()`:
```python
parser.add_argument("--encoder-len", type=int, default=24)  # БУЛО: 48
```

Додати **агреговані trend features** в `data_pipeline.py:_add_grouped_technical_features()`:

```python
# Trend slope features (замінюють raw лаги)
def _linreg_slope(series: pd.Series, window: int) -> pd.Series:
    """Rolling linear regression slope — ефективніше кодує тренд ніж raw лаги."""
    x = np.arange(window, dtype=np.float64)
    x -= x.mean()
    return series.rolling(window).apply(
        lambda y: np.dot(x, y - y.mean()) / (np.dot(x, x) + 1e-10),
        raw=True
    )

out["trend_slope_6"]  = _linreg_slope(close, 6)
out["trend_slope_12"] = _linreg_slope(close, 12)
out["trend_slope_24"] = _linreg_slope(close, 24)

# Volatility regime feature
out["vol_regime_ratio"] = (
    out["return_1"].rolling(24).std() /
    (out["return_1"].rolling(168).std() + 1e-10)   # 7 days
)
# > 1.0 = підвищена волатильність, < 1.0 = спокійний ринок

# Volume acceleration
out["volume_accel"] = out["volume_vs_ma_20"].diff(3)

# CVD momentum change
out["cvd_momentum"] = out["cvd_zscore_20"].diff(6)
```

Оновити `ENCODER_FEATURES` список (+6 нових ознак):
```python
TREND_FEATURES = [
    "trend_slope_6",
    "trend_slope_12",
    "trend_slope_24",
    "vol_regime_ratio",
    "volume_accel",
    "cvd_momentum",
]
```

Після `DATASET_CONTRACT_VERSION = "3.0.0"` змінити на `"4.0.0"` — це примусить retrain.

---

### C2 — Крос-символьні ознаки (BTC як market driver)

**Файл для зміни:** `module_ai/data_access.py`, `module_ai/data_pipeline.py`

Альтернативна криптовалюта із 80%+ кореляцією до BTC. BTC return — найсильніший leading indicator для більшості альтів.

**Що додати:**

В `data_pipeline.py` — нова функція:
```python
def add_market_driver_features(
    df: pd.DataFrame,
    *,
    driver_symbol: str = "BTCUSDT",
    driver_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Додає return_1 і return_6 для market driver (BTC) як cross-asset features.
    Якщо BTCUSDT є серед training_symbols — використовує його колонки.
    Якщо ні — приєднує окремо завантажений driver_df.
    """
    out = df.copy()
    
    if driver_symbol.upper() in df["symbol"].str.upper().unique():
        # BTC вже є в датасеті — беремо його колонки і приєднуємо
        btc_data = df[df["symbol"].str.upper() == driver_symbol.upper()][
            ["open_time", "return_1", "return_6", "volatility_24"]
        ].rename(columns={
            "return_1": "btc_return_1",
            "return_6": "btc_return_6",
            "volatility_24": "btc_volatility_24",
        })
        out = out.merge(btc_data, on="open_time", how="left")
    elif driver_df is not None:
        # BTC не в universe — merge окремого датасету
        ...
    
    return out
```

Нові features в `ENCODER_FEATURES`:
```python
CROSS_ASSET_FEATURES = [
    "btc_return_1",
    "btc_return_6",
    "btc_volatility_24",
]
```

**Нюанс:** BTCUSDT є в `DEFAULT_TRADING_SYMBOLS`. Але якщо робити прогноз для BTCUSDT — `btc_return_1` буде самим собою (leakage!). Потрібен захист:
```python
# Для symbol == BTCUSDT — btc_cross_features заповнювати NaN або середнім по universe
if row["symbol"] == "BTCUSDT":
    out.loc[btc_mask, ["btc_return_1", "btc_return_6"]] = np.nan
```

---

### C3 — Ринковий режим як ознака

**Файл для зміни:** `module_ai/data_pipeline.py`

Модель тренована на Bull-ринку буде погано торгувати на Bear-ринку. Додати явну ознаку режиму:

```python
def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volatility regime: порівняння поточної vol з довгостроковою.
    Trend regime: відношення ціни до 200-денної MA.
    """
    out = df.copy()
    
    for _, group in out.groupby("symbol", sort=False):
        close = group["close"].astype(float)
        
        # Volatility regime (0=low, 1=normal, 2=high)
        ret = close.pct_change(1)
        short_vol = ret.rolling(24).std()    # 1 day
        long_vol  = ret.rolling(720).std()   # 30 days
        vol_ratio = short_vol / (long_vol + 1e-10)
        
        # Trend regime: price vs 200-period MA
        ma200 = close.rolling(200, min_periods=100).mean()
        trend_position = close / (ma200 + 1e-10) - 1.0  # > 0 = above MA200
        
        out.loc[group.index, "vol_regime_score"] = vol_ratio
        out.loc[group.index, "trend_regime_score"] = trend_position
    
    return out
```

Нові ознаки (`ENCODER_FEATURES`):
```python
REGIME_FEATURES = [
    "vol_regime_score",
    "trend_regime_score",
]
```

---

### C4 — Покращена ціль: risk-adjusted return

**Файл для зміни:** `module_ai/data_pipeline.py`

Замість raw `target_return_h` → `target_return_h / rolling_volatility` (Sharpe-like target).

```python
def add_risk_adjusted_target(df: pd.DataFrame, horizon_h: int) -> pd.DataFrame:
    """
    target_sharpe_h = target_return_h / rolling_vol_24
    
    Переваги над raw return:
    - Однаковий масштаб між символами з різною волатильністю
    - Модель навчається вгадувати якість сигналу, а не просто розмір руху
    - Сигнали на low-vol символах будуть порівнюваними з high-vol
    """
    out = add_target_return_h(df, horizon_h)
    vol = out.groupby("symbol")["return_1"].transform(
        lambda x: x.rolling(24, min_periods=12).std()
    )
    out["target_sharpe_h"] = out[TARGET_NAME] / (vol + 1e-6)
    return out
```

**Увага:** при цьому треба змінити і `signal_calibration.py` — пороги будуть у "sigma units" замість return units.

---

## Milestone D — Архітектурний апгрейд (2-3 тижні)

> Найбільший impact, але найбільша складність. Потрібен окремий R&D цикл.

### D1 — Temporal Fusion Transformer (TFT)

**Чому TFT краще для цієї задачі:**

| | Поточний GBM flat | TFT |
|---|---|---|
| Часова структура | ❌ ігнорується | ✅ attention по часу |
| Multi-horizon | ❌ один горизонт | ✅ одночасно кілька |
| Regime detection | ❌ features вручну | ✅ learns automatically |
| Interpretability | partial (FI) | ✅ attention weights |
| Швидкість inference | ✅ миттєво | ⚠️ ~50ms per batch |

**Реалізація через `neuralforecast`:**

```bash
pip install neuralforecast
```

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT

model = TFT(
    h=6,                          # horizon = 6 candles
    input_size=48,                # encoder_len
    hidden_size=128,
    n_head=4,
    attn_dropout=0.1,
    dropout=0.1,
    loss="MQLoss",                # multi-quantile loss [0.1, 0.5, 0.9]
    quantiles=[0.1, 0.5, 0.9],
    max_steps=1000,
    early_stop_patience_steps=50,
    scaler_type="standard",
)

nf = NeuralForecast(models=[model], freq="1h")
nf.fit(train_df)   # train_df у форматі neuralforecast: [unique_id, ds, y, exog_cols...]
forecasts = nf.predict()
```

**Формат даних** (потрібна конвертація з поточного пайплайну):
```
unique_id | ds                  | y (target_return_h) | exog features...
SOLUSDT   | 2024-01-01 00:00:00 | 0.0023              | rsi=55.3, ...
SOLUSDT   | 2024-01-01 01:00:00 | -0.0011             | rsi=52.1, ...
```

**Інтеграція в поточний пайплайн:**
- Новий backend `"tft"` в `modeling.py`
- `SUPPORTED_MODEL_BACKENDS` додати `"tft"`
- `fit_runtime_bundle()` в `runtime.py` — окрема гілка для TFT
- Артефакт зберігати через `nf.save(path)` / `NeuralForecast.load(path)`

**Вимоги:** GPU рекомендований для навчання, CPU для inference.

---

### D2 — Chronos як додатковий feature generator

**Бібліотека:** `amazon/chronos-forecasting` (open source, Apache 2.0)

```bash
pip install chronos-forecasting
```

**Ідея:** не замінювати GBM, а збагачувати features Chronos-прогнозами.

```python
from chronos import ChronosPipeline
import torch

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",   # small: 46M params, fast inference
    device_map="cpu",
    torch_dtype=torch.float32,
)

def get_chronos_features(close_series: np.ndarray, horizon: int = 6) -> dict:
    """
    Запускає Chronos zero-shot прогноз на raw close prices.
    Повертає q10/q50/q90 як нові features.
    """
    context = torch.tensor(close_series[-64:]).unsqueeze(0)  # last 64 candles
    forecast = pipeline.predict(context, prediction_length=horizon, num_samples=20)
    
    return {
        "chronos_q10": float(np.quantile(forecast[0].numpy()[:, -1], 0.1)),
        "chronos_q50": float(np.quantile(forecast[0].numpy()[:, -1], 0.5)),
        "chronos_q90": float(np.quantile(forecast[0].numpy()[:, -1], 0.9)),
        "chronos_interval_width": float(
            np.quantile(forecast[0].numpy()[:, -1], 0.9) -
            np.quantile(forecast[0].numpy()[:, -1], 0.1)
        ),
    }
```

**Нові features (4 штуки):**
- `chronos_q10_return` — нижня межа за Chronos (у відсотках від current close)
- `chronos_q50_return` — медіана за Chronos
- `chronos_q90_return` — верхня межа за Chronos
- `chronos_interval_width` — невпевненість Chronos

**Обмеження:**
- Inference ~200ms на свічку на CPU (прийнятно для 1h candles)
- Chronos натрений на raw values, тому йому треба `close` series, а не normalized features
- Zero-shot якість — на рівні strong baseline, але не SOTA для конкретного символу

**Інтеграція:**
Додати `get_chronos_features()` в `module_ai/data_pipeline.py` як опційний крок.
В `inference.py` — запускати Chronos паралельно з основним GBM.

---

### D3 — Profit-optimized loss function через CatBoost custom objective

CatBoost підтримує кастомний objective. Замість `Quantile:alpha=0.5` → custom loss що напряму оптимізує directional accuracy з урахуванням порогу:

```python
class ProfitAlignedQuantileLoss:
    """
    Поєднує:
    1. Pinball loss (q10/q50/q90) — для calibrated intervals
    2. Direction penalty — штраф за неправильний напрям
    3. Cost awareness — штраф пропорційний commission якщо сигнал занадто малий
    """
    def __init__(self, quantile: float, cost_threshold: float = 0.001):
        self.quantile = quantile
        self.cost_threshold = cost_threshold
    
    def calc_ders_range(self, approxes, targets, weights):
        grad = []
        hess = []
        for pred, true in zip(approxes[0], targets):
            error = true - pred
            # Pinball gradient
            if error > 0:
                g = -self.quantile
            else:
                g = (1 - self.quantile)
            
            # Extra penalty: якщо pred-signal неправильний напрям
            direction_penalty = 0.0
            if abs(pred) < self.cost_threshold and true * pred < 0:
                direction_penalty = 0.5 * np.sign(pred - true)
            
            grad.append(g + direction_penalty)
            hess.append(1.0)
        return grad, hess
```

Застосування:
```python
model = CatBoostRegressor(
    loss_function=ProfitAlignedQuantileLoss(quantile=0.5, cost_threshold=0.001),
    ...
)
```

---

### D4 — Ensemble з vote-weighted combination

Замість одного backend — ensemble із кількох:

```python
class EnsembleForecaster:
    """
    Комбінує прогнози від:
    - gradient_boosting: стабільний baseline
    - lightgbm: швидший, менший overfitting
    - tft: краща часова структура (якщо D1 реалізовано)
    
    Weights навчаються на validation за принципом stacking.
    """
    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = [model.predict(X) for model in self.models]
        # Weighted average, ваги з validation performance
        return np.average(preds, weights=self.weights, axis=0)
```

---

## Послідовність виконання

```
Тиждень 1:
  ├─ A1: approval gate + торгові метрики
  ├─ A2: realistic calibration (transaction costs)
  ├─ A3: soft heuristic gate
  └─ A4: early stopping

Тиждень 2:
  ├─ B1: LightGBM backend
  ├─ C1: encoder_len=24 + trend slope features
  └─ B4: Optuna hyperopt (опціонально)

Тиждень 3:
  ├─ C2: cross-asset features (BTC returns)
  ├─ C3: regime features
  └─ B2: direction classifier

Тиждень 4-5:
  ├─ B3: walk-forward CV
  └─ D2: Chronos features (якщо є GPU/достатній CPU)

Тиждень 6-8:
  └─ D1: TFT (окремий R&D спринт)
```

---

## Метрики успіху

Після кожного milestone перезапускати `calibrate-thresholds` з реалістичними costs і перевіряти:

| Метрика | Поточний baseline (оцінка) | Milestone A | Milestone B | Milestone C/D |
|---|---|---|---|---|
| `directional_accuracy` (val) | ~0.50-0.53 | ≥ 0.53 | ≥ 0.55 | ≥ 0.58 |
| `hit_rate` (val, net of costs) | ~0.45-0.48 | ≥ 0.50 | ≥ 0.52 | ≥ 0.55 |
| `profit_factor` (val, 0.05% cost) | < 1.0 | ≥ 1.0 | ≥ 1.2 | ≥ 1.5 |
| `max_drawdown` (val) | невідомо | ≤ 0.25 | ≤ 0.20 | ≤ 0.15 |
| `sharpe_like` (val) | < 0 | ≥ 0 | ≥ 0.3 | ≥ 0.5 |

**Базова перевірка після кожного milestone:**
```bash
# 1. Retrain
.venv/bin/python cli.py train-model --symbol ALL --model-backend lightgbm

# 2. Evaluate з торговими метриками
.venv/bin/python cli.py evaluate-artifact --symbol ALL --artifact-dir ./artifacts/__multi__ \
  --execution-rule next_open_to_h_close \
  --transaction-cost 0.0005 \
  --slippage 0.0002

# 3. Calibrate з реалістичними costs
.venv/bin/python cli.py calibrate-thresholds --symbol ALL --artifact-dir ./artifacts/__multi__ \
  --execution-rule next_open_to_h_close \
  --transaction-cost 0.0005 \
  --slippage 0.0002

# 4. Перевірити звіт
cat artifacts/__multi__/signal_calibration_report.json | python -m json.tool | head -80
```

---

## Важливі застереження

1. **Ніколи не використовувати test set для вибору гіперпараметрів.** Threshold calibration допустима тільки на validation. Test — тільки для фінального звіту.

2. **Walk-forward важливіший ніж random split.** Ринкові дані мають temporal autocorrelation. K-fold з shuffle — недопустимий для часових рядів.

3. **Directional accuracy 55%+ — реалістична ціль.** Ринок ефективний. 55% при достатньому sample size і realistic costs — це хороший результат. 60%+ — відмінний.

4. **Комісії руйнують стратегії з частими сигналами.** При `horizon_h=6` (6 годин) і `transaction_cost=0.05%` — break-even потребує >0.1% return per trade. Якщо `average_trade_return < 0.1%` у backtест — стратегія збиткова на реальному ринку.

5. **Overfitting до Bitcoin Bull Run 2023-2024.** Якщо majority of training data — bull market, модель буде більше генерувати BUY сигналів і погано торгувати в sideways/bear режимі. Режимні features (Milestone C3) частково вирішують це.
