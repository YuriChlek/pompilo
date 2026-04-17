# Technical Roadmap

## Status Snapshot

Станом на поточну ітерацію:

- `Milestone 1`: done
- `Milestone 2`: done
- `Milestone 3`: done
- `Milestone 4`: done
- `Milestone 5`: in progress

Проміжний результат по реальному benchmark subset:

- benchmark завершено на subset `ETHUSDT`, `XRPUSDT`, `SOLUSDT`;
- проміжний winner backend: `gradient_boosting`;
- проміжний deployment mode: `shared`;
- обидва backend-и пройшли gate, але `gradient_boosting` має трохи кращий `final_test_rmse`;
- це ще не фінальний production verdict для всіх `TRADING_SYMBOLS`, а лише зафіксований interim result.

Що вже закрито:

- multi-symbol data loading;
- multi-symbol feature generation;
- `symbol_id` і `per-symbol` scaling;
- shared train/eval/forecast artifact path;
- per-symbol approval gate;
- benchmark `gradient_boosting` vs `catboost`;
- hybrid comparison: shared model vs per-symbol baselines.
- grouped reports за volatility / liquidity regime.

Що ще лишається в межах `Milestone 5`:

- прогнати benchmark на повнішому production-representative universe;
- підтвердити або переглянути interim winner model на ширшому наборі символів;
- за потреби додати `LightGBM` або `XGBoost`;

Поточний наступний робочий напрямок після фіксації interim winner:

- зібрати новий active shared artifact у `artifacts/__multi__` через `train -> eval`;
- перевірити, що artifact став `approved/deployable`;
- calibration `forecast -> signal -> trades` на `validation`;
- окреме підтвердження top threshold candidates на `final test`;
- підготовка thresholds для переносу в strategy layer без leakage.

Поточна практична послідовність робіт:

1. `train-model --symbol ALL --save-dir ./artifacts --model-backend gradient_boosting`
2. `evaluate-artifact --symbol ALL --artifact-dir ./artifacts/__multi__`
3. `validate-artifact --artifact-dir ./artifacts/__multi__`
4. `artifact-info --artifact-dir ./artifacts/__multi__`
5. `calibrate-thresholds --symbol ALL --artifact-dir ./artifacts/__multi__`
6. `forecast --symbol <SYMBOL> --model-dir ./artifacts/__multi__`

Поточне робоче припущення до повного rerun benchmark на всьому universe:

- backend для shared artifact: `gradient_boosting`
- model type: `gradient_boosting_quantile`
- deployment mode target: `shared`

## Goal

Перебудувати поточний `single-symbol` ML pipeline в `multi-symbol` pipeline, який:

- навчається на всіх символах із `TRADING_SYMBOLS`;
- коректно працює з різними масштабами цін, об'ємів і волатильності;
- використовує `future_return_h` як основну target;
- враховує `symbol_id` як окрему categorical feature;
- оцінюється як глобально, так і окремо по кожному символу.

## Current State

Поточний active path:

- `cli.py`
- `module_ai/train.py`
- `module_ai/evaluate_run.py`
- `module_ai/forecast.py`
- `module_signal/bot.py`

Поточні обмеження:

- train path працює лише з одним символом;
- split logic працює лише з одним символом;
- windowing працює лише з одним символом;
- scaler metadata і model metadata прив'язані до одного символа;
- inference path очікує `artifacts/<symbol>`;
- evaluation не перевіряє multi-symbol quality окремо по активах.

## Target Modeling Strategy

### Main Target

Основна target:

- `target_return_h = close[t + H] / close[t] - 1`

Причини:

- target scale-invariant;
- підходить для активів з різними рівнями цін;
- дозволяє пізніше отримувати `BUY / SELL / HOLD` через пороги;
- не втрачає інформацію про силу руху, як це робить чистий `up/down`.

### Secondary Evaluation Targets

Для evaluation та strategy layer використовувати похідні label-и:

- `BUY`, якщо `target_return_h > positive_threshold`
- `SELL`, якщо `target_return_h < -positive_threshold`
- `HOLD`, якщо `target_return_h` у нейтральній зоні

Пороги мають визначатися з урахуванням:

- transaction cost;
- slippage;
- середньої волатильності на горизонті `H`.

### Modeling Priority

Порядок моделей:

1. Поточний boosting baseline
2. `CatBoost`
3. `LightGBM` або `XGBoost`
4. Лише після цього sequence models

`Transformer` не є стартовою ціллю. Спершу потрібен сильний `tabular multi-symbol baseline`.

## Dataset Redesign

### Data Sources

Джерело символів:

- `utils.config.TRADING_SYMBOLS`

Очікувана таблиця для кожного символа:

- `_candles_trading_data.<symbol_lowercase>_p_candles`

### Canonical Multi-Symbol Frame

Новий канонічний raw dataset повинен містити:

- `symbol`
- `open_time`
- `close_time`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `cvd`

Після злиття по всіх символах усі операції над time series мають виконуватись через `groupby("symbol")`.

### Feature Rules

Для multi-symbol режиму не можна покладатися лише на абсолютні:

- `open`
- `high`
- `low`
- `close`
- `volume`

Пріоритетні типи фіч:

- returns: `return_1`, `return_3`, `return_6`
- price ratios: `high_close_ratio`, `low_close_ratio`, `open_close_ratio`
- distance to rolling mean: `close_vs_ma_20`, `close_vs_ma_50`
- rolling volatility
- `volume / volume_ma`
- `cvd_change`
- rolling z-score для ціни
- rolling z-score для об'єму
- rolling z-score для `cvd`
- existing cyclic time features

### Symbol Feature

Потрібно додати:

- `symbol_id`

Використання:

- як categorical feature у моделі;
- як частина metadata артефакту;
- як частина inference contract.

### Scaling Strategy

Нормалізація має бути `per-symbol`, а не глобальна на весь датасет.

Правила:

- scaler fit виконується окремо в межах кожного символа;
- metadata повинна зберігати список символів і правила scaling;
- якщо фіча вже scale-invariant, додатковий scaler не є обов'язковим.

## Split Strategy

### Main Split

Основний split:

- `train`
- `validation`
- `test`

Правила:

- лише часовий split;
- однакові часові межі для всіх символів;
- без shuffle;
- без mixing future data у train.

### Additional Validation Views

Окрім глобального test, потрібні:

- per-symbol metrics;
- metrics для груп символів за волатильністю;
- metrics для груп символів за liquidity / volume regime.

### Optional Generalization Validation

Після стабілізації основного pipeline додати експериментальний режим:

- `leave-symbol-out`

Мета:

- перевірити, чи модель узагальнює на символи, яких не було у train.

Це не blocker для першої production-ready версії.

## Training Strategy

### Phase 1

Побудувати `multi-symbol` baseline без зміни strategy layer.

Scope:

- train на всіх `TRADING_SYMBOLS`;
- `target_return_h`;
- evaluation на глобальному `validation/test`;
- per-symbol reports;
- один shared artifact bundle.

### Phase 2

Додати сильнішу модель.

Scope:

- `CatBoost` або `LightGBM`;
- підтримка `symbol_id`;
- порівняння з поточним boosting baseline;
- збереження unified evaluation report.

### Phase 3

Перевірити hybrid setup.

Scope:

- одна shared global model;
- окремі per-symbol baselines;
- порівняння shared vs per-symbol.

Рішення:

- якщо shared model стабільно краща або не гірша, залишити shared path;
- якщо окремі ключові символи деградують, дозволити hybrid deployment.

## Inference Strategy

Inference повинен лишитися `single-symbol` з точки зору запиту, але використовувати shared artifact.

Запит:

- `symbol`
- останні свічки цього символа

Shared artifact має містити:

- список символів train universe;
- feature contract;
- symbol encoder;
- scaler metadata;
- model bundle;
- evaluation summary;
- per-symbol evaluation summary.

Під час inference:

- символ має бути присутній у `train universe`;
- preprocessing виконується лише по цьому символу;
- `symbol_id` має бути побудований так само, як у train.

## Evaluation Strategy

### Forecast Metrics

Обов'язкові метрики:

- `MAE`
- `RMSE`
- interval coverage, якщо лишається quantile setup
- pinball loss, якщо модель лишається quantile-aware

### Direction / Decision Metrics

Додатково:

- directional accuracy
- `BUY` precision / recall
- `SELL` precision / recall
- `HOLD` share

### Trading Metrics

Потрібно зберегти offline trading simulation:

- cumulative return
- hit rate
- max drawdown
- turnover
- profit factor

### Reporting Dimensions

Кожен evaluation report повинен містити:

- global metrics
- per-symbol metrics
- ranking symbols by performance
- список worst symbols
- список best symbols

## Required Refactoring by Module

### `utils/config.py`

Завдання:

- зафіксувати `TRADING_SYMBOLS` як train universe;
- прибрати неоднозначність між general symbols і AI symbols у новому train path.

### `module_ai/data_access.py`

Завдання:

- додати multi-symbol loader;
- валідувати наявність таблиць для кожного символа;
- повертати один об'єднаний dataframe;
- зберегти існуючий single-symbol loader для inference.

### `module_ai/data_pipeline.py`

Завдання:

- додати multi-symbol friendly feature set;
- підтримати `symbol_id`;
- переробити scaler metadata під список символів;
- підтримати per-symbol scaling.

### `module_ai/splits.py`

Завдання:

- прибрати жорстке `single-symbol` обмеження для основного split path;
- додати global time split для multi-symbol data;
- зберегти symbol-level consistency.

### `module_ai/windowing.py`

Завдання:

- будувати windows всередині кожного символа;
- не допускати перетікання history між різними символами;
- включити `symbol_id` у feature vector або у model input contract.

### `module_ai/modeling.py`

Завдання:

- зберегти baseline;
- додати новий model backend;
- привести model metadata до multi-symbol format.

### `module_ai/train.py`

Завдання:

- замінити `--symbol` train path на train universe з `TRADING_SYMBOLS`;
- підтримати опційний список символів для ablation/experiments;
- зберігати shared artifact directory.

### `module_ai/evaluate_run.py`

Завдання:

- оцінювати shared artifact;
- додати per-symbol section у report;
- approval gate будувати не лише на global RMSE, а й на stability across symbols.

### `module_ai/forecast.py`

Завдання:

- використовувати shared artifact;
- перевіряти, що запитаний символ входить у train universe;
- правильно відновлювати `symbol_id` і symbol-level scaling.

### `module_signal/bot.py`

Завдання:

- залишити strategy layer окремо від ML;
- адаптувати лише контракт forecast input, якщо зміниться artifact layout.

## Artifact Design

Новий shared artifact повинен містити:

- `dataset_meta.pkl`
- `scalers.pkl`
- `forecast_model.joblib`
- `evaluation_report.pkl`
- за потреби `symbol_encoder.pkl`

Обов'язкові metadata fields:

- `training_symbols`
- `symbol_count`
- `feature_version`
- `dataset_contract_version`
- `evaluation_contract_version`
- `target_name`
- `horizon_h`
- `encoder_len`
- `supported_inference_entrypoint`
- `per_symbol_metrics`
- `global_metrics`

## Delivery Plan

### Milestone 1

Multi-symbol dataset loading and feature generation.

Definition of done:

- можна побудувати один feature dataframe по всіх `TRADING_SYMBOLS`;
- усі фічі рахуються без mixing між символами;
- metadata знає весь список символів.

### Milestone 2

Multi-symbol splitting and training baseline.

Definition of done:

- train запускається на всьому universe;
- зберігається shared artifact;
- baseline model дає валідний forecast output.

### Milestone 3

Evaluation and approval gate redesign.

Definition of done:

- є global metrics;
- є per-symbol metrics;
- approval gate не пропускає artifact із сильною деградацією на частині символів.

### Milestone 4

Inference migration to shared artifact.

Definition of done:

- `forecast` працює для будь-якого символа з train universe;
- preprocessing і feature contract збігаються з train;
- strategy layer продовжує працювати без змішування ML і heuristics.

### Milestone 5

Model comparison and optimization.

Definition of done:

- baseline порівняний з `CatBoost` або `LightGBM`;
- прийнято рішення про production candidate;
- зафіксовано winner model і її constraints.

Поточний статус milestone:

- interim winner already fixed for subset `ETHUSDT`, `XRPUSDT`, `SOLUSDT`;
- threshold calibration is now the active next step on top of the shared artifact;
- full milestone closure still depends on a wider production-representative benchmark rerun.

## Risks

- leakage між символами через неправильний split;
- leakage через глобальний scaler;
- деградація на low-volume symbols;
- домінування кількох символів у глобальній метриці;
- артефакт стане несумісним із поточним inference contract;
- shared model може бути гіршою за per-symbol baseline на частині активів.

## Non-Goals for First Iteration

- transformer-based modeling;
- online learning;
- automated hyperparameter search на великому scale;
- full portfolio optimization;
- multi-exchange feature fusion.

## Working Rules for Future Changes

- не змішувати train і inference refactor в одному великому кроці без проміжної валідації;
- кожен етап завершувати compile/test/evaluation перевіркою;
- не ламати strategy layer, поки forecast contract не стабілізовано;
- кожен structural change має оновлювати artifact metadata contract;
- global metric без per-symbol metric не вважається достатньою валідацією.
