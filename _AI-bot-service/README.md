# Pompilo AI Bot

Проєкт перебудований у `multi-symbol` ML pipeline для прогнозу `future_return_h` по всіх символах із `TRADING_SYMBOLS`, із shared artifact у `artifacts/__multi__`.

## Current Workflow

Основний production-like контур:

```text
train -> eval -> forecast -> bot
```

Що робить система:

- ML навчається на `multi-symbol` датасеті й прогнозує `q10 / q50 / q90` для `target_return_h`
- strategy layer окремо перетворює forecast у `BUY / SELL / HOLD`
- approval gate перевіряє як global метрики, так і `per-symbol` stability

## Requirements

- Python `3.11+`
- PostgreSQL з candle tables у схемі `_candles_trading_data`
- змінні середовища:
  - `DB_HOST`
  - `DB_PORT`
  - `DB_USER`
  - `DB_PASS`
  - `DATABASE`

Install:

```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Data Layout

Для кожного символа очікується таблиця:

```text
_candles_trading_data.<symbol_lowercase>_p_candles
```

Train universe береться з `utils/config.py -> TRADING_SYMBOLS`.

## Quick Start

Shared training on all configured symbols:

```
.venv/bin/python cli.py train-model --symbol ALL --model-backend catboost --compute-device auto
.venv/bin/python cli.py evaluate-artifact --symbol ALL --artifact-dir ./artifacts/__multi__
.venv/bin/python cli.py forecast --symbol SOLUSDT --model-dir ./artifacts/__multi__
.venv/bin/python cli.py start-bot --symbol SOLUSDT --artifact-dir ./artifacts/__multi__
```

Single-symbol experiment:

```
.venv/bin/python cli.py train-model --symbol SOLUSDT
.venv/bin/python cli.py evaluate-artifact --symbol SOLUSDT --artifact-dir ./artifacts/solusdt
```

Model comparison:

```
.venv/bin/python cli.py benchmark-models --symbols ALL --model-backends gradient_boosting,catboost --compare-per-symbol-baseline --compute-device auto
```

Signal threshold calibration:

```
.venv/bin/python cli.py calibrate-thresholds --symbol ALL --artifact-dir ./artifacts/__multi__
```

Detached benchmark launch:

```
.venv/bin/python cli.py launch-benchmark --symbols ALL --model-backends gradient_boosting,catboost --compare-per-symbol-baseline --output-dir ./artifacts/benchmarks/real_db_run
```

Benchmark run management:

```
.venv/bin/python cli.py benchmark-status --output-dir ./artifacts/benchmarks/real_db_run
.venv/bin/python cli.py benchmark-log --output-dir ./artifacts/benchmarks/real_db_run --lines 80
.venv/bin/python cli.py benchmark-stop --output-dir ./artifacts/benchmarks/real_db_run
```

Python CLI is the only supported operator entrypoint.

Unified operator workflow:

- [OPERATOR_RUNBOOK.md](/home/yurii/Proj/pompilo/_AI-bot/OPERATOR_RUNBOOK.md)

## GPU Training

GPU-aware training is controlled through:

- `--compute-device auto|cpu|gpu`
- `--gpu-device-id <N>`

Behavior:

- `auto`: use NVIDIA GPU when the selected backend supports it and a visible GPU is available
- `cpu`: force CPU training
- `gpu`: require NVIDIA GPU training, fail if unavailable

Current backend support:

- `catboost`: CPU and GPU
- `gradient_boosting`: CPU only

The detached launcher:

- starts benchmark in background
- returns prompt immediately
- writes log, PID and launcher state
- can be inspected and stopped through Python CLI commands only

## Artifact Contract

Shared artifact directory містить:

- `dataset_meta.pkl`
- `scalers.pkl`
- `forecast_model.joblib`
- `evaluation_report.pkl`

Ключові metadata поля:

- `training_symbols`
- `symbol_count`
- `symbol_id_map`
- `scaling_scope`
- `model_backend`
- `evaluation_approved`
- `deployable`

## Benchmarking

Поточний benchmark stage вміє:

- порівнювати `gradient_boosting` vs `catboost`
- будувати recommendation report
- додатково порівнювати `shared` модель з `per-symbol` baseline моделями
- будувати grouped reports за volatility / liquidity regime
- писати partial report після кожного завершеного backend-а

Output:

- `artifacts/benchmarks/model_benchmark_report.json`

## Signal Calibration

Поточний calibration stage вміє:

- брати вже навчений artifact без retrain
- будувати validation і final-test forecasts
- перебирати candidate policies для `quantile_barrier` і `median_with_width`
- ранжувати candidates тільки по `validation`
- окремо показувати confirmation на `final test`

Default output:

- `artifacts/__multi__/signal_calibration_report.json`

Interim real-DB benchmark result currently fixed in the project:

- subset: `ETHUSDT`, `XRPUSDT`, `SOLUSDT`
- recommended backend: `gradient_boosting`
- recommended model type: `gradient_boosting_quantile`
- recommended deployment mode: `shared`
- note: this is an interim subset-level result, not the final verdict for the full `TRADING_SYMBOLS` universe

## Roadmap Status

Поточний статус:

- `Milestone 1`: done
- `Milestone 2`: done
- `Milestone 3`: done
- `Milestone 4`: done
- `Milestone 5`: in progress

Що вже реалізовано:

- multi-symbol loading
- multi-symbol features
- `symbol_id`
- `per-symbol` scaling
- shared train/eval/forecast
- per-symbol approval gate
- backend comparison
- hybrid comparison: shared vs per-symbol baselines
- grouped regime reports

Що ще не завершено:

- підтвердження interim winner на ширшому production-representative universe
- optional `LightGBM` / `XGBoost`

## Notes

- `forecast` і `bot` працюють лише з approved/deployable artifact
- ML не приймає фінальне торгове рішення, лише повертає forecast
- global метрика без `per-symbol` аналізу не вважається достатньою валідацією
