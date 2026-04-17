# Real DB Benchmark Runbook

Цей файл містить Python-only сценарій запуску реального benchmark на вашій БД, з прогресом, логом, `PID` і поверненням prompt одразу після старту.

## Мета

Потрібно завершити незакритий пункт roadmap:

- прогнати benchmark на реальних даних з БД
- визначити поточний `production winner model`

## Передумови

Перед запуском переконайтесь, що:

- ви перебуваєте в корені проєкту
- `.venv` існує
- залежності встановлені
- `.env` налаштований
- PostgreSQL доступний з вашої консолі

## Рекомендований надійний запуск

Рекомендований варіант не тримає консоль зайнятою після старту процесу і не використовує зовнішні shell launcher-и.

Запуск:

```
.venv/bin/python cli.py launch-benchmark --symbols ALL --model-backends gradient_boosting,catboost --compare-per-symbol-baseline --compute-device auto --output-dir ./artifacts/benchmarks/real_db_run
```

Python launcher:

- запускає benchmark у фоні
- одразу повертає prompt
- записує лог у файл
- зберігає `PID`
- друкує Python-команди для status/log/stop
- не залишає процес "висіти" в поточній консолі після запуску

Дефолтно він запускає:

```
SYMBOLS=ALL
MODEL_BACKENDS=gradient_boosting,catboost
COMPARE_PER_SYMBOL_BASELINE=True
OUTPUT_DIR=./artifacts/benchmarks/real_db_run
```

За потреби можна перевизначити параметри:

```
.venv/bin/python cli.py launch-benchmark --symbols ALL --model-backends gradient_boosting,catboost --compute-device auto --output-dir ./artifacts/benchmarks/real_db_shared
```

## Точна команда для foreground запуску

Якщо все ж потрібен foreground запуск:

```
.venv/bin/python cli.py benchmark-models --symbols ALL --model-backends gradient_boosting,catboost --compare-per-symbol-baseline --compute-device auto --output-dir ./artifacts/benchmarks/real_db_run
```

## Прогрес під час виконання

Тепер benchmark друкує:

- stage progress для shared backend-ів
- progress bar для `per-symbol baseline`
- оновлення `partial report` після кожного завершеного backend-а

Тобто навіть якщо повний прогін довгий, у логах буде видно, на якому етапі він зараз.

Щоб подивитися останні рядки логу:

```
.venv/bin/python cli.py benchmark-log --output-dir ./artifacts/benchmarks/real_db_run --lines 80
```

Щоб перевірити, чи процес ще працює:

```
.venv/bin/python cli.py benchmark-status --output-dir ./artifacts/benchmarks/real_db_run
```

Щоб зупинити benchmark вручну:

```
.venv/bin/python cli.py benchmark-stop --output-dir ./artifacts/benchmarks/real_db_run
```

## Якщо повний benchmark занадто довгий

Можна спочатку запустити коротший shared-only benchmark:

```
.venv/bin/python cli.py benchmark-models --symbols ALL --model-backends gradient_boosting,catboost --compute-device auto --output-dir ./artifacts/benchmarks/real_db_shared
```

Цей варіант дозволить швидше визначити winner backend для shared model.

## Що з'явиться в результаті

Після запуску detached-режиму:

- лог:

```text
./artifacts/benchmarks/real_db_run/benchmark.log
```

- pid:

```text
./artifacts/benchmarks/real_db_run/benchmark.pid
```

- основний звіт:

```text
./artifacts/benchmarks/real_db_run/model_benchmark_report.json
```

Partial report може з’явитися ще до завершення всього процесу.

Перевірити проміжний стан можна так:

```
.venv/bin/python cli.py benchmark-status --output-dir ./artifacts/benchmarks/real_db_run
```

Після завершення основний звіт буде тут:

```text
./artifacts/benchmarks/real_db_run/model_benchmark_report.json
```

Або для короткого режиму:

```text
./artifacts/benchmarks/real_db_shared/model_benchmark_report.json
```

## Що мені потрібно від вас після запуску

Після завершення надішліть мені один з варіантів:

1. Сам файл `model_benchmark_report.json`

або

2. Або хоча б короткий вивід цієї команди:

```
.venv/bin/python cli.py benchmark-status --output-dir ./artifacts/benchmarks/real_db_run
```

## Якщо benchmark впаде

Тоді надішліть мені:

```
.venv/bin/python cli.py benchmark-status --output-dir ./artifacts/benchmarks/real_db_run
```

а також:

```
.venv/bin/python cli.py benchmark-log --output-dir ./artifacts/benchmarks/real_db_run --lines 200
```

або для shared-only режиму:

```
.venv/bin/python cli.py benchmark-log --output-dir ./artifacts/benchmarks/real_db_shared --lines 200
```

## Очікуваний наступний крок

Після того як ви дасте мені `model_benchmark_report.json`, я:

- визначу winner backend
- скажу, чи shared model достатня, чи вже видно потребу в hybrid deployment
- оновлю roadmap/documentation і зафіксую production candidate
