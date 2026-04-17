# Operator Runbook

Цей файл містить Python-only операторські сценарії для активного контуру:

- `train`
- `eval`
- `calibrate`
- `forecast`
- `bot`

Усі команди запускаються без `source`, shell launcher-ів і сторонніх обгорток.

## Передумови

- ви перебуваєте в корені проєкту
- `.venv` уже створений
- залежності встановлені
- `.env` налаштований
- PostgreSQL доступний

## Install

Створення середовища:

```
python3 -m venv .venv
```

Встановлення залежностей:

```
.venv/bin/python -m pip install -r requirements.txt
```

## Shared Training

Навчання shared multi-symbol artifact:

```
.venv/bin/python cli.py train-model --symbol ALL --save-dir ./artifacts --model-backend catboost --compute-device auto
```

Очікуваний artifact directory:

```text
./artifacts/__multi__
```

## Evaluation And Approval

Evaluation для shared artifact:

```
.venv/bin/python cli.py evaluate-artifact --symbol ALL --artifact-dir ./artifacts/__multi__
```

Перевірка artifact consistency:

```
.venv/bin/python cli.py validate-artifact --artifact-dir ./artifacts/__multi__
```

Коротка operator summary:

```
.venv/bin/python cli.py artifact-info --artifact-dir ./artifacts/__multi__
```

## Threshold Calibration

Calibration `forecast -> signal -> trades` поверх уже навченого artifact:

```
.venv/bin/python cli.py calibrate-thresholds --symbol ALL --artifact-dir ./artifacts/__multi__
```

Очікуваний report:

```text
./artifacts/__multi__/signal_calibration_report.json
```

## Forecast

Inference для окремого символа через shared artifact:

```
.venv/bin/python cli.py forecast --symbol SOLUSDT --model-dir ./artifacts/__multi__
```

## Strategy / Bot

Strategy-layer запуск для окремого символа:

```
.venv/bin/python cli.py start-bot --symbol SOLUSDT --artifact-dir ./artifacts/__multi__
```

## Detached Benchmark

Запуск real benchmark у background:

```
.venv/bin/python cli.py launch-benchmark --symbols ALL --model-backends gradient_boosting,catboost --compare-per-symbol-baseline --compute-device auto --output-dir ./artifacts/benchmarks/real_db_run
```

Статус:

```
.venv/bin/python cli.py benchmark-status --output-dir ./artifacts/benchmarks/real_db_run
```

Останні рядки логу:

```
.venv/bin/python cli.py benchmark-log --output-dir ./artifacts/benchmarks/real_db_run --lines 80
```

Зупинка:

```
.venv/bin/python cli.py benchmark-stop --output-dir ./artifacts/benchmarks/real_db_run
```

## Recommended Operator Sequence

1. `train-model`
2. `evaluate-artifact`
3. `validate-artifact`
4. `artifact-info`
5. `calibrate-thresholds`
6. `forecast`
7. `start-bot`

## GPU Notes

- Use `--model-backend catboost --compute-device auto` for automatic NVIDIA GPU training with CPU fallback
- Use `--compute-device gpu` only when GPU training is mandatory
- `gradient_boosting` remains CPU-only even when `--compute-device auto`
