# Backtesting Module

## Призначення

`backtesting/` проганяє поточну live-стратегію `trend_breakout` на історичних H1-свічках із PostgreSQL без взаємодії з реальною біржею.

Модуль використовує той самий signal path, що і live runtime, але замість Bybit execution застосовує локальний replay simulator.

## Що вміє модуль

- завантажувати історію свічок із PostgreSQL;
- відтворювати ринок свічка за свічкою;
- генерувати ті самі breakout-сигнали, що й live bot;
- симулювати `Market` і `Limit` fills;
- закривати позиції по `take_profit`, `stop_loss`, `signal_reversal`, `end_of_data`;
- збирати symbol-level і portfolio-level статистику.

## Поточні метрики

Базовий summary:
- `total_trades`
- `profitable_trades`
- `losing_trades`
- `breakeven_trades`
- `max_profit_streak`
- `max_loss_streak`
- `average_profit_pct`
- `average_loss_pct`
- `average_bars_held`
- `net_pnl_pct`

Розширена аналітика:
- `strategy_stats`
- `signal_counts_by_strategy`
- `filled_order_counts_by_strategy`
- `skipped_signal_counts`
- `exit_reason_counts`
- `pnl_by_regime`
- `expectancy_by_setup`
- `max_drawdown_by_cluster`

Trade journal тепер також зберігає:
- `regime`
- `setup_type`
- `cluster`
- `initial_risk_distance`
- `r_multiple`

## Основні файли

- `models.py` — dataclass-моделі конфігів, угод і результатів
- `data_loader.py` — читання свічок із PostgreSQL
- `adapters.py` — адаптація live signal logic для replay
- `execution.py` — локальний execution simulator
- `replay.py` — покроковий replay по свічках
- `portfolio.py` — portfolio aggregation
- `reporting.py` — статистика й JSON-safe serialization
- `runner.py` — orchestration і CLI report
- `run_backtest.py` — окремий CLI entrypoint

## Запуск

Один символ:

```bash
PYTHONPATH=. ./.venv/bin/python backtesting/run_backtest.py \
  --symbol SOLUSDT \
  --from 2026-03-07 \
  --to 2026-03-21
```

Кілька символів:

```bash
PYTHONPATH=. ./.venv/bin/python backtesting/run_backtest.py \
  --symbol SOLUSDT \
  --symbol ETHUSDT \
  --from 2026-01-01 \
  --to 2026-03-01
```

JSON export:

```bash
PYTHONPATH=. ./.venv/bin/python backtesting/run_backtest.py \
  --symbol SOLUSDT \
  --from 2026-03-07 \
  --to 2026-03-21 \
  --output-json artifacts/backtest_solusdt.json
```

## CLI параметри

- `--symbol`
- `--from`
- `--to`
- `--lookback-candles`
- `--min-candles`
- `--indicator-history-period`
- `--no-reversal`
- `--intrabar-exit-priority`
- `--output-json`

## Поточні обмеження

- комісії, funding і slippage не враховуються;
- replay не моделює реальний live trailing lifecycle крок за кроком всередині свічки;
- часткові виходи в backtest simulator поки не моделюються окремими sub-trades;
- drawdown рахується по cumulative trade PnL per cluster, а не як повна equity curve портфеля.

## Залежності

Для запуску потрібні:
- PostgreSQL з історичними свічками у схемі `_candles_trading_data`;
- коректні `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASS`, `DATABASE`;
- інтерпретатор `./.venv/bin/python`.
