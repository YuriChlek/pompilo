### Команда для початку тренування моделі 

```bash
python model_train.py   --db postgresql://admin:admin_pass@localhost:5432/pompilo_db   --table _candles_trading_data.solusdt_p_candles   --epochs 1000   --batch-size 128   --encoder-len 120   --pred-len 6
```

### Команда для тестового запуску скрипта 

```bash
python get_forecast.py --db postgresql://admin:admin_pass@localhost:5432/pompilo_db  --table _candles_trading_data.solusdt_p_candles --model-dir "./tft_runs" --symbol "SOLUSDT"
```
