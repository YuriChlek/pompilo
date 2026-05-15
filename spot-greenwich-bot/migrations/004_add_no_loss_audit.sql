ALTER TABLE _spot_trading_bot.order_ledger
    ADD COLUMN IF NOT EXISTS realized_pnl_usdt NUMERIC,
    ADD COLUMN IF NOT EXISTS realized_pnl_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS no_loss_check_price NUMERIC,
    ADD COLUMN IF NOT EXISTS signal_timeframe TEXT,
    ADD COLUMN IF NOT EXISTS signal_candle_id TEXT;

CREATE INDEX IF NOT EXISTS order_ledger_signal_dedupe_idx
    ON _spot_trading_bot.order_ledger (symbol, side, status, signal_timeframe, signal_candle_id);
