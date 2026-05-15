CREATE TABLE IF NOT EXISTS _spot_trading_bot.position_exit_state (
    symbol TEXT PRIMARY KEY,
    position_opened_at TIMESTAMP,
    initial_quantity NUMERIC NOT NULL DEFAULT 0,
    first_take_profit_done BOOLEAN NOT NULL DEFAULT FALSE,
    first_take_profit_order_id TEXT,
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);
