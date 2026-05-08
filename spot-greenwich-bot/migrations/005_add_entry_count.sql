ALTER TABLE _spot_trading_bot.position_state
    ADD COLUMN IF NOT EXISTS entry_count INTEGER NOT NULL DEFAULT 0;
