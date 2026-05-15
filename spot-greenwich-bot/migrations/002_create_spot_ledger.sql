DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.schemata
        WHERE schema_name = '_spot_trading_bot'
    ) THEN
        EXECUTE 'CREATE SCHEMA _spot_trading_bot';
    END IF;
END$$;

CREATE TABLE IF NOT EXISTS _spot_trading_bot.position_state (
    symbol TEXT PRIMARY KEY,
    quantity NUMERIC NOT NULL DEFAULT 0,
    avg_entry_price NUMERIC NOT NULL DEFAULT 0,
    total_cost NUMERIC NOT NULL DEFAULT 0,
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS _spot_trading_bot.order_ledger (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    status TEXT NOT NULL,
    signal_price NUMERIC NOT NULL,
    executed_price NUMERIC,
    quantity NUMERIC NOT NULL,
    quote_amount NUMERIC NOT NULL,
    avg_entry_price_before NUMERIC NOT NULL DEFAULT 0,
    avg_entry_price_after NUMERIC NOT NULL DEFAULT 0,
    exchange_order_id TEXT,
    exchange_payload JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
