DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.schemata
        WHERE schema_name = '_candles_trading_data'
    ) THEN
        EXECUTE 'CREATE SCHEMA _candles_trading_data';
    END IF;
END$$;

CREATE TABLE IF NOT EXISTS _candles_trading_data.ethusdt_p_candles_d1 (
    open_time TIMESTAMP NOT NULL,
    close_time TIMESTAMP NOT NULL,
    symbol TEXT NOT NULL,
    open NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    cvd NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,
    candle_id TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS _candles_trading_data.suiusdt_p_candles_d1 (
    open_time TIMESTAMP NOT NULL,
    close_time TIMESTAMP NOT NULL,
    symbol TEXT NOT NULL,
    open NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    cvd NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,
    candle_id TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS _candles_trading_data.taousdt_p_candles_d1 (
    open_time TIMESTAMP NOT NULL,
    close_time TIMESTAMP NOT NULL,
    symbol TEXT NOT NULL,
    open NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    cvd NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,
    candle_id TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS _candles_trading_data.solusdt_p_candles_d1 (
    open_time TIMESTAMP NOT NULL,
    close_time TIMESTAMP NOT NULL,
    symbol TEXT NOT NULL,
    open NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    cvd NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,
    candle_id TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS _candles_trading_data.btcusdt_p_candles_d1 (
    open_time TIMESTAMP NOT NULL,
    close_time TIMESTAMP NOT NULL,
    symbol TEXT NOT NULL,
    open NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    cvd NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,
    candle_id TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS _candles_trading_data.xrpusdt_p_candles_d1 (
    open_time TIMESTAMP NOT NULL,
    close_time TIMESTAMP NOT NULL,
    symbol TEXT NOT NULL,
    open NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    cvd NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,
    candle_id TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS _candles_trading_data.ltcusdt_p_candles_d1 (
    open_time TIMESTAMP NOT NULL,
    close_time TIMESTAMP NOT NULL,
    symbol TEXT NOT NULL,
    open NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    cvd NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,
    candle_id TEXT NOT NULL UNIQUE
);
