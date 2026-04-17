from utils.config import ORDERFLOW_DB_SCHEMA


def get_schema_statement() -> str:
    return f"""
        CREATE SCHEMA IF NOT EXISTS {ORDERFLOW_DB_SCHEMA};
        """


def get_table_statements() -> dict[str, str]:
    return {
        "order_events": f"""
        CREATE TABLE IF NOT EXISTS {ORDERFLOW_DB_SCHEMA}.order_events (
            id BIGSERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            symbol TEXT NOT NULL,
            event_type TEXT NOT NULL,
            payload JSONB NOT NULL DEFAULT '{{}}'::jsonb
        );
        """,
        "position_events": f"""
        CREATE TABLE IF NOT EXISTS {ORDERFLOW_DB_SCHEMA}.position_events (
            id BIGSERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            symbol TEXT NOT NULL,
            event_type TEXT NOT NULL,
            side TEXT,
            entry_price NUMERIC,
            mark_price NUMERIC,
            stop_price NUMERIC,
            take_profit_price NUMERIC,
            size NUMERIC,
            hold_time_ms BIGINT,
            reason TEXT,
            payload JSONB NOT NULL DEFAULT '{{}}'::jsonb
        );
        """,
        "runtime_transitions": f"""
        CREATE TABLE IF NOT EXISTS {ORDERFLOW_DB_SCHEMA}.runtime_transitions (
            id BIGSERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            symbol TEXT NOT NULL,
            from_state TEXT,
            to_state TEXT NOT NULL,
            reason TEXT,
            payload JSONB NOT NULL DEFAULT '{{}}'::jsonb
        );
        """,
    }


def get_index_statements() -> dict[str, str]:
    return {
        "order_events_symbol_created_at_idx": f"""
        CREATE INDEX IF NOT EXISTS order_events_symbol_created_at_idx
        ON {ORDERFLOW_DB_SCHEMA}.order_events (symbol, created_at DESC);
        """,
        "order_events_event_type_created_at_idx": f"""
        CREATE INDEX IF NOT EXISTS order_events_event_type_created_at_idx
        ON {ORDERFLOW_DB_SCHEMA}.order_events (event_type, created_at DESC);
        """,
        "position_events_symbol_created_at_idx": f"""
        CREATE INDEX IF NOT EXISTS position_events_symbol_created_at_idx
        ON {ORDERFLOW_DB_SCHEMA}.position_events (symbol, created_at DESC);
        """,
        "position_events_event_type_created_at_idx": f"""
        CREATE INDEX IF NOT EXISTS position_events_event_type_created_at_idx
        ON {ORDERFLOW_DB_SCHEMA}.position_events (event_type, created_at DESC);
        """,
        "runtime_transitions_symbol_created_at_idx": f"""
        CREATE INDEX IF NOT EXISTS runtime_transitions_symbol_created_at_idx
        ON {ORDERFLOW_DB_SCHEMA}.runtime_transitions (symbol, created_at DESC);
        """,
        "runtime_transitions_to_state_created_at_idx": f"""
        CREATE INDEX IF NOT EXISTS runtime_transitions_to_state_created_at_idx
        ON {ORDERFLOW_DB_SCHEMA}.runtime_transitions (to_state, created_at DESC);
        """,
    }
