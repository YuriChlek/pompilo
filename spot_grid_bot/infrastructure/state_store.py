from __future__ import annotations

import json

from domain.models import RegimeType, RiskRuntimeState, StrategyState, SymbolRuntimeState
from infrastructure.db import create_connection
from utils.config import STATE_SCHEMA, STATE_TABLE


class PostgresStateStore:
    """PostgreSQL-backed persistence adapter for per-symbol runtime state."""

    async def initialize(self) -> None:
        """Create the runtime state schema and table when they do not exist."""
        conn = await create_connection()
        try:
            await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {STATE_SCHEMA}")
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {STATE_SCHEMA}.{STATE_TABLE} (
                    symbol TEXT PRIMARY KEY,
                    regime TEXT NOT NULL,
                    bars_in_state INTEGER NOT NULL DEFAULT 0,
                    cooldown_remaining INTEGER NOT NULL DEFAULT 0,
                    volatility_cooldown_remaining INTEGER NOT NULL DEFAULT 0,
                    pending_regime TEXT NULL,
                    pending_count INTEGER NOT NULL DEFAULT 0,
                    last_rebuild_price DOUBLE PRECISION NULL,
                    kill_switch_count INTEGER NOT NULL DEFAULT 0,
                    recent_equity JSONB NOT NULL DEFAULT '[]'::jsonb,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        finally:
            await conn.close()

    async def load_symbol_state(self, symbol: str) -> SymbolRuntimeState | None:
        """Load persisted runtime state for one symbol when a row exists."""
        conn = await create_connection()
        try:
            row = await conn.fetchrow(
                f"""
                SELECT
                    symbol,
                    regime,
                    bars_in_state,
                    cooldown_remaining,
                    volatility_cooldown_remaining,
                    pending_regime,
                    pending_count,
                    last_rebuild_price,
                    kill_switch_count,
                    recent_equity
                FROM {STATE_SCHEMA}.{STATE_TABLE}
                WHERE symbol = $1
                """,
                symbol.upper(),
            )
        finally:
            await conn.close()

        if row is None:
            return None

        recent_equity = row["recent_equity"] or []
        if isinstance(recent_equity, str):
            recent_equity = json.loads(recent_equity)

        pending_regime = row["pending_regime"]
        return SymbolRuntimeState(
            symbol=str(row["symbol"]).upper(),
            strategy_state=StrategyState(
                regime=RegimeType(str(row["regime"])),
                bars_in_state=int(row["bars_in_state"]),
                cooldown_remaining=int(row["cooldown_remaining"]),
                volatility_cooldown_remaining=int(row["volatility_cooldown_remaining"]),
                pending_regime=RegimeType(str(pending_regime)) if pending_regime else None,
                pending_count=int(row["pending_count"]),
                last_rebuild_price=float(row["last_rebuild_price"]) if row["last_rebuild_price"] is not None else None,
            ),
            risk_state=RiskRuntimeState(
                kill_switch_count=int(row["kill_switch_count"]),
                recent_equity=[float(value) for value in recent_equity],
            ),
        )

    async def save_symbol_state(self, state: SymbolRuntimeState) -> None:
        """Insert or update the persisted runtime snapshot for one symbol."""
        conn = await create_connection()
        try:
            await conn.execute(
                f"""
                INSERT INTO {STATE_SCHEMA}.{STATE_TABLE} (
                    symbol,
                    regime,
                    bars_in_state,
                    cooldown_remaining,
                    volatility_cooldown_remaining,
                    pending_regime,
                    pending_count,
                    last_rebuild_price,
                    kill_switch_count,
                    recent_equity,
                    updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, NOW())
                ON CONFLICT (symbol) DO UPDATE
                SET
                    regime = EXCLUDED.regime,
                    bars_in_state = EXCLUDED.bars_in_state,
                    cooldown_remaining = EXCLUDED.cooldown_remaining,
                    volatility_cooldown_remaining = EXCLUDED.volatility_cooldown_remaining,
                    pending_regime = EXCLUDED.pending_regime,
                    pending_count = EXCLUDED.pending_count,
                    last_rebuild_price = EXCLUDED.last_rebuild_price,
                    kill_switch_count = EXCLUDED.kill_switch_count,
                    recent_equity = EXCLUDED.recent_equity,
                    updated_at = NOW()
                """,
                state.symbol.upper(),
                state.strategy_state.regime.value,
                state.strategy_state.bars_in_state,
                state.strategy_state.cooldown_remaining,
                state.strategy_state.volatility_cooldown_remaining,
                state.strategy_state.pending_regime.value if state.strategy_state.pending_regime else None,
                state.strategy_state.pending_count,
                state.strategy_state.last_rebuild_price,
                state.risk_state.kill_switch_count,
                json.dumps(state.risk_state.recent_equity),
            )
        finally:
            await conn.close()
