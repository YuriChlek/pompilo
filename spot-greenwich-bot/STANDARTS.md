# Pompilo Bot Standards

## Purpose

This document defines the required naming, structural, and architectural conventions for all trading bot projects in the Pompilo suite.

The goal is consistency:

- modules look the same across projects
- boundaries between layers are explicit and enforced
- domain logic is portable and testable without infrastructure
- configuration lives in predictable, layered places
- integration code is isolated from business rules

---

## Naming Rules

### General

- Use `snake_case` for folders and file names.
- Use `PascalCase` for classes, dataclasses, protocols, and exceptions.
- Use `snake_case` for functions, methods, and variables.
- Use `SCREAMING_SNAKE_CASE` for module-level constants and sentinel values.

### Private vs. public

- Public module-level functions: no underscore prefix.
- Private module-level helpers: single underscore prefix (`_build_something`).
- Do not use double underscores (`__`) outside of Python dunder methods.

### Required file names

These names are canonical. If a file serves one of these roles, it must use this name:

| Role | File Name |
|---|---|
| Domain value objects and aggregates | `models.py` |
| Strategy configuration dataclasses | `strategy_config.py` |
| Signal generation pipeline | `signal_generation.py` |
| Shared signal utilities | `signal_common.py` |
| Public signal API (barrel) | `signals.py` |
| Domain execution rules | `execution.py` |
| Application port protocols | `ports.py` |
| Application orchestration services | `services.py` |
| Dependency injection wiring | `bootstrap.py` |
| Scheduled trading loop | `scheduler.py` |
| Async entry points | `runner.py` |
| Indicator calculation classes | `analyzers.py` |
| Indicator pipeline orchestrator | `trend_bot.py` |
| Indicator public API | `api.py` |
| Candle DB loader | `data_fetcher.py` |
| Exchange adapter | Named for the exchange, e.g. `bybit.py` |
| Execution adapter | `execution_service.py` |
| Market data adapter | `market_data.py` |
| Notification adapter | `notifications.py` |
| Data sync adapter | `data_collection.py` |
| Environment loading | `env.py` |
| Runtime configuration | `config.py` |
| Database helpers | `db_actions.py` |
| Table schema setup | `create_tables.py` |

### Function naming

Adopt the following naming prefixes for clarity:

| Prefix | Use |
|---|---|
| `build_*` | Constructs and returns a domain object or config |
| `_build_*` | Private builder helper |
| `calculate_*` | Pure numeric computation |
| `detect_*` | Classification / regime detection |
| `resolve_*` | Picks the right value from multiple options |
| `normalize_*` | Coerces a value to exchange/domain format |
| `evaluate_*` | Returns a decision object |
| `sync_*` | Writes exchange state to persistence |
| `fetch_*` | Reads from external source (DB, API) |
| `can_*` | Returns bool (capability or pre-condition check) |
| `_is_*` | Private bool predicate |
| `notify_*` | Sends a notification |
| `run_*` | Top-level async entrypoint |

### Naming consistency

The same concept must have the same name everywhere in the project:

- folder names
- file names
- class names
- variable names
- DB column names
- log field names

No mixed spellings. If a term is chosen, use it everywhere. Examples of what to avoid:

- `analyse` vs. `analyze`
- `canceled` vs. `cancelled`
- `tp` vs. `take_profit` in the same context
- `symbol` vs. `ticker` for the same concept

---

## Architecture

All bots follow **Clean Architecture** with three horizontal layers. Dependencies only flow inward.

```
┌─────────────────────────────────────────────────┐
│                  Infrastructure                  │  ← exchange, DB, Telegram, API clients
├─────────────────────────────────────────────────┤
│                   Application                    │  ← orchestration, ports, scheduling, DI
├─────────────────────────────────────────────────┤
│                     Domain                       │  ← models, signals, execution rules, config
└─────────────────────────────────────────────────┘
```

### Layer rules

**Domain** (`trading/domain/`)

- Contains all trading logic: models, signal generation, execution rules, strategy config.
- Has **no imports** from application, infrastructure, utils I/O, or third-party networking libraries.
- Pure functions and frozen dataclasses only.
- Numerical values use `Decimal`, never `float`.

**Application** (`trading/application/`)

- Contains ports (interfaces), service orchestration, scheduler, bootstrap wiring.
- Imports from domain.
- Defines ports as `Protocol` classes — infrastructure implements them.
- Does **not** import exchange clients, DB drivers, or HTTP libraries directly.
- Does **not** contain business rules.

**Infrastructure** (`trading/infrastructure/`)

- Contains exchange adapters, notification adapters, data sync adapters.
- Imports from domain and implements application ports.
- Owns all vendor-specific logic: signing, retries, normalization to exchange format, pagination.
- Does **not** contain business decisions.

### What goes where

**Domain models** (`domain/models.py`)

- Frozen dataclasses for all signal and position data.
- `Mapping[str, Any]` compatibility only when the model must serialize to a dict.
- No methods with side effects.
- No external imports.

**Strategy config** (`domain/strategy_config.py`)

- One frozen dataclass per concern group (risk, exit, regime, breakout, portfolio, etc.).
- Conservative code-level defaults in `DEFAULT_STRATEGY_CONFIG`.
- All numeric config values use `Decimal`.

**Signal generation** (`domain/signal_generation.py`)

- All entry decision logic lives here.
- Helper builders prefixed `_build_*`.
- Must remain testable without any I/O.

**Execution rules** (`domain/execution.py`)

- Portfolio admission, stop management, partial close logic.
- Pure functions; returns domain decision objects.
- No exchange API calls.

**Ports** (`application/ports.py`)

- `Protocol` class per concern: `MarketDataProvider`, `PositionExecutor`, `SignalNotifier`, `MarketDataSynchronizer`.
- Only type signatures — no implementation.
- Docstring on every method explaining the contract.

**Services** (`application/services.py`)

- Orchestrates one trading cycle: data fetch → signal → management → execution → notification.
- Receives dependencies via constructor injection (no global imports of adapters).
- Does not contain business rules.

**Bootstrap** (`application/bootstrap.py`)

- Single entry point for all dependency injection.
- Reads `utils/config.py` and overlays env onto domain defaults.
- Returns fully wired objects, not partially constructed ones.

**Exchange adapter** (`infrastructure/bybit.py` etc.)

- All vendor-specific logic: signing, retries, status normalization.
- Normalizes prices and quantities to exchange constraints before returning.
- Does not make business decisions about position sizing or signal validity.

**Execution adapter** (`infrastructure/execution_service.py`)

- Implements `PositionExecutor` port.
- Calls domain execution rules, then calls exchange adapter.
- Syncs state to DB after every exchange interaction.

---

## Architectural Boundaries

### Import rules

| From | May import | May not import |
|---|---|---|
| `domain/` | Standard library, `decimal` | `infrastructure/`, `application/`, `utils/db_actions.py`, third-party HTTP/DB |
| `application/` | `domain/`, standard library | `infrastructure/` (except in `bootstrap.py`) |
| `infrastructure/` | `domain/`, `application/ports`, `utils/` | Other infrastructure modules directly (pass through ports) |
| `utils/` | Standard library, third-party | `trading/`, `indicators/` |
| `indicators/` | `utils/`, standard library, numpy/pandas | `trading/` |
| `tests/` | Everything (for test setup), domain and application for assertions | Should not call real exchange APIs |

In `bootstrap.py` only: infrastructure imports are permitted because this is the composition root.

### Circular imports

Circular imports are forbidden. If two modules need each other, extract the shared concept into a third module.

### Direction normalization

Direction strings exist at multiple layers:

- Domain uses lowercase strings: `"buy"`, `"sell"`.
- Bybit API uses title case: `"Buy"`, `"Sell"`.
- Normalization functions that convert between these belong in the infrastructure adapter only.
- Domain functions must not call infrastructure normalizers.

---

## Domain Model Rules

- All domain models are **frozen dataclasses** (`@dataclass(frozen=True)`).
- Prices, sizes, distances, and percentages are **always `Decimal`** — no `float`.
- Timestamps are `int` (milliseconds) in infrastructure, `pd.Timestamp` or ISO string at the domain boundary.
- `metadata` dicts in signals are `dict[str, Any]`. When metadata grows beyond 3–4 keys with shared structure, promote it to a typed dataclass.
- Models that must serialize to dicts implement `Mapping[str, Any]` via `__iter__` and `__len__`.
- Domain models do not contain logging, retries, or network calls.

---

## Configuration Hierarchy

Configuration is assembled in three stages. Each stage narrows the previous:

```
1. Code defaults (domain/strategy_config.py)
        ↓
2. Environment overrides (utils/config.py reads .env)
        ↓
3. Runtime assembly (application/bootstrap.py wires into domain config)
```

### Rules

- Code defaults live in `DEFAULT_STRATEGY_CONFIG` or equivalent constant in `domain/strategy_config.py`.
- Environment parsing helpers (`_env_flag`, `_env_decimal`, `_env_int`) belong in `utils/config.py` only.
- `bootstrap.py` is the only place that reads from both `utils/config` and domain defaults and merges them.
- No module outside `utils/config.py` should call `os.environ` directly.
- No module outside `bootstrap.py` should merge env values into domain config.
- Hardcoded values that might change per deployment (symbols list, rounding rules) belong in `utils/config.py` or env, not in domain files.

### Environment variable naming

- Boolean flags: `ENABLE_*` (e.g., `ENABLE_BREAKEVEN_STOP_MANAGEMENT`).
- Numeric thresholds: descriptive uppercase (e.g., `BREAKEVEN_TRIGGER_R`, `MAX_PORTFOLIO_HEAT_PCT`).
- Credentials: `EXCHANGE_API_KEY`, `EXCHANGE_API_SECRET`, `EXCHANGE_API_ENDPOINT`.
- DB connection: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASS`.

---

## Database Access Patterns

- Use `asyncpg` directly — no ORM.
- All DB access helpers live in `utils/db_actions.py`.
- Table creation scripts live in `utils/create_tables.py`.
- Use two schemas per bot:
  - `_candles_trading_data` — time-series candle tables, one table per symbol.
  - `_live_trading_state` — position snapshots, order snapshots, reconciliation runs.
- Table names for candles: `{schema}.{symbol_lower}_1h`.
- Use `asyncpg.Pool` for connection pooling.
- Use parameterized queries for all user-facing or external-data values.
- Schema and table names are template-filled (not parameterized), and must only come from trusted config, not user input.
- Upserts must be idempotent: use `ON CONFLICT DO UPDATE` with a stable natural key.
- After every exchange state fetch, sync positions and orders to DB immediately.
- `JSONB raw_payload` column stores the full exchange response alongside normalized columns.

### Sync pattern

```python
positions = await exchange.get_open_positions(symbol)
await sync_live_positions(symbol, positions, pool)
```

Never read positions from DB to make trading decisions — always fetch live from exchange, then sync.

---

## Error Handling

- Do not swallow unexpected errors silently.
- Catch only the exceptions you can meaningfully handle at that boundary.
- Wrap infrastructure errors at the adapter boundary with a module-specific exception type before they reach the application layer.
- Re-raise domain errors unchanged through the application layer.
- Log `exc_info=True` (or `logger.exception(...)`) before re-raising unexpected errors.
- Use `tenacity` for exchange API retries inside the infrastructure adapter — do not implement manual retry loops.
- Retry only on recoverable errors (rate limits, transient network failures). Do not retry on authentication failures or invalid parameter errors.

---

## Logging

- Use the standard `logging` module. Logger is `logging.getLogger(__name__)` in every module.
- Log messages follow the pattern: `"event_name key=value key=value"`.
- Always include `symbol` as a log field when logging in a symbol-specific context.
- Log levels:
  - `DEBUG` — per-candle detail, raw indicator values.
  - `INFO` — signal generated, position opened, state synced.
  - `WARNING` — admission rejected, stale order found, partial data.
  - `ERROR` — unexpected exception caught, exchange error.
- Do not log secrets, API keys, or full order payloads at INFO or above.
- Use `logger.exception(...)` (not `logger.error(...)`) when logging inside an `except` block that captures a traceback.

---

## Testing

### Approach

- Unit tests cover pure domain logic and application service orchestration.
- Infrastructure adapters are tested by mocking the exchange HTTP layer, not the adapter itself.
- Tests do not call real exchange APIs or real databases.
- Test files are named `test_{module_name}.py` and mirror the production module under test.

### Test infrastructure

- Stubs and fakes for external dependencies live in `tests/support.py`.
- `install_common_test_stubs()` installs all stubs needed for domain-level tests.
- Fake market data builders (`_make_trend_data`, `_make_range`, etc.) live in `support.py` and accept override kwargs.
- Use `unittest.TestCase` as the base class.
- Use `AsyncMock` and `unittest.mock.patch` for async I/O dependencies.
- Use `SimpleNamespace` or dataclass stubs for fake indicator/market data.
- Call `asyncio.run(...)` directly for async tests — do not add pytest-asyncio unless the project already uses pytest.

### What to test

- All public domain functions.
- All regime detection and signal generation paths (including rejections).
- Portfolio admission conditions: each limit enforced separately.
- Stop management: breakeven trigger, trailing, regime exit.
- Execution adapter behavior: quantity resolution, normalization, admission gate.
- State sync: positions and orders written correctly after exchange fetch.
- Config parsing: env flags and decimals parse correctly with defaults.

### What not to test

- Private helper functions directly — test them through the public API.
- Exact log output format.
- Third-party library internals.

---

## Indicators Subsystem

- Indicator calculation classes live in `indicators/analyzers.py`, one class per indicator family (e.g., `GMMAAnalyzer`, `FractalAnalyzer`, `SuperTrendAnalyzer`).
- `indicators/trend_bot.py` is the orchestrator: loads candles from DB, runs all analyzers, assembles the result object.
- `indicators/api.py` is the public entry point called by the application layer.
- `indicators/data_fetcher.py` handles all PostgreSQL reads for candle data.
- `indicators/range.py` is isolated for range-market detection logic.
- The result type (e.g., `TrendResult`) is a dataclass that bundles multi-timeframe indicator snapshots.
- Indicator classes must not call the exchange API or write to the DB.

---

## Backtesting Subsystem

- Backtesting lives in `backtesting/` and is a parallel system to live trading.
- Backtesting must use the same domain signal logic as live trading (same `signal_generation.py`).
- Exchange interactions are replaced with in-memory simulation adapters in `backtesting/adapters.py`.
- Backtest results are written to files or DB, not to the live state schema.
- Do not modify domain logic to accommodate backtesting behavior — adapt the backtesting adapters instead.

---

## Review Checklist

Before merging any trading bot code, verify:

**Naming**
- [ ] File names match the canonical names in this document
- [ ] Functions use the correct verb prefix for their role
- [ ] No mixed spellings of domain terms
- [ ] Constants are SCREAMING_SNAKE_CASE
- [ ] No `float` used for prices, sizes, or percentages

**Architecture**
- [ ] Domain has no infrastructure imports
- [ ] Application has no infrastructure imports (except `bootstrap.py`)
- [ ] Ports are defined in `ports.py` as `Protocol` classes
- [ ] Bootstrap is the only place that wires adapters to ports
- [ ] Business decisions do not live in infrastructure adapters

**Configuration**
- [ ] New env vars are parsed in `utils/config.py` only
- [ ] New knobs have a code-level default in `strategy_config.py`
- [ ] Bootstrap merges env into domain config
- [ ] No `os.environ` calls outside `utils/config.py`

**Database**
- [ ] All DB access goes through `utils/db_actions.py`
- [ ] Upserts use `ON CONFLICT DO UPDATE`
- [ ] Exchange state is synced after every interaction
- [ ] No ORM introduced

**Error handling**
- [ ] No silent `except Exception: pass` blocks
- [ ] Retries use `tenacity` in infrastructure, not manual loops
- [ ] Infrastructure exceptions are wrapped before crossing into application layer

**Tests**
- [ ] New domain logic has unit tests
- [ ] New service flows have integration-style tests with mocked I/O
- [ ] Fakes and stubs are in `tests/support.py`
- [ ] Tests do not call real exchange APIs or DB
