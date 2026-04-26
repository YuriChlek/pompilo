# Backlog Status

## Completed Phases

### Phase 1. Prepare Target Architecture Skeleton

Status: `done`

Implemented:

- top-level `application/`, `domain/`, `infrastructure/`
- initial module layout for the new architecture

### Phase 2. Define Stable Domain Models

Status: `done`

Implemented:

- canonical models moved to [domain/models.py](/home/yurii/Proj/pompilo/spot-trading-bot/domain/models.py:1)

### Phase 3. Introduce Application Ports

Status: `done`

Implemented:

- canonical ports in [application/ports.py](/home/yurii/Proj/pompilo/spot-trading-bot/application/ports.py:1)

### Phase 4. Extract Greenwich Strategy Core

Status: `done`

Implemented:

- signal logic in [domain/signals.py](/home/yurii/Proj/pompilo/spot-trading-bot/domain/signals.py:1)
- execution policy in [domain/execution.py](/home/yurii/Proj/pompilo/spot-trading-bot/domain/execution.py:1)
- planner in [domain/planner.py](/home/yurii/Proj/pompilo/spot-trading-bot/domain/planner.py:1)

### Phase 5. Build Infrastructure Adapters in New Layout

Status: `done`

Implemented:

- PostgreSQL market data provider
- Binance synchronizer
- Binance spot execution adapter
- logging notifier

### Phase 6. Add Initialization Service

Status: `done`

Implemented:

- centralized startup service in [application/initialization_service.py](/home/yurii/Proj/pompilo/spot-trading-bot/application/initialization_service.py:1)

### Phase 7. Implement New Trading Cycle Service

Status: `done`

Implemented:

- canonical trading cycle orchestration in [application/trading_cycle_service.py](/home/yurii/Proj/pompilo/spot-trading-bot/application/trading_cycle_service.py:1)

### Phase 8. Implement New Execution Service Layer

Status: `done`

Implemented:

- application-level execution side effects in [application/execution_service.py](/home/yurii/Proj/pompilo/spot-trading-bot/application/execution_service.py:1)

### Phase 9. Rebuild Bootstrap Composition

Status: `done`

Implemented:

- canonical composition root in [application/bootstrap.py](/home/yurii/Proj/pompilo/spot-trading-bot/application/bootstrap.py:1)

### Phase 10. Rebuild Scheduler

Status: `done`

Implemented:

- canonical scheduler in [application/scheduler.py](/home/yurii/Proj/pompilo/spot-trading-bot/application/scheduler.py:1)

### Phase 11. Rebuild CLI Entrypoint

Status: `done`

Implemented:

- thin CLI entrypoint in [main.py](/home/yurii/Proj/pompilo/spot-trading-bot/main.py:1)
- command routing through [application/command_dispatcher.py](/home/yurii/Proj/pompilo/spot-trading-bot/application/command_dispatcher.py:1)
- concrete handlers in [application/runtime_commands.py](/home/yurii/Proj/pompilo/spot-trading-bot/application/runtime_commands.py:1)

### Phase 12. Add Backward-Compatibility Layer

Status: `done during migration`, `removed after Phase 15`

Implemented and later removed:

- temporary compatibility wrappers under `trading/*`

### Phase 13. Expand Test Coverage for Behavioral Equivalence

Status: `done`

Implemented:

- unit tests for command routing
- runtime command routing
- initialization
- trading cycle
- execution service
- scheduler
- signal generation
- existing execution and infrastructure helper behavior

### Phase 14. Switch Default Runtime to New Path

Status: `done`

Implemented:

- default runtime routed entirely through `application/*`

### Phase 15. Remove Legacy Structure

Status: `done`

Implemented:

- removed legacy `trading/*` runtime layer
- updated tests to canonical imports
- moved `binance_spot` into canonical `infrastructure/*`

## Current Remaining Items

These are not unfinished migration phases. They are follow-up tasks.

### Documentation polish

Status: `done for core docs`

Included:

- updated implementation status docs
- new English and Ukrainian readmes

### Full dependency-backed test run

Status: `pending environment`

Blocked by:

- missing `pandas`
- missing `asyncpg`

### Optional cleanup

Status: `optional`

Possible follow-ups:

- remove stale `__pycache__` files
- decide whether to implement or delete `infrastructure/state_store.py`
- document environment setup in more operational detail if needed
