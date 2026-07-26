# Trade Execution Service

`trade-execution-service` is the exchange-agnostic execution boundary for Pampilo trading.

It consumes persisted bot signals, validates execution permissions and live risk, routes
execution through venue adapters, records order/fill/position state, and owns user/account
cost basis such as average entry price.

Bot Platform stays signal-only. This service is the only place that may call private
exchange execution APIs.

## Responsibilities

- Consume `bot_signal.persisted.v1` events.
- Treat `bot_signal.persisted.v1` as metadata only.
- Read full signal payloads from `_bot_platform.bot_signals.payload_json` by `signal_id`.
- Validate tenant/user/account/bot permissions.
- Resolve exchange account secret references.
- Read live balances, positions, open orders, fills, and venue constraints.
- Compute final order quantity and notional.
- Place, cancel, replace, and reconcile orders through exchange adapters.
- Track fills, positions, average entry price, and execution audit.
- Enforce global kill switch and per-account, per-bot, and per-symbol pauses.

## Multi-Exchange Design

The service is designed to support multiple exchanges. Domain and application code depend
on `ExchangeExecutionAdapter` protocols, not concrete SDKs.

Initial placeholder adapters exist for:

- Binance
- Bybit
- OKX

Concrete adapters should live under:

```text
src/trade_execution_service/infrastructure/exchanges/
```

Each adapter must normalize venue-specific behavior into the service domain contracts.
Venue SDKs and private credentials must not leak into domain code.

## Initial CLI

```bash
python -m trade_execution_service.main
python -m trade_execution_service.main healthcheck
python -m trade_execution_service.main worker
```

The initial `worker` command is a scaffold and does not place orders yet.

## Signal Consumer Rollout

The execution worker must subscribe only to the stable `bot_signal.persisted.v1`
stream. The event is metadata-only; the worker must load the canonical full payload from
`_bot_platform.bot_signals.payload_json` by `signal_id` before making any execution
decision.

Runtime controls are enforced inside execution service:

- global execution kill switch blocks every signal before exchange calls;
- per-bot and per-symbol pauses block only matching signals;
- idempotency is keyed by `signal_id` plus the persisted `signal_key` when available.

Rollback during rollout is operational: disable the platform bot instance that emits
signals or stop/disable the execution consumer. The legacy bot remains the reference path
until execution rollout is complete.
