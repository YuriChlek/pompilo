#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="${PROJECT_NAME:-pampilo-platform}"
ENV_FILE="${ENV_FILE:-control-panel/.env}"
COMPOSE_FILES=(
  -f infra/compose/docker-compose.yaml
  -f infra/compose/docker-compose.dev.yaml
)
MARKET_DATA_URL="${MARKET_DATA_URL:-http://localhost:${MARKET_DATA_HTTP_HOST_PORT:-8010}}"
DB_USER_VALUE="${DB_USER:-admin}"
DB_PASSWORD_VALUE="${DB_PASSWORD:-admin_pass}"
DB_NAME_VALUE="${DB_NAME:-pampilo_db}"
OUTBOX_STREAM="${MARKET_DATA_OUTBOX_STREAM:-market-data-events}"
SMOKE_WAIT_SECONDS="${SMOKE_WAIT_SECONDS:-90}"
SMOKE_SYNC_ITERATIONS="${SMOKE_SYNC_ITERATIONS:-1}"

export MARKET_DATA_PROVIDER_MODE="${MARKET_DATA_PROVIDER_MODE:-fixture}"
export MARKET_DATA_PROVIDER_SYMBOLS="${MARKET_DATA_PROVIDER_SYMBOLS:-ETHUSDT}"
export MARKET_DATA_TIMEFRAMES="${MARKET_DATA_TIMEFRAMES:-1h}"
export MARKET_DATA_SCHEDULER_JITTER_SECONDS="${MARKET_DATA_SCHEDULER_JITTER_SECONDS:-0}"
export MARKET_DATA_1H_SAFETY_DELAY_SECONDS="${MARKET_DATA_1H_SAFETY_DELAY_SECONDS:-0}"

compose() {
  docker compose -p "$PROJECT_NAME" --env-file "$ENV_FILE" "${COMPOSE_FILES[@]}" \
    --profile infra --profile platform "$@"
}

wait_for_http() {
  local url="$1"
  local deadline=$((SECONDS + SMOKE_WAIT_SECONDS))
  until curl -fsS "$url" >/dev/null; do
    if (( SECONDS >= deadline )); then
      echo "Smoke failed: $url did not become ready within ${SMOKE_WAIT_SECONDS}s" >&2
      compose logs --tail=200 market_data >&2 || true
      return 1
    fi
    sleep 2
  done
}

psql_scalar() {
  local sql="$1"
  compose exec -T -e PGPASSWORD="$DB_PASSWORD_VALUE" postgres \
    psql -U "$DB_USER_VALUE" -d "$DB_NAME_VALUE" -At -c "$sql"
}

redis_scalar() {
  compose exec -T redis redis-cli "$@"
}

require_positive_count() {
  local label="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value <= 0 )); then
    echo "Smoke failed: expected ${label} to be > 0, got '${value}'." >&2
    echo "This usually means the runtime did not complete a provider sync into candles/snapshots/outbox." >&2
    compose logs --tail=200 market_data >&2 || true
    return 1
  fi
}

main() {
  compose stop market_data || true
  compose up --build -d postgres redis
  compose up --build market_data_migrate
  compose run --rm market_data_symbols_sync
  compose run --rm market_data python -m market_data_service.main scheduler:run-once
  for _ in $(seq 1 "$SMOKE_SYNC_ITERATIONS"); do
    compose run --rm market_data python -m market_data_service.main sync:run-next
  done
  compose run --rm market_data python -m market_data_service.main outbox:publish-once
  compose up --build -d market_data

  wait_for_http "${MARKET_DATA_URL}/health/ready"
  curl -fsS "${MARKET_DATA_URL}/metrics" >/dev/null

  local sync_jobs_count
  local batches_count
  local candles_count
  local snapshots_count
  local outbox_count
  local redis_stream_count

  sync_jobs_count="$(psql_scalar "select count(*) from _market_data.sync_jobs;")"
  batches_count="$(psql_scalar "select count(*) from _market_data.market_data_batches;")"
  candles_count="$(psql_scalar "select count(*) from _market_data.market_candles;")"
  snapshots_count="$(psql_scalar "select count(*) from _market_data.market_snapshots;")"
  outbox_count="$(psql_scalar "select count(*) from _market_data.outbox_events;")"
  redis_stream_count="$(redis_scalar XLEN "$OUTBOX_STREAM")"

  require_positive_count "sync_jobs" "$sync_jobs_count"
  require_positive_count "market_data_batches" "$batches_count"
  require_positive_count "market_candles" "$candles_count"
  require_positive_count "market_snapshots" "$snapshots_count"
  require_positive_count "outbox_events" "$outbox_count"
  require_positive_count "Redis stream ${OUTBOX_STREAM}" "$redis_stream_count"

  echo "Market Data runtime smoke passed."
}

main "$@"
