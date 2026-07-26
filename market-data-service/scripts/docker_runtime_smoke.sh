#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="${PROJECT_NAME:-pampilo-platform}"
ENV_FILE="${ENV_FILE:-control-panel/.env}"
COMPOSE_FILES=(
  -f infra/compose/docker-compose.yaml
)
DB_USER_VALUE="${DB_USER:-admin}"
DB_PASSWORD_VALUE="${DB_PASSWORD:-admin_pass}"
DB_NAME_VALUE="${DB_NAME:-pampilo_db}"
SMOKE_WAIT_SECONDS="${SMOKE_WAIT_SECONDS:-90}"
SMOKE_SYMBOL="${SMOKE_SYMBOL:-ETHUSDT}"
SMOKE_TIMEFRAME="${SMOKE_TIMEFRAME:-1h}"
SMOKE_TRAILING_CANDLES="${SMOKE_TRAILING_CANDLES:-2}"

export MARKET_DATA_PROVIDER_MODE="${MARKET_DATA_PROVIDER_MODE:-fixture}"
export MARKET_DATA_PROVIDER_SYMBOLS="${MARKET_DATA_PROVIDER_SYMBOLS:-$SMOKE_SYMBOL}"
export MARKET_DATA_TIMEFRAMES="${MARKET_DATA_TIMEFRAMES:-$SMOKE_TIMEFRAME}"
export MARKET_DATA_SCHEDULER_JITTER_SECONDS="${MARKET_DATA_SCHEDULER_JITTER_SECONDS:-0}"
export MARKET_DATA_1H_SAFETY_DELAY_SECONDS="${MARKET_DATA_1H_SAFETY_DELAY_SECONDS:-0}"
export MARKET_DATA_OUTBOX_STREAM="${MARKET_DATA_OUTBOX_STREAM:-market-data-events-smoke}"
export MARKET_DATA_COLLECT_BOOTSTRAP_LOOKBACK_YEARS="${MARKET_DATA_COLLECT_BOOTSTRAP_LOOKBACK_YEARS:-1}"
export MARKET_DATA_COLLECT_BOOTSTRAP_MAX_CHUNKS_PER_TICK="${MARKET_DATA_COLLECT_BOOTSTRAP_MAX_CHUNKS_PER_TICK:-20}"
export MARKET_DATA_COLLECT_MAX_JOBS_PER_TICK="${MARKET_DATA_COLLECT_MAX_JOBS_PER_TICK:-50}"
export MARKET_DATA_COLLECT_POLL_INTERVAL_SECONDS="${MARKET_DATA_COLLECT_POLL_INTERVAL_SECONDS:-2}"
export MARKET_DATA_COLLECT_ON_START="${MARKET_DATA_COLLECT_ON_START:-true}"
export MARKET_DATA_OUTBOX_RETENTION_DAYS="${MARKET_DATA_OUTBOX_RETENTION_DAYS:-5}"

OUTBOX_STREAM="$MARKET_DATA_OUTBOX_STREAM"

if [[ "${SMOKE_INCLUDE_DEV_COMPOSE:-false}" == "true" ]]; then
  COMPOSE_FILES+=(-f infra/compose/docker-compose.dev.yaml)
fi

compose() {
  docker compose -p "$PROJECT_NAME" --env-file "$ENV_FILE" "${COMPOSE_FILES[@]}" \
    --profile infra --profile platform "$@"
}

wait_for_market_data_health() {
  local deadline=$((SECONDS + SMOKE_WAIT_SECONDS))
  until compose exec -T market_data python -m market_data_service.main healthcheck >/dev/null; do
    if (( SECONDS >= deadline )); then
      echo "Smoke failed: market_data did not become ready within ${SMOKE_WAIT_SECONDS}s" >&2
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

psql_exec() {
  local sql="$1"
  compose exec -T -e PGPASSWORD="$DB_PASSWORD_VALUE" postgres \
    psql -v ON_ERROR_STOP=1 -U "$DB_USER_VALUE" -d "$DB_NAME_VALUE" -c "$sql" >/dev/null
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

require_zero_count() {
  local label="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value != 0 )); then
    echo "Smoke failed: expected ${label} to be 0, got '${value}'." >&2
    compose logs --tail=200 market_data >&2 || true
    return 1
  fi
}

require_count_greater_than() {
  local label="$1"
  local value="$2"
  local minimum="$3"
  if [[ ! "$value" =~ ^[0-9]+$ ]] || [[ ! "$minimum" =~ ^[0-9]+$ ]] || (( value <= minimum )); then
    echo "Smoke failed: expected ${label} to be > ${minimum}, got '${value}'." >&2
    compose logs --tail=200 market_data >&2 || true
    return 1
  fi
}

require_count_unchanged() {
  local label="$1"
  local before="$2"
  local after="$3"
  if [[ "$before" != "$after" ]]; then
    echo "Smoke failed: expected ${label} to stay ${before}, got '${after}'." >&2
    compose logs --tail=200 market_data >&2 || true
    return 1
  fi
}

reset_market_data_smoke_state() {
  redis_scalar DEL "$OUTBOX_STREAM" >/dev/null
  psql_exec "
    truncate table
      _market_data.outbox_events,
      _market_data.sync_jobs,
      _market_data.market_snapshot_candles,
      _market_data.market_snapshots,
      _market_data.market_data_batches,
      _market_data.market_candles,
      _market_data.collection_states,
      _market_data.provider_symbol_availability,
      _market_data.provider_symbols,
      _market_data.market_symbols
    cascade;
  "
}

prepare_incremental_gap() {
  psql_scalar "
    with gap_candles as (
      select open_time
      from _market_data.market_candles
      where source = 'BINANCE_SPOT'
        and canonical_symbol = '$SMOKE_SYMBOL'
        and timeframe = '$SMOKE_TIMEFRAME'
      order by open_time desc
      limit $SMOKE_TRAILING_CANDLES
    ),
    bounds as (
      select min(open_time) as incremental_from
      from gap_candles
    ),
    deleted as (
      delete from _market_data.market_candles candle
      using gap_candles
      where candle.source = 'BINANCE_SPOT'
        and candle.canonical_symbol = '$SMOKE_SYMBOL'
        and candle.timeframe = '$SMOKE_TIMEFRAME'
        and candle.open_time = gap_candles.open_time
      returning 1
    ),
    updated_state as (
      update _market_data.collection_states
      set bootstrap_to = bounds.incremental_from,
          bootstrap_next_from = bounds.incremental_from,
          last_successful_close_time = bounds.incremental_from,
          updated_at = now()
      from bounds
      where source = 'BINANCE_SPOT'
        and canonical_symbol = '$SMOKE_SYMBOL'
        and timeframe = '$SMOKE_TIMEFRAME'
        and bounds.incremental_from is not null
      returning 1
    )
    select count(*) from deleted;
  "
}

insert_old_terminal_outbox_record() {
  psql_exec "
    insert into _market_data.outbox_events (
      id,
      event_type,
      aggregate_type,
      aggregate_id,
      payload_json,
      idempotency_key,
      status,
      attempts,
      created_at,
      published_at
    )
    values (
      'stage39-old-published-retention-smoke',
      'Stage39RetentionSmoke',
      'market_snapshot',
      'stage39-retention-smoke',
      '{\"source\":\"stage39\"}'::jsonb,
      'stage39-old-published-retention-smoke',
      'PUBLISHED',
      0,
      now() - interval '6 days',
      now() - interval '6 days'
    )
    on conflict (id) do update
      set status = 'PUBLISHED',
          published_at = now() - interval '6 days',
          created_at = now() - interval '6 days';
  "
}

mark_collection_current_for_idempotent_rerun() {
  psql_exec "
    update _market_data.collection_states
    set last_successful_close_time = date_trunc('hour', now()),
        bootstrap_to = date_trunc('hour', now()),
        bootstrap_next_from = date_trunc('hour', now()),
        updated_at = now()
    where source = 'BINANCE_SPOT'
      and canonical_symbol = '$SMOKE_SYMBOL'
      and timeframe = '$SMOKE_TIMEFRAME';
  "
}

main() {
  if [[ "$MARKET_DATA_PROVIDER_MODE" != "fixture" ]]; then
    echo "Smoke failed: docker runtime smoke must run with MARKET_DATA_PROVIDER_MODE=fixture." >&2
    return 1
  fi

  compose stop market_data || true
  compose up --build -d postgres redis
  compose up --build market_data_migrate

  reset_market_data_smoke_state

  compose run --rm market_data python -m market_data_service.main collect --once

  local bootstrap_candles_count
  local bootstrap_snapshots_count
  local bootstrap_outbox_count
  local bootstrap_redis_stream_count

  bootstrap_candles_count="$(psql_scalar "select count(*) from _market_data.market_candles where source = 'BINANCE_SPOT' and canonical_symbol = '$SMOKE_SYMBOL' and timeframe = '$SMOKE_TIMEFRAME';")"
  bootstrap_snapshots_count="$(psql_scalar "select count(*) from _market_data.market_snapshots where source = 'BINANCE_SPOT' and canonical_symbol = '$SMOKE_SYMBOL' and timeframe = '$SMOKE_TIMEFRAME';")"
  bootstrap_outbox_count="$(psql_scalar "select count(*) from _market_data.outbox_events;")"
  bootstrap_redis_stream_count="$(redis_scalar XLEN "$OUTBOX_STREAM")"

  require_positive_count "bootstrap market_candles" "$bootstrap_candles_count"
  require_positive_count "bootstrap market_snapshots" "$bootstrap_snapshots_count"
  require_zero_count "bootstrap outbox_events" "$bootstrap_outbox_count"
  require_zero_count "bootstrap Redis stream ${OUTBOX_STREAM}" "$bootstrap_redis_stream_count"

  local deleted_trailing_candles_count
  deleted_trailing_candles_count="$(prepare_incremental_gap)"
  require_positive_count "prepared trailing candle gap" "$deleted_trailing_candles_count"

  compose run --rm market_data python -m market_data_service.main collect --once

  local incremental_candles_count
  local incremental_snapshots_count
  local incremental_published_outbox_count
  local incremental_redis_stream_count

  incremental_candles_count="$(psql_scalar "select count(*) from _market_data.market_candles where source = 'BINANCE_SPOT' and canonical_symbol = '$SMOKE_SYMBOL' and timeframe = '$SMOKE_TIMEFRAME';")"
  incremental_snapshots_count="$(psql_scalar "select count(*) from _market_data.market_snapshots where source = 'BINANCE_SPOT' and canonical_symbol = '$SMOKE_SYMBOL' and timeframe = '$SMOKE_TIMEFRAME';")"
  incremental_published_outbox_count="$(psql_scalar "select count(*) from _market_data.outbox_events where status = 'PUBLISHED';")"
  incremental_redis_stream_count="$(redis_scalar XLEN "$OUTBOX_STREAM")"

  require_count_greater_than "incremental market_candles" "$incremental_candles_count" "$((bootstrap_candles_count - deleted_trailing_candles_count))"
  require_count_greater_than "incremental market_snapshots" "$incremental_snapshots_count" "$bootstrap_snapshots_count"
  require_positive_count "incremental published outbox_events" "$incremental_published_outbox_count"
  require_count_greater_than "incremental Redis stream ${OUTBOX_STREAM}" "$incremental_redis_stream_count" "$bootstrap_redis_stream_count"

  local candles_before_cleanup_count
  local old_outbox_before_cleanup_count
  local old_outbox_after_cleanup_count
  local candles_after_cleanup_count

  candles_before_cleanup_count="$(psql_scalar "select count(*) from _market_data.market_candles;")"
  insert_old_terminal_outbox_record
  old_outbox_before_cleanup_count="$(psql_scalar "select count(*) from _market_data.outbox_events where id = 'stage39-old-published-retention-smoke';")"
  require_positive_count "old terminal outbox retention fixture" "$old_outbox_before_cleanup_count"

  compose run --rm market_data python -m market_data_service.main outbox:cleanup --batch-size 1000

  old_outbox_after_cleanup_count="$(psql_scalar "select count(*) from _market_data.outbox_events where id = 'stage39-old-published-retention-smoke';")"
  candles_after_cleanup_count="$(psql_scalar "select count(*) from _market_data.market_candles;")"
  require_zero_count "old terminal outbox retention fixture after cleanup" "$old_outbox_after_cleanup_count"
  require_count_unchanged "market_candles after outbox cleanup" "$candles_before_cleanup_count" "$candles_after_cleanup_count"

  local candles_before_idempotent_count
  local candles_after_idempotent_count
  local redis_before_idempotent_count
  local redis_after_idempotent_count
  local duplicate_natural_key_count

  mark_collection_current_for_idempotent_rerun
  candles_before_idempotent_count="$(psql_scalar "select count(*) from _market_data.market_candles;")"
  redis_before_idempotent_count="$(redis_scalar XLEN "$OUTBOX_STREAM")"
  compose run --rm market_data python -m market_data_service.main collect --once
  candles_after_idempotent_count="$(psql_scalar "select count(*) from _market_data.market_candles;")"
  redis_after_idempotent_count="$(redis_scalar XLEN "$OUTBOX_STREAM")"
  duplicate_natural_key_count="$(psql_scalar "select count(*) from (select source, canonical_symbol, timeframe, open_time from _market_data.market_candles group by source, canonical_symbol, timeframe, open_time having count(*) > 1) duplicates;")"

  require_count_unchanged "market_candles after idempotent rerun" "$candles_before_idempotent_count" "$candles_after_idempotent_count"
  require_count_unchanged "Redis stream ${OUTBOX_STREAM} after idempotent rerun" "$redis_before_idempotent_count" "$redis_after_idempotent_count"
  require_zero_count "duplicate candle natural keys" "$duplicate_natural_key_count"

  compose up --build -d market_data

  wait_for_market_data_health
  compose exec -T market_data python -c "import httpx; response = httpx.get('http://127.0.0.1:8010/metrics', timeout=5.0); response.raise_for_status()" >/dev/null

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
  require_positive_count "published outbox_events" "$outbox_count"
  require_positive_count "Redis stream ${OUTBOX_STREAM}" "$redis_stream_count"

  echo "Market Data full flow smoke passed."
}

main "$@"
