#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="${PROJECT_NAME:-pampilo-platform}"
ENV_FILE="${ENV_FILE:-control-panel/.env}"
COMPOSE_FILES=(
  -f infra/compose/docker-compose.yaml
)

API_URL="${API_URL:-http://127.0.0.1:3000}"
MARKET_DATA_URL="${MARKET_DATA_URL:-http://127.0.0.1:8010}"
SMOKE_CLIENT_ORIGIN="${SMOKE_CLIENT_ORIGIN:-${CLIENT_ORIGIN:-https://localhost}}"

DB_USER_VALUE="${DB_USER:-admin}"
DB_PASSWORD_VALUE="${DB_PASSWORD:-admin_pass}"
DB_NAME_VALUE="${DB_NAME:-pampilo_db}"

SMOKE_WAIT_SECONDS="${SMOKE_WAIT_SECONDS:-240}"
SMOKE_ADMIN_EMAIL="${SMOKE_ADMIN_EMAIL:-stage26-admin@example.com}"
SMOKE_ADMIN_PASSWORD="${SMOKE_ADMIN_PASSWORD:-Stage26Password123}"
SMOKE_INSTANCE_ID="${SMOKE_INSTANCE_ID:-stage26-spot-grid}"
SMOKE_SYMBOL="${SMOKE_SYMBOL:-ETHUSDT}"
SMOKE_CANONICAL_SYMBOL="${SMOKE_CANONICAL_SYMBOL:-$SMOKE_SYMBOL}"
SMOKE_TIMEFRAME="${SMOKE_TIMEFRAME:-1h}"
SMOKE_TRAILING_CANDLES="${SMOKE_TRAILING_CANDLES:-2}"
MARKET_DATA_EVENT_STREAM="${BOT_PLATFORM_MARKET_DATA_EVENTS_STREAM:-market-data-events-stage40}"
BOT_SIGNAL_STREAM="${BOT_PLATFORM_SIGNAL_EVENTS_STREAM:-bot-platform-signals-stage40}"

export MARKET_DATA_PROVIDER_SYMBOLS="${MARKET_DATA_PROVIDER_SYMBOLS:-$SMOKE_SYMBOL}"
export MARKET_DATA_TIMEFRAMES="${MARKET_DATA_TIMEFRAMES:-$SMOKE_TIMEFRAME}"
export MARKET_DATA_PROVIDER_MODE="${MARKET_DATA_PROVIDER_MODE:-fixture}"
export MARKET_DATA_OUTBOX_STREAM="${MARKET_DATA_OUTBOX_STREAM:-$MARKET_DATA_EVENT_STREAM}"
export MARKET_DATA_COLLECT_POLL_INTERVAL_SECONDS="${MARKET_DATA_COLLECT_POLL_INTERVAL_SECONDS:-60}"
export MARKET_DATA_COLLECT_ON_START="${MARKET_DATA_COLLECT_ON_START:-false}"
export MARKET_DATA_COLLECT_BOOTSTRAP_LOOKBACK_YEARS="${MARKET_DATA_COLLECT_BOOTSTRAP_LOOKBACK_YEARS:-1}"
export MARKET_DATA_COLLECT_BOOTSTRAP_MAX_CHUNKS_PER_TICK="${MARKET_DATA_COLLECT_BOOTSTRAP_MAX_CHUNKS_PER_TICK:-20}"
export MARKET_DATA_COLLECT_MAX_JOBS_PER_TICK="${MARKET_DATA_COLLECT_MAX_JOBS_PER_TICK:-50}"
export MARKET_DATA_SCHEDULER_JITTER_SECONDS="${MARKET_DATA_SCHEDULER_JITTER_SECONDS:-0}"
export MARKET_DATA_1H_SAFETY_DELAY_SECONDS="${MARKET_DATA_1H_SAFETY_DELAY_SECONDS:-0}"
export BOT_PLATFORM_SIGNAL_EVENTS_ENABLED="${BOT_PLATFORM_SIGNAL_EVENTS_ENABLED:-true}"
export BOT_PLATFORM_SIGNAL_EVENTS_STREAM="$BOT_SIGNAL_STREAM"
export BOT_PLATFORM_MARKET_DATA_EVENTS_ENABLED="${BOT_PLATFORM_MARKET_DATA_EVENTS_ENABLED:-true}"
export BOT_PLATFORM_MARKET_DATA_EVENTS_STREAM="$MARKET_DATA_EVENT_STREAM"
export BOT_PLATFORM_MARKET_DATA_EVENTS_BLOCK_MILLISECONDS="${BOT_PLATFORM_MARKET_DATA_EVENTS_BLOCK_MILLISECONDS:-1000}"
export BOT_PLATFORM_MARKET_DATA_EVENTS_RETRY_BACKOFF_SECONDS="${BOT_PLATFORM_MARKET_DATA_EVENTS_RETRY_BACKOFF_SECONDS:-1}"
export LOGIN_CHECKPOINT_ENABLED="${LOGIN_CHECKPOINT_ENABLED:-false}"

ADMIN_COOKIE_HEADER=""

if [[ "${SMOKE_INCLUDE_DEV_COMPOSE:-false}" == "true" ]]; then
  COMPOSE_FILES+=(-f infra/compose/docker-compose.dev.yaml)
fi

compose() {
  docker compose -p "$PROJECT_NAME" --env-file "$ENV_FILE" "${COMPOSE_FILES[@]}" \
    --profile infra --profile platform --profile identity "$@"
}

fail() {
  echo "Full Docker smoke failed: $*" >&2
  compose ps >&2 || true
  compose logs --tail=160 market_data bot_platform bot_platform_runner api >&2 || true
  exit 1
}

trap 'status=$?; fail "unexpected command failure at line ${LINENO} with exit code ${status}"' ERR

wait_for_service_health() {
  local service="$1"
  local label="$2"
  local deadline=$((SECONDS + SMOKE_WAIT_SECONDS))
  local command=()
  if [[ "$service" == "market_data" ]]; then
    command=(python -m market_data_service.main healthcheck)
  elif [[ "$service" == "bot_platform" ]]; then
    command=(python -m bot_platform_service.main healthcheck)
  else
    fail "unsupported healthcheck service: $service"
  fi
  until compose exec -T "$service" "${command[@]}" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      fail "$label did not become ready within ${SMOKE_WAIT_SECONDS}s"
    fi
    sleep 2
  done
}

wait_for_api() {
  local deadline=$((SECONDS + SMOKE_WAIT_SECONDS))
  until compose exec -T api node -e "
fetch('${API_URL}/health/readiness')
  .then(response => process.exit(response.ok ? 0 : 1))
  .catch(() => process.exit(1));
" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      fail "control-panel/api did not become ready within ${SMOKE_WAIT_SECONDS}s"
    fi
    sleep 2
  done
}

wait_for_snapshot() {
  local deadline=$((SECONDS + SMOKE_WAIT_SECONDS))
  local url="${MARKET_DATA_URL}/snapshots/latest?source=BINANCE_SPOT&symbol=${SMOKE_SYMBOL}&timeframe=${SMOKE_TIMEFRAME}"
  until compose exec -T market_data python -c "
import httpx
response = httpx.get('${url}', timeout=5.0)
response.raise_for_status()
raise SystemExit(0 if response.json().get('status') == 'ready' else 1)
" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      fail "market snapshot ${SMOKE_SYMBOL}/${SMOKE_TIMEFRAME} did not become ready"
    fi
    sleep 5
  done
}

wait_for_event_bot_run() {
  local deadline=$((SECONDS + SMOKE_WAIT_SECONDS))
  local run_count signal_count processed_count
  while true; do
    run_count="$(
      psql_scalar "
        select count(*)
        from _bot_platform.bot_runs
        where instance_id = '${SMOKE_INSTANCE_ID}'
          and trigger_type = 'event'
          and status = 'COMPLETE';
      "
    )"
    signal_count="$(
      psql_scalar "
        select count(*)
        from _bot_platform.bot_signals
        where instance_id = '${SMOKE_INSTANCE_ID}';
      "
    )"
    processed_count="$(psql_scalar "select count(*) from _bot_platform.market_data_processed_events;")"
    if [[ "$run_count" =~ ^[0-9]+$ && "$signal_count" =~ ^[0-9]+$ && "$processed_count" =~ ^[0-9]+$ ]] &&
      (( run_count > 0 && signal_count > 0 && processed_count > 0 )); then
      return 0
    fi
    if (( SECONDS >= deadline )); then
      fail "event-driven bot run did not complete for ${SMOKE_INSTANCE_ID}"
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

csv_to_json_array() {
  local csv="$1"
  local output="["
  local separator=""
  local item
  IFS=',' read -ra items <<< "$csv"
  for item in "${items[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "$item" ]]; then
      output+="${separator}\"${item^^}\""
      separator=","
    fi
  done
  output+="]"
  printf '%s' "$output"
}

api_get_with_cookie() {
  local path="$1"
  compose exec -T -e ADMIN_COOKIE_HEADER="$ADMIN_COOKIE_HEADER" -e API_REQUEST_PATH="$path" api node -e "
const url = '${API_URL}' + process.env.API_REQUEST_PATH;
fetch(url, { headers: { Cookie: process.env.ADMIN_COOKIE_HEADER || '' } })
  .then(async response => {
    const body = await response.text();
    if (!response.ok) {
      process.stderr.write(body);
      process.exit(1);
    }
    process.stdout.write(body);
  })
  .catch(error => {
    process.stderr.write(String(error));
    process.exit(1);
  });
"
}

api_post_json() {
  local path="$1"
  local body="$2"
  compose exec -T \
    -e ADMIN_COOKIE_HEADER="$ADMIN_COOKIE_HEADER" \
    -e SMOKE_CLIENT_ORIGIN="$SMOKE_CLIENT_ORIGIN" \
    -e API_REQUEST_PATH="$path" \
    -e API_REQUEST_BODY="$body" \
    api node -e "
const url = '${API_URL}' + process.env.API_REQUEST_PATH;
fetch(url, {
  method: 'POST',
  headers: {
    Cookie: process.env.ADMIN_COOKIE_HEADER || '',
    Origin: process.env.SMOKE_CLIENT_ORIGIN || '',
    'content-type': 'application/json',
  },
  body: process.env.API_REQUEST_BODY || '{}',
})
  .then(async response => {
    const body = await response.text();
    if (!response.ok) {
      process.stderr.write('API POST ' + process.env.API_REQUEST_PATH + ' failed with HTTP ' + response.status + '\n');
      process.stderr.write(body + '\n');
      process.exit(1);
    }
    process.stdout.write(body);
  })
  .catch(error => {
    process.stderr.write(String(error));
    process.exit(1);
  });
"
}

require_positive_count() {
  local label="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value <= 0 )); then
    fail "expected ${label} to be > 0, got '${value}'"
  fi
}

require_zero_count() {
  local label="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value != 0 )); then
    fail "expected ${label} to be 0, got '${value}'"
  fi
}

require_count_greater_than() {
  local label="$1"
  local value="$2"
  local minimum="$3"
  if [[ ! "$value" =~ ^[0-9]+$ ]] || [[ ! "$minimum" =~ ^[0-9]+$ ]] || (( value <= minimum )); then
    fail "expected ${label} to be > ${minimum}, got '${value}'"
  fi
}

require_count_unchanged() {
  local label="$1"
  local before="$2"
  local after="$3"
  if [[ "$before" != "$after" ]]; then
    fail "expected ${label} to stay ${before}, got '${after}'"
  fi
}

reset_market_data_smoke_state() {
  redis_scalar DEL "$MARKET_DATA_EVENT_STREAM" >/dev/null
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
  seed_market_data_smoke_symbol
}

seed_market_data_smoke_symbol() {
  psql_exec "
    insert into _market_data.market_symbols (
      canonical_symbol,
      base_asset,
      quote_asset,
      status
    )
    values (
      '${SMOKE_CANONICAL_SYMBOL}',
      regexp_replace('${SMOKE_CANONICAL_SYMBOL}', 'USDT$', ''),
      'USDT',
      'ACTIVE'
    )
    on conflict (canonical_symbol) do update
      set base_asset = excluded.base_asset,
          quote_asset = excluded.quote_asset,
          status = excluded.status,
          updated_at = now();

    insert into _market_data.provider_symbols (
      source,
      canonical_symbol,
      provider_symbol,
      status,
      supported_timeframes,
      max_backfill_days,
      metadata_json
    )
    values (
      'BINANCE_SPOT',
      '${SMOKE_CANONICAL_SYMBOL}',
      '${SMOKE_SYMBOL}',
      'TRADING',
      array['${SMOKE_TIMEFRAME}']::text[],
      1095,
      '{}'::jsonb
    )
    on conflict (source, provider_symbol) do update
      set canonical_symbol = excluded.canonical_symbol,
          status = excluded.status,
          supported_timeframes = excluded.supported_timeframes,
          max_backfill_days = excluded.max_backfill_days,
          metadata_json = excluded.metadata_json,
          updated_at = now();
  "
}

reset_bot_platform_smoke_state() {
  redis_scalar DEL "$BOT_SIGNAL_STREAM" >/dev/null
  psql_exec "
    delete from _bot_platform.market_data_processed_events;
    delete from _bot_platform.bot_audit_events
    where instance_id = '${SMOKE_INSTANCE_ID}'
       or actor_id = 'market_data_event_consumer'
       or event_id like 'stage40-%';
    delete from _bot_platform.bot_instances
    where instance_id = '${SMOKE_INSTANCE_ID}';
  "
}

prepare_incremental_gap() {
  psql_scalar "
    with gap_candles as (
      select open_time
      from _market_data.market_candles
      where source = 'BINANCE_SPOT'
        and canonical_symbol = '$SMOKE_CANONICAL_SYMBOL'
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
        and candle.canonical_symbol = '$SMOKE_CANONICAL_SYMBOL'
        and candle.timeframe = '$SMOKE_TIMEFRAME'
        and candle.open_time = gap_candles.open_time
      returning 1
    ),
    updated_state as (
      update _market_data.collection_states
      set bootstrap_to = bounds.incremental_from,
          bootstrap_next_from = bounds.incremental_from,
          bootstrap_completed_at = now(),
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

insert_old_event_cleanup_records() {
  psql_exec "
    insert into _bot_platform.market_data_processed_events (
      idempotency_key,
      event_type,
      contract_version,
      source,
      symbol,
      timeframe,
      snapshot_id,
      redis_message_id,
      processing_status,
      payload_json,
      processed_at
    )
    values (
      'stage40-old-processed-event',
      'MarketDataCandlesCollected',
      '1',
      'BINANCE_SPOT',
      '${SMOKE_SYMBOL}',
      '${SMOKE_TIMEFRAME}',
      'stage40-cleanup-snapshot',
      'stage40-cleanup-redis-message',
      'PROCESSED',
      '{\"source\":\"stage40\"}'::jsonb,
      now() - interval '6 days'
    )
    on conflict (idempotency_key) do update
      set processing_status = 'PROCESSED',
          processed_at = now() - interval '6 days';

    insert into _bot_platform.bot_audit_events (
      event_id,
      event_type,
      actor_type,
      actor_id,
      payload_json,
      created_at
    )
    values (
      'stage40-old-market-data-event-audit',
      'MARKET_DATA_EVENT_PROCESSED',
      'bot_platform',
      'market_data_event_consumer',
      '{\"source\":\"stage40\"}'::jsonb,
      now() - interval '6 days'
    )
    on conflict (event_id) do update
      set created_at = now() - interval '6 days';
  "
}

mark_collection_current_for_idempotent_rerun() {
  psql_exec "
    update _market_data.collection_states
    set last_successful_close_time = date_trunc('hour', now()),
        bootstrap_to = date_trunc('hour', now()),
        bootstrap_next_from = date_trunc('hour', now()),
        bootstrap_completed_at = coalesce(bootstrap_completed_at, now()),
        updated_at = now()
    where source = 'BINANCE_SPOT'
      and canonical_symbol = '$SMOKE_SYMBOL'
      and timeframe = '$SMOKE_TIMEFRAME';
  "
}

ensure_admin_user() {
  compose run --rm --no-deps api npm run cli -- admin:create \
    --admin-email="$SMOKE_ADMIN_EMAIL" \
    --admin-password="$SMOKE_ADMIN_PASSWORD" \
    --admin-firstname=Stage \
    --admin-lastname=Smoke \
    --role=platformAdmin >/tmp/stage26-admin-create.log 2>&1 || {
      if ! grep -q "already exists" /tmp/stage26-admin-create.log; then
        cat /tmp/stage26-admin-create.log >&2
        fail "admin:create failed"
      fi
    }
}

login_admin() {
  ADMIN_COOKIE_HEADER="$(
    compose exec -T \
      -e SMOKE_CLIENT_ORIGIN="$SMOKE_CLIENT_ORIGIN" \
      -e SMOKE_ADMIN_EMAIL="$SMOKE_ADMIN_EMAIL" \
      -e SMOKE_ADMIN_PASSWORD="$SMOKE_ADMIN_PASSWORD" \
      api node -e "
fetch('${API_URL}/admin/login', {
  method: 'POST',
  headers: {
    Origin: process.env.SMOKE_CLIENT_ORIGIN || '',
    'content-type': 'application/json',
  },
  body: JSON.stringify({
    login: process.env.SMOKE_ADMIN_EMAIL,
    password: process.env.SMOKE_ADMIN_PASSWORD,
  }),
})
  .then(async response => {
    if (!response.ok) {
      process.stderr.write(await response.text());
      process.exit(1);
    }
    const rawCookies = typeof response.headers.getSetCookie === 'function'
      ? response.headers.getSetCookie()
      : [response.headers.get('set-cookie')].filter(Boolean);
    const cookieHeader = rawCookies
      .flatMap(cookie => String(cookie).split(/,(?=[^;,]+=)/))
      .map(cookie => cookie.split(';')[0].trim())
      .filter(Boolean)
      .join('; ');
    process.stdout.write(cookieHeader);
  })
  .catch(error => {
    process.stderr.write(String(error));
    process.exit(1);
  });
"
  )" || fail "admin login failed"
  if [[ -z "$ADMIN_COOKIE_HEADER" ]]; then
    fail "admin login did not produce cookies"
  fi
}

ensure_instance() {
  local create_body
  local instances_json status
  instances_json="$(api_get_with_cookie "/admin/bot-instances")"
  if echo "$instances_json" | node -e "
let data = '';
process.stdin.on('data', c => data += c);
process.stdin.on('end', () => {
  const payload = JSON.parse(data);
  const instances = payload.data || payload.instances || [];
  const instance = instances.find(item => (item.instance_id || item.instanceId) === '${SMOKE_INSTANCE_ID}');
  process.exit(instance ? 0 : 1);
});
"; then
    status="$(echo "$instances_json" | node -e "
let data = '';
process.stdin.on('data', c => data += c);
process.stdin.on('end', () => {
  const payload = JSON.parse(data);
  const instances = payload.data || payload.instances || [];
  const instance = instances.find(item => (item.instance_id || item.instanceId) === '${SMOKE_INSTANCE_ID}');
  process.stdout.write(instance ? String(instance.status) : '');
});
")"
    if [[ "$status" != "ENABLED" ]]; then
      api_post_json "/admin/bot-instances/${SMOKE_INSTANCE_ID}/enable" "{}" >/dev/null ||
        fail "failed to enable existing bot instance"
    fi
    return
  fi

  local smoke_bot_symbols_json
  smoke_bot_symbols_json="$(csv_to_json_array "$MARKET_DATA_PROVIDER_SYMBOLS")"
  create_body="$(
    cat <<JSON
{"instanceId":"${SMOKE_INSTANCE_ID}","moduleId":"spot_grid","name":"Stage 26 Spot Grid","mode":"signal_only","symbols":${smoke_bot_symbols_json},"timeframes":["${SMOKE_TIMEFRAME}"],"configSchemaVersion":1,"config":{"symbols":${smoke_bot_symbols_json},"primary_timeframe":"${SMOKE_TIMEFRAME}","supporting_timeframes":[],"max_position_fraction":"0.10","max_grid_levels":6,"emit_diagnostics":true}}
JSON
  )"
  api_post_json "/admin/bot-instances" "$create_body" >/dev/null ||
    fail "failed to create bot instance"

  instances_json="$(api_get_with_cookie "/admin/bot-instances")"
  echo "$instances_json" | node -e "
let data = '';
process.stdin.on('data', c => data += c);
process.stdin.on('end', () => {
  const payload = JSON.parse(data);
  const instances = payload.data || payload.instances || [];
  const instance = instances.find(item => (item.instance_id || item.instanceId) === '${SMOKE_INSTANCE_ID}');
  process.exit(instance ? 0 : 1);
});
" || fail "created instance is not visible through control-panel API"

  status="$(echo "$instances_json" | node -e "
let data = '';
process.stdin.on('data', c => data += c);
process.stdin.on('end', () => {
  const payload = JSON.parse(data);
  const instances = payload.data || payload.instances || [];
  const instance = instances.find(item => (item.instance_id || item.instanceId) === '${SMOKE_INSTANCE_ID}');
  process.stdout.write(instance ? String(instance.status) : '');
});
")"
  if [[ "$status" != "ENABLED" ]]; then
    api_post_json "/admin/bot-instances/${SMOKE_INSTANCE_ID}/enable" "{}" >/dev/null ||
      fail "failed to enable bot instance"
  fi
}

assert_smoke_state() {
  local module_count run_count signal_count processed_count market_stream_count signal_stream_count legacy_count
  module_count="$(api_get_with_cookie "/admin/bot-modules" | node -e "let data=''; process.stdin.on('data', c => data += c); process.stdin.on('end', () => { const payload = JSON.parse(data); process.stdout.write(String((payload.data || payload.modules || []).length)); });")"
  require_positive_count "control-panel API bot module list" "$module_count"

  run_count="$(psql_scalar "select count(*) from _bot_platform.bot_runs where instance_id = '${SMOKE_INSTANCE_ID}' and trigger_type = 'event' and status = 'COMPLETE';")"
  signal_count="$(psql_scalar "select count(*) from _bot_platform.bot_signals where instance_id = '${SMOKE_INSTANCE_ID}';")"
  processed_count="$(psql_scalar "select count(*) from _bot_platform.market_data_processed_events;")"
  market_stream_count="$(redis_scalar XLEN "$MARKET_DATA_EVENT_STREAM")"
  signal_stream_count="$(redis_scalar XLEN "$BOT_SIGNAL_STREAM")"
  legacy_count="$(
    compose ps --services --filter status=running |
      awk '/(^|_)(spot_grid_bot|spot-greenwich-bot|legacy)/ { count++ } END { print count + 0 }'
  )"

  require_positive_count "event bot_runs for smoke instance" "$run_count"
  require_positive_count "bot_signals for smoke instance" "$signal_count"
  require_positive_count "processed market-data event records" "$processed_count"
  require_positive_count "Redis stream ${MARKET_DATA_EVENT_STREAM}" "$market_stream_count"
  require_positive_count "Redis stream ${BOT_SIGNAL_STREAM}" "$signal_stream_count"
  if [[ "$legacy_count" != "0" ]]; then
    fail "legacy bot services are running"
  fi
}

main() {
  if [[ "$MARKET_DATA_PROVIDER_MODE" != "fixture" ]]; then
    fail "full Docker smoke must run with MARKET_DATA_PROVIDER_MODE=fixture"
  fi

  compose stop market_data bot_platform_runner bot_platform api || true
  compose up --build -d postgres redis
  compose up --build market_data_migrate
  compose up --build bot_platform_migrate
  compose build api
  compose run --rm --no-deps api npm run cli -- db:migration:run
  compose run --rm bot_modules_sync
  reset_market_data_smoke_state
  reset_bot_platform_smoke_state

  compose run --rm market_data python -m market_data_service.main collect --once

  local bootstrap_candles_count
  local bootstrap_snapshots_count
  local bootstrap_market_stream_count
  local bootstrap_bot_run_count

  bootstrap_candles_count="$(psql_scalar "select count(*) from _market_data.market_candles where source = 'BINANCE_SPOT' and canonical_symbol = '$SMOKE_CANONICAL_SYMBOL' and timeframe = '$SMOKE_TIMEFRAME';")"
  bootstrap_snapshots_count="$(psql_scalar "select count(*) from _market_data.market_snapshots where source = 'BINANCE_SPOT' and canonical_symbol = '$SMOKE_CANONICAL_SYMBOL' and timeframe = '$SMOKE_TIMEFRAME';")"
  bootstrap_market_stream_count="$(redis_scalar XLEN "$MARKET_DATA_EVENT_STREAM")"
  bootstrap_bot_run_count="$(psql_scalar "select count(*) from _bot_platform.bot_runs where instance_id = '${SMOKE_INSTANCE_ID}';")"

  require_positive_count "bootstrap market_candles" "$bootstrap_candles_count"
  require_positive_count "bootstrap market_snapshots" "$bootstrap_snapshots_count"
  require_zero_count "bootstrap Redis stream ${MARKET_DATA_EVENT_STREAM}" "$bootstrap_market_stream_count"
  require_zero_count "bot_runs before incremental market-data event" "$bootstrap_bot_run_count"

  compose up --build -d market_data bot_platform api

  wait_for_service_health market_data "market-data-service"
  wait_for_service_health bot_platform "bot-platform-service"
  wait_for_api
  wait_for_snapshot

  ensure_admin_user
  login_admin
  ensure_instance

  compose up --build -d bot_platform_runner

  local deleted_trailing_candles_count
  deleted_trailing_candles_count="$(prepare_incremental_gap)"
  require_positive_count "prepared trailing candle gap" "$deleted_trailing_candles_count"

  compose run --rm market_data python -m market_data_service.main collect --once
  wait_for_event_bot_run
  assert_smoke_state

  local runs_before_cleanup_count
  local signals_before_cleanup_count
  local old_processed_before_cleanup_count
  local old_processed_after_cleanup_count
  local old_audit_after_cleanup_count
  local runs_after_cleanup_count
  local signals_after_cleanup_count

  runs_before_cleanup_count="$(psql_scalar "select count(*) from _bot_platform.bot_runs where instance_id = '${SMOKE_INSTANCE_ID}';")"
  signals_before_cleanup_count="$(psql_scalar "select count(*) from _bot_platform.bot_signals where instance_id = '${SMOKE_INSTANCE_ID}';")"
  insert_old_event_cleanup_records
  old_processed_before_cleanup_count="$(psql_scalar "select count(*) from _bot_platform.market_data_processed_events where idempotency_key = 'stage40-old-processed-event';")"
  require_positive_count "old market-data processed event cleanup fixture" "$old_processed_before_cleanup_count"

  compose run --rm bot_platform_runner python -m bot_platform_service.main events:cleanup

  old_processed_after_cleanup_count="$(psql_scalar "select count(*) from _bot_platform.market_data_processed_events where idempotency_key = 'stage40-old-processed-event';")"
  old_audit_after_cleanup_count="$(psql_scalar "select count(*) from _bot_platform.bot_audit_events where event_id = 'stage40-old-market-data-event-audit';")"
  runs_after_cleanup_count="$(psql_scalar "select count(*) from _bot_platform.bot_runs where instance_id = '${SMOKE_INSTANCE_ID}';")"
  signals_after_cleanup_count="$(psql_scalar "select count(*) from _bot_platform.bot_signals where instance_id = '${SMOKE_INSTANCE_ID}';")"
  require_zero_count "old market-data processed event cleanup fixture after cleanup" "$old_processed_after_cleanup_count"
  require_zero_count "old market-data event audit cleanup fixture after cleanup" "$old_audit_after_cleanup_count"
  require_count_unchanged "bot_runs after event cleanup" "$runs_before_cleanup_count" "$runs_after_cleanup_count"
  require_count_unchanged "bot_signals after event cleanup" "$signals_before_cleanup_count" "$signals_after_cleanup_count"

  local runs_before_idempotent_count
  local signals_before_idempotent_count
  local market_stream_before_idempotent_count
  local runs_after_idempotent_count
  local signals_after_idempotent_count
  local market_stream_after_idempotent_count

  mark_collection_current_for_idempotent_rerun
  runs_before_idempotent_count="$(psql_scalar "select count(*) from _bot_platform.bot_runs where instance_id = '${SMOKE_INSTANCE_ID}';")"
  signals_before_idempotent_count="$(psql_scalar "select count(*) from _bot_platform.bot_signals where instance_id = '${SMOKE_INSTANCE_ID}';")"
  market_stream_before_idempotent_count="$(redis_scalar XLEN "$MARKET_DATA_EVENT_STREAM")"
  compose run --rm market_data python -m market_data_service.main collect --once
  sleep 3
  runs_after_idempotent_count="$(psql_scalar "select count(*) from _bot_platform.bot_runs where instance_id = '${SMOKE_INSTANCE_ID}';")"
  signals_after_idempotent_count="$(psql_scalar "select count(*) from _bot_platform.bot_signals where instance_id = '${SMOKE_INSTANCE_ID}';")"
  market_stream_after_idempotent_count="$(redis_scalar XLEN "$MARKET_DATA_EVENT_STREAM")"
  require_count_unchanged "event bot_runs after idempotent rerun" "$runs_before_idempotent_count" "$runs_after_idempotent_count"
  require_count_unchanged "bot_signals after idempotent rerun" "$signals_before_idempotent_count" "$signals_after_idempotent_count"
  require_count_unchanged "market-data Redis stream after idempotent rerun" "$market_stream_before_idempotent_count" "$market_stream_after_idempotent_count"

  echo "Full Docker event-driven smoke passed."
}

main "$@"
