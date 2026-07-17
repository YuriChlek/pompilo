#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="${PROJECT_NAME:-pampilo-platform}"
ENV_FILE="${ENV_FILE:-control-panel/.env}"
COMPOSE_FILES=(
  -f infra/compose/docker-compose.yaml
  -f infra/compose/docker-compose.dev.yaml
)

API_URL="${API_URL:-http://localhost:${API_HOST_PORT:-3000}}"
BOT_PLATFORM_URL="${BOT_PLATFORM_URL:-http://localhost:${BOT_PLATFORM_HTTP_HOST_PORT:-8092}}"
MARKET_DATA_URL="${MARKET_DATA_URL:-http://localhost:${MARKET_DATA_HTTP_HOST_PORT:-8010}}"
SMOKE_CLIENT_ORIGIN="${SMOKE_CLIENT_ORIGIN:-${CLIENT_ORIGIN:-https://localhost}}"

DB_USER_VALUE="${DB_USER:-admin}"
DB_PASSWORD_VALUE="${DB_PASSWORD:-admin_pass}"
DB_NAME_VALUE="${DB_NAME:-pampilo_db}"

SMOKE_WAIT_SECONDS="${SMOKE_WAIT_SECONDS:-240}"
SMOKE_ADMIN_EMAIL="${SMOKE_ADMIN_EMAIL:-stage26-admin@example.com}"
SMOKE_ADMIN_PASSWORD="${SMOKE_ADMIN_PASSWORD:-Stage26Password123}"
SMOKE_INSTANCE_ID="${SMOKE_INSTANCE_ID:-stage26-spot-grid}"
SMOKE_IDEMPOTENCY_KEY="${SMOKE_IDEMPOTENCY_KEY:-stage26-full-docker-smoke-v1}"
SMOKE_SYMBOL="${SMOKE_SYMBOL:-ETHUSDT}"
SMOKE_TIMEFRAME="${SMOKE_TIMEFRAME:-1h}"
BOT_SIGNAL_STREAM="${BOT_PLATFORM_SIGNAL_EVENTS_STREAM:-bot-platform-signals}"

export MARKET_DATA_PROVIDER_SYMBOLS="${MARKET_DATA_PROVIDER_SYMBOLS:-$SMOKE_SYMBOL}"
export MARKET_DATA_TIMEFRAMES="${MARKET_DATA_TIMEFRAMES:-$SMOKE_TIMEFRAME}"
export MARKET_DATA_PROVIDER_MODE="${MARKET_DATA_PROVIDER_MODE:-fixture}"
export MARKET_DATA_SCHEDULER_POLL_INTERVAL_SECONDS="${MARKET_DATA_SCHEDULER_POLL_INTERVAL_SECONDS:-5}"
export MARKET_DATA_SCHEDULER_JITTER_SECONDS="${MARKET_DATA_SCHEDULER_JITTER_SECONDS:-0}"
export MARKET_DATA_1H_SAFETY_DELAY_SECONDS="${MARKET_DATA_1H_SAFETY_DELAY_SECONDS:-0}"
export BOT_PLATFORM_SIGNAL_EVENTS_ENABLED="${BOT_PLATFORM_SIGNAL_EVENTS_ENABLED:-true}"
export BOT_PLATFORM_SIGNAL_EVENTS_STREAM="$BOT_SIGNAL_STREAM"
export LOGIN_CHECKPOINT_ENABLED="${LOGIN_CHECKPOINT_ENABLED:-false}"

TMP_DIR="$(mktemp -d)"
COOKIE_JAR="$TMP_DIR/admin-cookies.txt"
ADMIN_COOKIE_HEADER=""
trap 'rm -rf "$TMP_DIR"' EXIT

compose() {
  docker compose -p "$PROJECT_NAME" --env-file "$ENV_FILE" "${COMPOSE_FILES[@]}" \
    --profile infra --profile platform --profile identity "$@"
}

fail() {
  echo "Full Docker smoke failed: $*" >&2
  compose ps >&2 || true
  compose logs --tail=160 market_data bot_platform api >&2 || true
  exit 1
}

trap 'status=$?; fail "unexpected command failure at line ${LINENO} with exit code ${status}"' ERR

wait_for_http() {
  local url="$1"
  local label="$2"
  local deadline=$((SECONDS + SMOKE_WAIT_SECONDS))
  until curl -fsS "$url" >/dev/null; do
    if (( SECONDS >= deadline )); then
      fail "$label did not become ready at $url within ${SMOKE_WAIT_SECONDS}s"
    fi
    sleep 2
  done
}

wait_for_snapshot() {
  local deadline=$((SECONDS + SMOKE_WAIT_SECONDS))
  local url="${MARKET_DATA_URL}/snapshots/latest?source=BINANCE_SPOT&symbol=${SMOKE_SYMBOL}&timeframe=${SMOKE_TIMEFRAME}"
  until curl -sS "$url" | node -e "let data=''; process.stdin.on('data', c => data += c); process.stdin.on('end', () => { try { process.exit(JSON.parse(data).status === 'ready' ? 0 : 1); } catch { process.exit(1); } });"; do
    if (( SECONDS >= deadline )); then
      fail "market snapshot ${SMOKE_SYMBOL}/${SMOKE_TIMEFRAME} did not become ready"
    fi
    sleep 5
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

json_field() {
  local field_path="$1"
  node -e "
let data = '';
process.stdin.on('data', chunk => data += chunk);
process.stdin.on('end', () => {
  const payload = JSON.parse(data);
  const value = '$field_path'.split('.').reduce((current, key) => current == null ? undefined : current[key], payload);
  if (value === undefined || value === null) {
    process.exit(1);
  }
  process.stdout.write(String(value));
});
"
}

api_get_with_cookie() {
  local path="$1"
  curl -fsS -H "Cookie: ${ADMIN_COOKIE_HEADER}" "${API_URL}${path}"
}

api_post_json() {
  local path="$1"
  local body="$2"
  local response_file status
  response_file="$(mktemp "$TMP_DIR/api-response.XXXXXX")"
  status="$(
    curl -sS -o "$response_file" -w "%{http_code}" \
      -H "Cookie: ${ADMIN_COOKIE_HEADER}" \
      -H "Origin: ${SMOKE_CLIENT_ORIGIN}" \
      -H "content-type: application/json" \
      -X POST "${API_URL}${path}" \
      -d "$body" || true
  )"

  if [[ ! "$status" =~ ^2[0-9][0-9]$ ]]; then
    echo "API POST ${path} failed with HTTP ${status}" >&2
    cat "$response_file" >&2
    echo >&2
    return 1
  fi

  cat "$response_file"
}

require_positive_count() {
  local label="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value <= 0 )); then
    fail "expected ${label} to be > 0, got '${value}'"
  fi
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
  curl -fsS -c "$COOKIE_JAR" \
    -H "Origin: ${SMOKE_CLIENT_ORIGIN}" \
    -H "content-type: application/json" \
    -X POST "${API_URL}/admin/login" \
    -d "{\"login\":\"${SMOKE_ADMIN_EMAIL}\",\"password\":\"${SMOKE_ADMIN_PASSWORD}\"}" >/dev/null ||
    fail "admin login failed"
  ADMIN_COOKIE_HEADER="$(
    awk '
      BEGIN { sep = "" }
      /^#HttpOnly_/ { sub(/^#HttpOnly_/, "", $1) }
      /^#/ { next }
      NF >= 7 { printf "%s%s=%s", sep, $6, $7; sep = "; " }
    ' "$COOKIE_JAR"
  )"
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

  create_body="$(
    cat <<JSON
{"instanceId":"${SMOKE_INSTANCE_ID}","moduleId":"spot_grid","name":"Stage 26 Spot Grid","mode":"signal_only","symbols":["${SMOKE_SYMBOL}"],"timeframes":["${SMOKE_TIMEFRAME}"],"configSchemaVersion":1,"config":{"symbols":["${SMOKE_SYMBOL}"],"primary_timeframe":"${SMOKE_TIMEFRAME}","supporting_timeframes":[],"max_position_fraction":"0.10","max_grid_levels":6,"emit_diagnostics":true}}
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

manual_run() {
  local run_body
  run_body="{\"idempotencyKey\":\"${SMOKE_IDEMPOTENCY_KEY}\",\"correlationId\":\"stage26-full-docker-smoke\"}"
  local response
  response="$(api_post_json "/admin/bot-instances/${SMOKE_INSTANCE_ID}/run" "$run_body" || true)"
  if [[ -z "$response" ]]; then
    fail "manual run returned an empty response"
  fi

  local accepted duplicate error_code
  accepted="$(echo "$response" | json_field "data.accepted" || true)"
  duplicate="$(echo "$response" | json_field "data.duplicate" || true)"
  error_code="$(echo "$response" | json_field "data.error_code" || true)"

  if [[ "$accepted" != "true" && "$duplicate" != "true" ]]; then
    echo "$response" >&2
    fail "manual run was not accepted and was not an idempotent duplicate, error_code=${error_code:-unknown}"
  fi
}

assert_smoke_state() {
  local module_count run_count signal_count stream_count legacy_count
  module_count="$(api_get_with_cookie "/admin/bot-modules" | node -e "let data=''; process.stdin.on('data', c => data += c); process.stdin.on('end', () => { const payload = JSON.parse(data); process.stdout.write(String((payload.data || payload.modules || []).length)); });")"
  require_positive_count "control-panel API bot module list" "$module_count"

  run_count="$(psql_scalar "select count(*) from _bot_platform.bot_runs where instance_id = '${SMOKE_INSTANCE_ID}' and idempotency_key = '${SMOKE_IDEMPOTENCY_KEY}';")"
  signal_count="$(psql_scalar "select count(*) from _bot_platform.bot_signals where instance_id = '${SMOKE_INSTANCE_ID}';")"
  stream_count="$(redis_scalar XLEN "$BOT_SIGNAL_STREAM")"
  legacy_count="$(
    compose ps --services --filter status=running |
      awk '/(^|_)(spot_grid_bot|spot-greenwich-bot|legacy)/ { count++ } END { print count + 0 }'
  )"

  require_positive_count "bot_runs for smoke instance" "$run_count"
  require_positive_count "bot_signals for smoke instance" "$signal_count"
  require_positive_count "Redis stream ${BOT_SIGNAL_STREAM}" "$stream_count"
  if [[ "$legacy_count" != "0" ]]; then
    fail "legacy bot services are running"
  fi
}

main() {
  compose stop market_data bot_platform_runner bot_platform api || true
  compose up --build -d postgres redis
  compose up --build market_data_migrate
  compose run --rm market_data_symbols_sync
  compose run --rm market_data python -m market_data_service.main scheduler:run-once
  compose run --rm market_data python -m market_data_service.main sync:run-next
  compose run --rm market_data python -m market_data_service.main outbox:publish-once
  compose up --build bot_platform_migrate
  compose build api
  compose run --rm --no-deps api npm run cli -- db:migration:run
  compose run --rm bot_modules_sync
  compose up --build -d market_data bot_platform bot_platform_runner api

  wait_for_http "${MARKET_DATA_URL}/health/ready" "market-data-service"
  wait_for_http "${BOT_PLATFORM_URL}/health/ready" "bot-platform-service"
  wait_for_http "${API_URL}/health/readiness" "control-panel/api"
  wait_for_snapshot

  ensure_admin_user
  login_admin
  ensure_instance
  manual_run
  assert_smoke_state

  echo "Full Docker smoke passed."
}

main "$@"
