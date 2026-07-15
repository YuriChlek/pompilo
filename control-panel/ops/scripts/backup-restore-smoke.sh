#!/usr/bin/env bash
set -euo pipefail

POSTGRES_CONTAINER="${POSTGRES_CONTAINER:-pampilo-platform-postgres-1}"
REDIS_CONTAINER="${REDIS_CONTAINER:-pampilo-platform-redis-1}"
DB_USER="${DB_USER:-admin}"
DB_NAME="${DB_NAME:-pampilo_db}"
RESTORE_DB="${RESTORE_DB:-phase23_restore_probe}"
STAMP="$(date +%Y%m%d%H%M%S)"
PG_DUMP_PATH="/tmp/phase23_${DB_NAME}_${STAMP}.dump"
REDIS_RDB_PATH="/tmp/phase23_redis_${STAMP}.rdb"

docker exec "$POSTGRES_CONTAINER" pg_dump -U "$DB_USER" -d "$DB_NAME" -Fc -f "$PG_DUMP_PATH"
docker exec "$POSTGRES_CONTAINER" psql -U "$DB_USER" -d postgres -v ON_ERROR_STOP=1 \
  -c "drop database if exists ${RESTORE_DB};" \
  -c "create database ${RESTORE_DB};"
docker exec "$POSTGRES_CONTAINER" pg_restore -U "$DB_USER" -d "$RESTORE_DB" "$PG_DUMP_PATH"
docker exec "$POSTGRES_CONTAINER" psql -U "$DB_USER" -d postgres -v ON_ERROR_STOP=1 \
  -c "drop database if exists ${RESTORE_DB};"

docker exec "$REDIS_CONTAINER" redis-cli --rdb "$REDIS_RDB_PATH"
docker exec "$REDIS_CONTAINER" ls -lh "$REDIS_RDB_PATH"

echo "Backup/restore smoke completed: postgres=${PG_DUMP_PATH}, redis=${REDIS_RDB_PATH}"
