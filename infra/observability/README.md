# Local Observability Stack

This stack is optional and is intended for local verification of structured API logs.

Cloud and provider integration examples are documented in [cloud-integrations.md](./cloud-integrations.md).

## Start

```bash
docker compose --profile observability up -d
```

Grafana is available at `http://localhost:3002`.

Default credentials:

```text
admin / admin
```

## Log Flow

```text
api stdout/stderr
  -> Docker logging stream
  -> Grafana Alloy
  -> Loki
  -> Grafana
```

## Queries

All API logs:

```logql
{service="pampilo-api"} | json
```

Errors:

```logql
{service="pampilo-api", level=~"error|fatal"} | json
```

Channel logs:

```logql
{service="pampilo-api", channel="mail"} | json
```

Request correlation:

```logql
{service="pampilo-api"} | json | requestId = "req_..."
```

## Label Policy

Alloy only promotes low-cardinality fields to Loki labels:

- `service`
- `environment`
- `level`
- `channel`

High-cardinality values such as `requestId`, `userId`, `jobId`, and `conversationId` remain searchable log fields or structured metadata, not labels.

## Optional File Output

The recommended Docker/cloud path is still `LOG_OUTPUT=console`.

For local VM-style deployments, the API can write rotating JSON log files:

```env
LOG_OUTPUT=file
LOG_FORMAT=json
LOG_FILE_DIR=logs
LOG_FILE_MAX_DAYS=14
```

File output writes daily files named `<service-name>-YYYY-MM-DD.log` and removes files older than `LOG_FILE_MAX_DAYS`.

When running in a container, mount a writable volume to `LOG_FILE_DIR`. Do not use local file output for horizontally scaled cloud deployments unless an external collector or shared log shipping process is responsible for collecting those files.
