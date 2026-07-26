# Cloud and Provider Logging Integrations

This document shows deployment snippets for routing Pampilo API structured logs to common log backends without changing application code.

The API writes JSON logs to `stdout` / `stderr`. Infrastructure decides where those logs go.

## Log Fields

Expected API log fields:

```json
{
  "timestamp": "2026-07-07T12:00:00.000Z",
  "level": "info",
  "service": "pampilo-api",
  "environment": "production",
  "channel": "mail",
  "context": "MailProcessorService",
  "requestId": "req_...",
  "message": "Mail job completed"
}
```

Use these fields for routing and filtering:

- Low-cardinality labels/tags: `service`, `environment`, `level`, `channel`
- Search fields only: `context`, `requestId`, `userId`, `jobId`, `conversationId`

Do not promote request-specific identifiers to labels/tags.

## Docker Logs

For a plain Docker deployment, keep the default `json-file` driver or configure a platform-specific driver outside the application.

```bash
docker logs pampilo-platform-api-1
```

Basic filtering with `jq`:

```bash
docker logs pampilo-platform-api-1 2>&1 \
  | jq -r 'select(.service == "pampilo-api" and (.level == "error" or .level == "fatal"))'
```

Follow logs for one request:

```bash
docker logs -f pampilo-platform-api-1 2>&1 \
  | jq -r 'select(.requestId == "req_...")'
```

No application change is required as long as the API logs to `stdout` / `stderr`.

## Loki Through Grafana Alloy

Use the local stack in [README.md](./README.md) as the baseline. The same collector pattern applies to production container hosts.

Minimal Alloy pipeline:

```alloy
discovery.docker "containers" {
  host = "unix:///var/run/docker.sock"
}

loki.source.docker "containers" {
  host       = "unix:///var/run/docker.sock"
  targets    = discovery.docker.containers.targets
  forward_to = [loki.process.api.receiver]
}

loki.process "api" {
  stage.json {
    expressions = {
      timestamp   = "timestamp",
      level       = "level",
      service     = "service",
      environment = "environment",
      channel     = "channel",
      context     = "context",
      requestId   = "requestId",
      message     = "message",
    }
  }

  stage.labels {
    values = {
      service     = "",
      environment = "",
      level       = "",
      channel     = "",
    }
  }

  stage.structured_metadata {
    values = {
      context   = "",
      requestId = "",
    }
  }

  stage.timestamp {
    source = "timestamp"
    format = "RFC3339Nano"
  }

  forward_to = [loki.write.default.receiver]
}

loki.write "default" {
  endpoint {
    url = "http://loki:3100/loki/api/v1/push"
  }
}
```

LogQL examples:

```logql
{service="pampilo-api"} | json
{service="pampilo-api", level=~"error|fatal"} | json
{service="pampilo-api", channel="mail"} | json
{service="pampilo-api"} | json | context = "MailProcessorService"
{service="pampilo-api"} | json | requestId = "req_..."
```

## ELK or OpenSearch Through Fluent Bit

Use Fluent Bit to read container logs, parse the nested JSON payload, and send to Elasticsearch/OpenSearch.

Example `fluent-bit.conf`:

```ini
[SERVICE]
    Flush        1
    Log_Level    info
    Parsers_File parsers.conf

[INPUT]
    Name              tail
    Path              /var/lib/docker/containers/*/*.log
    Parser            docker
    Tag               docker.*
    Refresh_Interval  5
    Mem_Buf_Limit     50MB
    Skip_Long_Lines   On

[FILTER]
    Name          parser
    Match         docker.*
    Key_Name      log
    Parser        platform_json
    Reserve_Data  On

[FILTER]
    Name   modify
    Match  docker.*
    Add    app pampilo-platform

[OUTPUT]
    Name                  es
    Match                 docker.*
    Host                  ${OPENSEARCH_HOST}
    Port                  443
    TLS                   On
    HTTP_User             ${OPENSEARCH_USER}
    HTTP_Passwd           ${OPENSEARCH_PASSWORD}
    Index                 platform-logs
    Logstash_Format       On
    Logstash_Prefix       platform-api
    Suppress_Type_Name    On
    Replace_Dots          On
```

Example `parsers.conf`:

```ini
[PARSER]
    Name        docker
    Format      json
    Time_Key    time
    Time_Format %Y-%m-%dT%H:%M:%S.%L

[PARSER]
    Name        platform_json
    Format      json
    Time_Key    timestamp
    Time_Format %Y-%m-%dT%H:%M:%S.%LZ
```

OpenSearch Query DSL examples:

```json
{
  "query": {
    "bool": {
      "filter": [
        { "term": { "service.keyword": "pampilo-api" } },
        { "terms": { "level.keyword": ["error", "fatal"] } }
      ]
    }
  }
}
```

```json
{
  "query": {
    "bool": {
      "filter": [
        { "term": { "service.keyword": "pampilo-api" } },
        { "term": { "requestId.keyword": "req_..." } }
      ]
    }
  }
}
```

Keep `${OPENSEARCH_PASSWORD}` and similar credentials in the deployment secret store, not in the repository.

## Datadog Agent

For Docker hosts, enable Docker log collection in the Datadog Agent and parse JSON logs.

Example `docker-compose` service:

```yaml
services:
  datadog-agent:
    image: gcr.io/datadoghq/agent:7
    environment:
      DD_API_KEY: ${DD_API_KEY}
      DD_SITE: ${DD_SITE:-datadoghq.com}
      DD_LOGS_ENABLED: "true"
      DD_LOGS_CONFIG_CONTAINER_COLLECT_ALL: "true"
      DD_CONTAINER_EXCLUDE_LOGS: "name:pampilo-platform-datadog-agent-1"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - /proc/:/host/proc/:ro
      - /sys/fs/cgroup/:/host/sys/fs/cgroup:ro
```

Recommended Datadog tags:

```text
service:pampilo-api
env:production
level:error
channel:mail
```

Datadog log search examples:

```text
service:pampilo-api @level:(error OR fatal)
service:pampilo-api @channel:mail
service:pampilo-api @context:MailProcessorService
service:pampilo-api @requestId:req_...
```

Store `DD_API_KEY` in Docker secrets, ECS secrets, Kubernetes secrets, or the provider secret manager.

## AWS ECS or Fargate CloudWatch

The simplest ECS/Fargate path is the `awslogs` log driver. The application stays unchanged.

Task definition container log configuration:

```json
{
  "name": "api",
  "image": "ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/pampilo-api:TAG",
  "essential": true,
  "logConfiguration": {
    "logDriver": "awslogs",
    "options": {
      "awslogs-group": "/ecs/pampilo-api",
      "awslogs-region": "REGION",
      "awslogs-stream-prefix": "api"
    }
  }
}
```

CloudWatch Logs Insights examples:

```sql
fields @timestamp, level, service, environment, channel, context, requestId, message
| filter service = "pampilo-api"
| sort @timestamp desc
| limit 100
```

```sql
fields @timestamp, level, channel, context, requestId, message
| filter service = "pampilo-api" and level in ["error", "fatal"]
| sort @timestamp desc
```

```sql
fields @timestamp, level, channel, context, message
| filter service = "pampilo-api" and requestId = "req_..."
| sort @timestamp asc
```

For routing ECS/Fargate logs to OpenSearch, Datadog, or Loki, use FireLens/Fluent Bit sidecars and keep credentials in AWS Secrets Manager or SSM Parameter Store.

## AWS EKS Through Fluent Bit or Datadog Agent

For EKS, run Fluent Bit or Datadog Agent as a DaemonSet. Application pods only write JSON logs to stdout.

Example Fluent Bit Kubernetes filter/output shape:

```ini
[INPUT]
    Name              tail
    Tag               kube.*
    Path              /var/log/containers/*pampilo-api*.log
    Parser            cri
    Mem_Buf_Limit     50MB
    Skip_Long_Lines   On

[FILTER]
    Name                kubernetes
    Match               kube.*
    Merge_Log           On
    Keep_Log            Off
    K8S-Logging.Parser  On

[FILTER]
    Name          parser
    Match         kube.*
    Key_Name      log
    Parser        platform_json
    Reserve_Data  On

[OUTPUT]
    Name              cloudwatch_logs
    Match             kube.*
    region            ${AWS_REGION}
    log_group_name    /eks/pampilo-api
    log_stream_prefix api-
    auto_create_group true
```

For Datadog on EKS, enable container log collection and add pod labels/annotations for service and environment:

```yaml
metadata:
  labels:
    tags.datadoghq.com/service: pampilo-api
    tags.datadoghq.com/env: production
spec:
  containers:
    - name: api
      env:
        - name: DD_LOGS_INJECTION
          value: "false"
```

Datadog query examples are the same as the Docker Agent section.

## Google Cloud Run Cloud Logging

Cloud Run automatically collects `stdout` / `stderr` into Cloud Logging. No sidecar or code change is required.

Recommended service env:

```yaml
env:
  - name: NODE_ENV
    value: production
  - name: LOG_LEVEL
    value: info
  - name: LOG_FORMAT
    value: json
  - name: LOG_OUTPUT
    value: console
  - name: SERVICE_NAME
    value: pampilo-api
```

Cloud Logging query examples:

```text
resource.type="cloud_run_revision"
jsonPayload.service="pampilo-api"
```

```text
resource.type="cloud_run_revision"
jsonPayload.service="pampilo-api"
jsonPayload.level=("error" OR "fatal")
```

```text
resource.type="cloud_run_revision"
jsonPayload.service="pampilo-api"
jsonPayload.channel="mail"
```

```text
resource.type="cloud_run_revision"
jsonPayload.service="pampilo-api"
jsonPayload.requestId="req_..."
```

Use Secret Manager for database, JWT, SMTP, and other credentials.

## Google GKE Through Cloud Logging, Fluent Bit, or Datadog Agent

GKE can route container stdout to Cloud Logging through the managed logging pipeline. The API does not need collector-specific code.

Cloud Logging query examples:

```text
resource.type="k8s_container"
resource.labels.container_name="api"
jsonPayload.service="pampilo-api"
```

```text
resource.type="k8s_container"
jsonPayload.service="pampilo-api"
jsonPayload.level=("error" OR "fatal")
```

```text
resource.type="k8s_container"
jsonPayload.service="pampilo-api"
jsonPayload.context="MailProcessorService"
```

```text
resource.type="k8s_container"
jsonPayload.service="pampilo-api"
jsonPayload.requestId="req_..."
```

If sending GKE logs to Loki or OpenSearch, use Fluent Bit or Alloy as a DaemonSet and apply the same label policy:

- Labels/tags: `service`, `environment`, `level`, `channel`
- Search fields: `context`, `requestId`, `userId`, `jobId`, `conversationId`

For Datadog on GKE, deploy the Datadog Agent with log collection enabled and store the API key in a Kubernetes Secret or Google Secret Manager integration.

## Backend Change Policy

Changing between CloudWatch, Cloud Logging, Loki, OpenSearch, or Datadog must not require application code changes.

Allowed changes:

- environment variables such as `LOG_LEVEL`;
- deployment log driver or sidecar configuration;
- collector parser/routing configuration;
- dashboard/query definitions.

Disallowed changes:

- backend code that imports a provider SDK only for logging transport;
- credentials committed to the repository;
- adding high-cardinality fields as Loki labels, Datadog tags, or index-routing dimensions.
