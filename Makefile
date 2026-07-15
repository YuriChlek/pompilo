COMPOSE_PROJECT = pampilo-platform
COMPOSE_DEV = docker compose -p $(COMPOSE_PROJECT) --env-file register-service/.env -f infra/compose/docker-compose.yaml -f infra/compose/docker-compose.dev.yaml
COMPOSE_PROD = docker compose -p $(COMPOSE_PROJECT) --env-file register-service/.env.production -f infra/compose/docker-compose.yaml -f infra/compose/docker-compose.prod.yaml
PROD_PROFILES = --profile infra --profile identity --profile trading --profile observability
DEV_PROFILES = --profile infra --profile identity --profile trading --profile dev --profile mail --profile observability
PROD_SERVICES = postgres redis api client gateway spot_grid_bot loki alloy grafana

.PHONY: docker-dev docker-prod docker-config-dev docker-config-prod
.PHONY: docker-dev-ps docker-prod-ps docker-dev-logs docker-prod-logs docker-dev-down docker-prod-down
.PHONY: ops-load-smoke ops-backup-restore-smoke
.PHONY: build-api build-client test-api test-client

docker-dev:
	@node infra/scripts/docker-dev-select.mjs

docker-prod:
	$(COMPOSE_PROD) $(PROD_PROFILES) up -d --build $(PROD_SERVICES)

docker-config-dev:
	$(COMPOSE_DEV) $(DEV_PROFILES) config --quiet

docker-config-prod:
	$(COMPOSE_PROD) $(PROD_PROFILES) config --quiet

docker-dev-ps:
	$(COMPOSE_DEV) $(DEV_PROFILES) ps

docker-prod-ps:
	$(COMPOSE_PROD) $(PROD_PROFILES) ps

docker-dev-logs:
	$(COMPOSE_DEV) $(DEV_PROFILES) logs -f

docker-prod-logs:
	$(COMPOSE_PROD) $(PROD_PROFILES) logs -f

docker-dev-down:
	$(COMPOSE_DEV) $(DEV_PROFILES) down

docker-prod-down:
	$(COMPOSE_PROD) $(PROD_PROFILES) down

ops-load-smoke:
	node register-service/ops/load/register-service-smoke-load.mjs

ops-backup-restore-smoke:
	bash register-service/ops/scripts/backup-restore-smoke.sh

build-api:
	npm --prefix register-service/api run build

build-client:
	npm --prefix register-service/client run build

test-api:
	npm --prefix register-service/api run test

test-client:
	npm --prefix register-service/client run test
