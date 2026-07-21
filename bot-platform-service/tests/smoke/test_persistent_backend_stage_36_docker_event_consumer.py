from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_stage_36_runner_starts_market_data_event_consumer_by_default() -> None:
    compose = (REPO_ROOT / "infra/compose/docker-compose.platform.yaml").read_text(encoding="utf-8")
    runner_block = compose.split("  bot_platform_runner:", 1)[1].split("  bot_modules_sync:", 1)[0]

    assert 'command: ["python", "-m", "bot_platform_service.main", "runner"]' in runner_block
    assert "redis:" in runner_block
    assert "condition: service_healthy" in runner_block
    assert "BOT_PLATFORM_MARKET_DATA_EVENTS_ENABLED: ${BOT_PLATFORM_MARKET_DATA_EVENTS_ENABLED:-true}" in runner_block
    assert (
        "BOT_PLATFORM_MARKET_DATA_EVENTS_REDIS_URL: "
        "${BOT_PLATFORM_MARKET_DATA_EVENTS_REDIS_URL:-redis://redis:6379/0}"
    ) in runner_block
    assert (
        "BOT_PLATFORM_MARKET_DATA_EVENTS_STREAM: "
        "${BOT_PLATFORM_MARKET_DATA_EVENTS_STREAM:-market-data-events}"
    ) in runner_block
    assert "BOT_PLATFORM_MARKET_DATA_EVENTS_CONSUMER_GROUP" in runner_block
    assert "BOT_PLATFORM_MARKET_DATA_EVENTS_CONSUMER_NAME" in runner_block
    assert "BOT_PLATFORM_MARKET_DATA_EVENTS_READ_COUNT" in runner_block
    assert "BOT_PLATFORM_MARKET_DATA_EVENTS_BLOCK_MILLISECONDS" in runner_block
    assert "BOT_PLATFORM_MARKET_DATA_EVENTS_RETRY_BACKOFF_SECONDS" in runner_block
    assert "BOT_PLATFORM_EVENT_IDEMPOTENCY_RETENTION_DAYS" in runner_block
    assert "BOT_PLATFORM_EVENT_AUDIT_RETENTION_DAYS" in runner_block


def test_stage_36_http_api_does_not_depend_on_redis_consumer_runtime() -> None:
    compose = (REPO_ROOT / "infra/compose/docker-compose.platform.yaml").read_text(encoding="utf-8")
    api_block = compose.split("  bot_platform:", 1)[1].split("  bot_platform_runner:", 1)[0]

    assert 'command: ["python", "-m", "bot_platform_service.main", "serve"]' in api_block
    assert "      redis:" not in api_block
    assert "BOT_PLATFORM_MARKET_DATA_EVENTS_ENABLED" not in api_block
    assert "BOT_PLATFORM_MARKET_DATA_EVENTS_REDIS_URL" not in api_block
    assert 'test: ["CMD", "python", "-m", "bot_platform_service.main", "healthcheck"]' in api_block
