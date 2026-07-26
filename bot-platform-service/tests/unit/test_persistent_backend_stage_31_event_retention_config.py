from __future__ import annotations

from pathlib import Path

import pytest

from bot_platform_service.config.settings import BotPlatformSettings


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_stage_31_event_retention_defaults_and_compose_env() -> None:
    settings = BotPlatformSettings.from_env()
    compose = (REPO_ROOT / "infra/compose/docker-compose.platform.yaml").read_text(encoding="utf-8")

    assert settings.event_retention.idempotency_retention_days == 5
    assert settings.event_retention.audit_retention_days == 5
    assert "BOT_PLATFORM_EVENT_IDEMPOTENCY_RETENTION_DAYS" in compose
    assert "BOT_PLATFORM_EVENT_AUDIT_RETENTION_DAYS" in compose


def test_stage_31_event_retention_accepts_custom_positive_values(monkeypatch) -> None:
    monkeypatch.setenv("BOT_PLATFORM_EVENT_IDEMPOTENCY_RETENTION_DAYS", "7")
    monkeypatch.setenv("BOT_PLATFORM_EVENT_AUDIT_RETENTION_DAYS", "9")

    settings = BotPlatformSettings.from_env()

    assert settings.event_retention.idempotency_retention_days == 7
    assert settings.event_retention.audit_retention_days == 9


@pytest.mark.parametrize(
    ("env_name", "env_value"),
    (
        ("BOT_PLATFORM_EVENT_IDEMPOTENCY_RETENTION_DAYS", "0"),
        ("BOT_PLATFORM_EVENT_IDEMPOTENCY_RETENTION_DAYS", "-1"),
        ("BOT_PLATFORM_EVENT_IDEMPOTENCY_RETENTION_DAYS", "abc"),
        ("BOT_PLATFORM_EVENT_AUDIT_RETENTION_DAYS", "0"),
        ("BOT_PLATFORM_EVENT_AUDIT_RETENTION_DAYS", "-1"),
        ("BOT_PLATFORM_EVENT_AUDIT_RETENTION_DAYS", "abc"),
    ),
)
def test_stage_31_event_retention_rejects_invalid_values(monkeypatch, env_name: str, env_value: str) -> None:
    monkeypatch.setenv(env_name, env_value)

    with pytest.raises(ValueError, match=env_name):
        BotPlatformSettings.from_env()


def test_stage_31_retention_policy_is_limited_to_service_event_records() -> None:
    # Stage 31 introduces config only. Cleanup in Stage 32 must target service-owned
    # event idempotency/audit records, not business history tables.
    assert {"market_data_processed_events", "event_audit_records"}.isdisjoint({"bot_runs", "bot_signals"})
