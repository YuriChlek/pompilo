from __future__ import annotations

import importlib


def test_spot_greenwich_runtime_uses_platform_native_adapter() -> None:
    module = importlib.import_module("bot_platform_service.trading_bots.spot_greenwich.adapter")

    assert hasattr(module, "SpotGreenwichAdapter")


def test_spot_greenwich_legacy_infrastructure_adapter_path_is_retired() -> None:
    try:
        importlib.import_module("bot_platform_service.infrastructure.bot_modules.spot_greenwich_adapter")
    except ModuleNotFoundError as exc:
        assert "bot_platform_service.infrastructure.bot_modules" in str(exc)
    else:  # pragma: no cover - documents the retired path expectation
        raise AssertionError("legacy spot_greenwich infrastructure adapter must not be importable")
