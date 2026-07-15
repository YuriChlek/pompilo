from __future__ import annotations

import importlib


def test_spot_grid_runtime_uses_platform_native_adapter() -> None:
    module = importlib.import_module("bot_platform_service.trading_bots.spot_grid.adapter")

    assert hasattr(module, "SpotGridAdapter")


def test_spot_grid_legacy_infrastructure_adapter_path_is_retired() -> None:
    try:
        importlib.import_module("bot_platform_service.infrastructure.bot_modules.spot_grid_adapter")
    except ModuleNotFoundError as exc:
        assert "bot_platform_service.infrastructure.bot_modules" in str(exc)
    else:  # pragma: no cover - documents the retired path expectation
        raise AssertionError("legacy spot_grid infrastructure adapter must not be importable")
