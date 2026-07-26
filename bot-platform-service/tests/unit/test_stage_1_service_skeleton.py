from __future__ import annotations

import importlib

import bot_platform_service
from bot_platform_service.main import main


def test_package_imports() -> None:
    assert bot_platform_service.__version__ == "0.1.0"


def test_layer_packages_import() -> None:
    modules = [
        "bot_platform_service.application",
        "bot_platform_service.config",
        "bot_platform_service.domain",
        "bot_platform_service.infrastructure",
        "bot_platform_service.observability",
        "bot_platform_service.persistence",
        "bot_platform_service.registry",
        "bot_platform_service.trading_bots",
        "bot_platform_service.workers",
    ]

    for module_name in modules:
        assert importlib.import_module(module_name).__name__ == module_name


def test_stage_1_entrypoint_is_safe_noop() -> None:
    assert main() == 0
