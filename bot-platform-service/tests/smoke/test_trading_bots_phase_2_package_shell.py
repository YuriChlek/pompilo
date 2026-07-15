from __future__ import annotations

import importlib
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = SERVICE_ROOT / "src" / "bot_platform_service"
TRADING_BOTS_PACKAGE = PACKAGE_ROOT / "trading_bots"


def test_phase_2_trading_bots_package_shell_exists() -> None:
    init_file = TRADING_BOTS_PACKAGE / "__init__.py"

    assert TRADING_BOTS_PACKAGE.is_dir()
    assert init_file.exists()


def test_phase_2_trading_bots_package_documents_plugin_boundary() -> None:
    text = (TRADING_BOTS_PACKAGE / "__init__.py").read_text(encoding="utf-8")

    assert "discovery boundary" in text
    assert "signal-only bot modules" in text
    assert "manifest and config-schema metadata" in text


def test_phase_2_trading_bots_package_imports_cleanly() -> None:
    module = importlib.import_module("bot_platform_service.trading_bots")

    assert module.__name__ == "bot_platform_service.trading_bots"
