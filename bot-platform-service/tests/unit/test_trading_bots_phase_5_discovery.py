from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path

import pytest

from bot_platform_service.registry import (
    BotManifestValidationError,
    BotModuleRegistry,
    discover_and_register_trading_bots,
    discover_trading_bot_registrations,
)


class _Repository:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None, object | None]] = []

    async def register_module(self, manifest, *, adapter_path: str, adapter_class: str | None = None, config_schema=None) -> bool:
        self.calls.append((manifest.module_id, adapter_path, adapter_class, config_schema))
        return True


def test_phase_5_discovers_fake_module_without_importing_adapter(tmp_path: Path) -> None:
    _write_bot_package(tmp_path, "spot_grid")

    with _tmp_trading_bots_path(tmp_path):
        registrations = discover_trading_bot_registrations()

    assert len(registrations) == 1
    registration = registrations[0]
    assert registration.manifest.module_id == "spot_grid"
    assert registration.adapter_path == "bot_platform_service.trading_bots.spot_grid.adapter"
    assert registration.adapter_class == "SpotGridAdapter"
    assert registration.config_schema is not None
    assert "bot_platform_service.trading_bots.spot_grid.adapter" not in sys.modules


def test_phase_5_registers_discovered_metadata_through_registry(tmp_path: Path) -> None:
    _write_bot_package(tmp_path, "spot_grid")
    _write_bot_package(tmp_path, "spot_greenwich", adapter_class="SpotGreenwichAdapter", timeframes=("1d", "4h"))
    repository = _Repository()
    registry = BotModuleRegistry(repository)

    with _tmp_trading_bots_path(tmp_path):
        registered = asyncio.run(discover_and_register_trading_bots(registry))

    assert registered == ("spot_greenwich", "spot_grid")
    assert [call[0] for call in repository.calls] == ["spot_greenwich", "spot_grid"]
    assert repository.calls[0][2] == "SpotGreenwichAdapter"
    assert repository.calls[1][2] == "SpotGridAdapter"


def test_phase_5_discovery_rejects_bot_suffix_ids(tmp_path: Path) -> None:
    _write_bot_package(tmp_path, "spot_grid_bot")

    with _tmp_trading_bots_path(tmp_path), pytest.raises(BotManifestValidationError, match="_bot"):
        discover_trading_bot_registrations()


def test_phase_5_discovery_rejects_module_id_package_mismatch(tmp_path: Path) -> None:
    _write_bot_package(tmp_path, "spot_grid", module_id="grid")

    with _tmp_trading_bots_path(tmp_path), pytest.raises(BotManifestValidationError, match="must match"):
        discover_trading_bot_registrations()


def _write_bot_package(
    tmp_path: Path,
    child_name: str,
    *,
    module_id: str | None = None,
    adapter_class: str = "SpotGridAdapter",
    timeframes: tuple[str, ...] = ("1h", "4h"),
) -> None:
    bot_dir = tmp_path / child_name
    bot_dir.mkdir()
    (bot_dir / "__init__.py").write_text("", encoding="utf-8")
    (bot_dir / "adapter.py").write_text("raise RuntimeError('adapter imported during discovery')\n", encoding="utf-8")
    raw_module_id = module_id or child_name
    (bot_dir / "manifest.py").write_text(
        "RAW_MANIFEST = {\n"
        f"    'module_id': '{raw_module_id}',\n"
        f"    'display_name': '{raw_module_id.replace('_', ' ').title()}',\n"
        "    'version': '1.0.0',\n"
        "    'supported_modes': ('dry_run', 'notification_only', 'signal_only'),\n"
        f"    'required_timeframes': {timeframes!r},\n"
        "    'required_market_data': ('snapshots',),\n"
        "    'supports_multi_symbol': True,\n"
        "    'config_schema_version': 1,\n"
        "}\n"
        f"ADAPTER_PATH = 'bot_platform_service.trading_bots.{child_name}.adapter'\n"
        f"ADAPTER_CLASS = '{adapter_class}'\n",
        encoding="utf-8",
    )
    (bot_dir / "config_schema.py").write_text(
        "CONFIG_SCHEMA = {\n"
        "    'schema_version': 1,\n"
        "    'sections': [\n"
        "        {\n"
        "            'key': 'market_data',\n"
        "            'label': 'Market Data',\n"
        "            'fields': [\n"
        "                {'key': 'symbols', 'type': 'symbol_list', 'label': 'Symbols', 'required': True, 'default': ['ETHUSDT']},\n"
        "            ],\n"
        "        }\n"
        "    ],\n"
        "}\n",
        encoding="utf-8",
    )


class _tmp_trading_bots_path:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self):
        import bot_platform_service.trading_bots as trading_bots

        self.package = trading_bots
        self.original_path = list(trading_bots.__path__)
        trading_bots.__path__[:] = [str(self.path)]
        importlib.invalidate_caches()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.package.__path__[:] = self.original_path
        for module_name in list(sys.modules):
            if module_name.startswith("bot_platform_service.trading_bots.") and module_name != "bot_platform_service.trading_bots":
                sys.modules.pop(module_name, None)
        importlib.invalidate_caches()
