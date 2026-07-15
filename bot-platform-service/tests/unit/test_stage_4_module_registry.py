from __future__ import annotations

import asyncio

import pytest

from bot_platform_service.domain import BotMode
from bot_platform_service.registry import (
    BotManifestValidationError,
    BotModuleRegistry,
    TRADING_BOTS_ADAPTER_PATH_PREFIX,
    build_registration,
    parse_manifest,
    validate_adapter_class,
    validate_adapter_path,
    validate_platform_module_id,
)


class _Repository:
    def __init__(self, changed: bool = True) -> None:
        self.changed = changed
        self.calls: list[tuple[str, str, str | None, object | None]] = []

    async def register_module(self, manifest, *, adapter_path: str, adapter_class: str | None = None, config_schema=None) -> bool:
        self.calls.append((manifest.module_id, adapter_path, adapter_class, config_schema))
        return self.changed


def test_parse_manifest_accepts_spot_grid_metadata() -> None:
    manifest = parse_manifest(_spot_grid_manifest())

    assert manifest.module_id == "spot_grid"
    assert manifest.supported_modes == (BotMode.DRY_RUN, BotMode.NOTIFICATION_ONLY, BotMode.SIGNAL_ONLY)
    assert manifest.required_timeframes == ("1h", "4h")


def test_build_registration_accepts_spot_greenwich_metadata_without_importing_adapter() -> None:
    registration = build_registration(
        _spot_greenwich_manifest(),
        adapter_path=f"{TRADING_BOTS_ADAPTER_PATH_PREFIX}spot_greenwich.adapter",
    )

    assert registration.manifest.module_id == "spot_greenwich"
    assert registration.adapter_path.endswith("spot_greenwich.adapter")
    assert registration.adapter_class is None
    assert registration.config_schema is None


def test_build_registration_accepts_adapter_class_and_config_schema_metadata() -> None:
    config_schema = _config_schema()
    registration = build_registration(
        _spot_grid_manifest(),
        adapter_path=f"{TRADING_BOTS_ADAPTER_PATH_PREFIX}spot_grid.adapter",
        adapter_class="SpotGridAdapter",
        config_schema=config_schema,
    )

    assert registration.adapter_class == "SpotGridAdapter"
    assert registration.config_schema is config_schema


def test_adapter_class_must_be_pascal_case_class_name() -> None:
    validate_adapter_class("SpotGridAdapter")

    with pytest.raises(BotManifestValidationError):
        validate_adapter_class("spot_grid_adapter")


def test_adapter_path_must_point_to_trading_bots_package() -> None:
    with pytest.raises(BotManifestValidationError):
        validate_adapter_path("bot_platform_service.domain.spot_grid_adapter")


def test_adapter_path_accepts_trading_bots_package_local_adapter() -> None:
    validate_adapter_path(f"{TRADING_BOTS_ADAPTER_PATH_PREFIX}spot_grid.adapter")


def test_adapter_path_rejects_retired_infrastructure_bot_modules_package() -> None:
    with pytest.raises(BotManifestValidationError):
        validate_adapter_path("bot_platform_service.infrastructure.bot_modules.spot_grid_adapter")


def test_platform_module_id_rejects_bot_suffix() -> None:
    validate_platform_module_id("spot_grid")

    with pytest.raises(BotManifestValidationError):
        validate_platform_module_id("spot_grid_bot")


def test_manifest_rejects_unsupported_timeframe() -> None:
    raw_manifest = dict(_spot_grid_manifest())
    raw_manifest["required_timeframes"] = ("15m",)

    with pytest.raises(BotManifestValidationError):
        parse_manifest(raw_manifest)


def test_manifest_rejects_non_snake_case_module_id() -> None:
    raw_manifest = dict(_spot_grid_manifest())
    raw_manifest["module_id"] = "SpotGridBot"

    with pytest.raises(BotManifestValidationError):
        parse_manifest(raw_manifest)


def test_registry_registers_module_metadata_duplicate_safely() -> None:
    repository = _Repository(changed=False)
    registry = BotModuleRegistry(repository)
    registration = build_registration(
        _spot_grid_manifest(),
        adapter_path=f"{TRADING_BOTS_ADAPTER_PATH_PREFIX}spot_grid.adapter",
    )

    result = asyncio.run(registry.register(registration))

    assert result.module_id == "spot_grid"
    assert result.changed is False
    assert repository.calls == [("spot_grid", f"{TRADING_BOTS_ADAPTER_PATH_PREFIX}spot_grid.adapter", None, None)]


def test_registry_persists_extended_module_metadata_when_available() -> None:
    repository = _Repository(changed=True)
    registry = BotModuleRegistry(repository)
    config_schema = _config_schema()
    registration = build_registration(
        _spot_grid_manifest(),
        adapter_path=f"{TRADING_BOTS_ADAPTER_PATH_PREFIX}spot_grid.adapter",
        adapter_class="SpotGridAdapter",
        config_schema=config_schema,
    )

    result = asyncio.run(registry.register(registration))

    assert result.changed is True
    assert repository.calls == [("spot_grid", f"{TRADING_BOTS_ADAPTER_PATH_PREFIX}spot_grid.adapter", "SpotGridAdapter", config_schema)]


def test_registry_rejects_invalid_config_schema_metadata() -> None:
    with pytest.raises(Exception):
        build_registration(
            _spot_grid_manifest(),
            adapter_path=f"{TRADING_BOTS_ADAPTER_PATH_PREFIX}spot_grid.adapter",
            adapter_class="SpotGridAdapter",
            config_schema={"schema_version": 1, "sections": []},
        )


def _spot_grid_manifest() -> dict[str, object]:
    return {
        "module_id": "spot_grid",
        "display_name": "Spot Grid Bot",
        "version": "1.0.0",
        "supported_modes": ("dry_run", "notification_only", "signal_only"),
        "required_timeframes": ("1h", "4h"),
        "required_market_data": ("candles", "snapshots"),
        "supports_multi_symbol": True,
        "config_schema_version": 1,
    }


def _spot_greenwich_manifest() -> dict[str, object]:
    return {
        "module_id": "spot_greenwich",
        "display_name": "Spot Greenwich Bot",
        "version": "1.0.0",
        "supported_modes": ("dry_run", "notification_only", "signal_only"),
        "required_timeframes": ("1d", "4h"),
        "required_market_data": ("candles", "snapshots"),
        "supports_multi_symbol": True,
        "config_schema_version": 1,
    }


def _config_schema() -> dict[str, object]:
    return {
        "schema_version": 1,
        "sections": [
            {
                "key": "market_data",
                "label": "Market Data",
                "fields": [
                    {
                        "key": "symbols",
                        "type": "symbol_list",
                        "label": "Symbols",
                        "required": True,
                        "default": ["ETHUSDT"],
                    }
                ],
            }
        ],
    }
