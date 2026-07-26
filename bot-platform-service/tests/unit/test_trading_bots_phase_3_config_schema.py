from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from bot_platform_service.domain import ConfigSchemaValidationError, SUPPORTED_CONFIG_FIELD_TYPES, assert_json_safe, validate_config_schema


def test_phase_3_valid_config_schema_fixture_passes() -> None:
    validate_config_schema(_valid_config_schema())


def test_phase_3_supported_field_types_are_stable() -> None:
    assert SUPPORTED_CONFIG_FIELD_TYPES == {
        "string",
        "integer",
        "decimal",
        "boolean",
        "enum",
        "symbol",
        "symbol_list",
        "timeframe",
        "timeframe_list",
        "secret_ref",
        "object",
        "array",
    }


def test_phase_3_config_schema_rejects_unsupported_field_type() -> None:
    schema = _valid_config_schema()
    schema["sections"][0]["fields"][0]["type"] = "exchange_client"

    with pytest.raises(ConfigSchemaValidationError, match="unsupported"):
        validate_config_schema(schema)


def test_phase_3_config_schema_rejects_non_json_safe_values() -> None:
    schema = _valid_config_schema()
    schema["sections"][0]["fields"][0]["default"] = ("ETHUSDT",)

    with pytest.raises(ConfigSchemaValidationError, match="not JSON-safe"):
        validate_config_schema(schema)


def test_phase_3_json_safety_rejects_float_metadata() -> None:
    with pytest.raises(ConfigSchemaValidationError, match="must not use float"):
        assert_json_safe({"min": 0.1})


def test_phase_3_config_schema_rejects_unknown_keys() -> None:
    schema = _valid_config_schema()
    schema["sections"][0]["fields"][0]["widget"] = "custom"

    with pytest.raises(ConfigSchemaValidationError, match="unsupported keys"):
        validate_config_schema(schema)


def test_phase_3_object_fields_require_nested_fields() -> None:
    schema = _valid_config_schema()
    schema["sections"][0]["fields"].append(
        {
            "key": "risk",
            "type": "object",
            "label": "Risk",
        }
    )

    with pytest.raises(ConfigSchemaValidationError, match="fields is required"):
        validate_config_schema(schema)


def test_phase_3_array_fields_require_item_schema() -> None:
    schema = _valid_config_schema()
    schema["sections"][0]["fields"].append(
        {
            "key": "thresholds",
            "type": "array",
            "label": "Thresholds",
        }
    )

    with pytest.raises(ConfigSchemaValidationError, match="item_schema is required"):
        validate_config_schema(schema)


def test_phase_3_fixture_config_schema_validates_without_importing_adapter(tmp_path: Path) -> None:
    package_dir = tmp_path / "fixture_bot"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "adapter.py").write_text("raise RuntimeError('adapter imported')\n", encoding="utf-8")
    (package_dir / "config_schema.py").write_text(
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

    module_name = "fixture_bot.config_schema"
    spec = importlib.util.spec_from_file_location(module_name, package_dir / "config_schema.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    validate_config_schema(module.CONFIG_SCHEMA)
    assert "fixture_bot.adapter" not in sys.modules


def _valid_config_schema() -> dict[str, object]:
    return {
        "schema_version": 1,
        "sections": [
            {
                "key": "market_data",
                "label": "Market Data",
                "description": "Snapshot inputs.",
                "fields": [
                    {
                        "key": "symbols",
                        "type": "symbol_list",
                        "label": "Symbols",
                        "required": True,
                        "default": ["ETHUSDT"],
                    },
                    {
                        "key": "timeframes",
                        "type": "timeframe_list",
                        "label": "Timeframes",
                        "required": True,
                        "default": ["1h", "4h"],
                        "allowed": ["1h", "4h", "1d"],
                    },
                    {
                        "key": "mode",
                        "type": "enum",
                        "label": "Mode",
                        "allowed": ["dry_run", "notification_only", "signal_only"],
                    },
                    {
                        "key": "risk",
                        "type": "object",
                        "label": "Risk",
                        "fields": [
                            {
                                "key": "max_allocation",
                                "type": "decimal",
                                "label": "Max Allocation",
                                "min": "0",
                                "max": "1",
                                "default": "0.25",
                            },
                            {
                                "key": "enabled",
                                "type": "boolean",
                                "label": "Enabled",
                                "default": True,
                            },
                        ],
                    },
                    {
                        "key": "notes",
                        "type": "array",
                        "label": "Notes",
                        "item_schema": {
                            "type": "string",
                        },
                        "default": [],
                    },
                    {
                        "key": "api_secret",
                        "type": "secret_ref",
                        "label": "API Secret",
                    },
                ],
            }
        ],
    }
