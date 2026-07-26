from __future__ import annotations

from collections.abc import Mapping

from bot_platform_service.domain.exceptions import ConfigSchemaValidationError

SUPPORTED_CONFIG_FIELD_TYPES = frozenset(
    {
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
)

CONFIG_SCHEMA_TOP_LEVEL_KEYS = frozenset({"schema_version", "sections"})
CONFIG_SCHEMA_SECTION_KEYS = frozenset({"key", "label", "description", "fields"})
CONFIG_SCHEMA_FIELD_KEYS = frozenset(
    {
        "key",
        "type",
        "label",
        "description",
        "required",
        "default",
        "allowed",
        "min",
        "max",
        "item_schema",
        "fields",
    }
)


def validate_config_schema(schema: Mapping[str, object]) -> None:
    """Validate machine-readable bot config schema metadata.

    The schema must be lightweight JSON-safe metadata suitable for discovery and admin
    rendering. This helper intentionally does not import bot adapters or strategy code.
    """

    _validate_allowed_keys(schema, CONFIG_SCHEMA_TOP_LEVEL_KEYS, "config schema")
    _require_positive_int(schema.get("schema_version"), "schema_version")
    sections = _require_mapping_list(schema.get("sections"), "sections")
    for index, section in enumerate(sections):
        _validate_section(section, path=f"sections[{index}]")


def assert_json_safe(value: object, *, path: str = "value") -> None:
    """Raise when a value cannot be represented as deterministic JSON metadata."""

    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        raise ConfigSchemaValidationError(f"{path} must not use float; use string for decimal metadata")
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise ConfigSchemaValidationError(f"{path} object keys must be strings")
            assert_json_safe(nested_value, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_json_safe(item, path=f"{path}[{index}]")
        return
    raise ConfigSchemaValidationError(f"{path} is not JSON-safe: {type(value).__name__}")


def _validate_section(section: Mapping[str, object], *, path: str) -> None:
    _validate_allowed_keys(section, CONFIG_SCHEMA_SECTION_KEYS, path)
    _require_text(section.get("key"), f"{path}.key")
    _require_text(section.get("label"), f"{path}.label")
    if "description" in section:
        _require_text(section["description"], f"{path}.description")
    fields = _require_mapping_list(section.get("fields"), f"{path}.fields")
    for index, field in enumerate(fields):
        _validate_field(field, path=f"{path}.fields[{index}]", require_identity=True)


def _validate_field(field: Mapping[str, object], *, path: str, require_identity: bool) -> None:
    _validate_allowed_keys(field, CONFIG_SCHEMA_FIELD_KEYS, path)
    if require_identity:
        _require_text(field.get("key"), f"{path}.key")
        _require_text(field.get("label"), f"{path}.label")
    elif "key" in field:
        _require_text(field["key"], f"{path}.key")
    elif "label" in field:
        _require_text(field["label"], f"{path}.label")

    field_type = _require_text(field.get("type"), f"{path}.type")
    if field_type not in SUPPORTED_CONFIG_FIELD_TYPES:
        raise ConfigSchemaValidationError(f"{path}.type is unsupported: {field_type}")

    if "description" in field:
        _require_text(field["description"], f"{path}.description")
    if "required" in field and not isinstance(field["required"], bool):
        raise ConfigSchemaValidationError(f"{path}.required must be a boolean")

    for json_key in ("default", "allowed", "min", "max"):
        if json_key in field:
            assert_json_safe(field[json_key], path=f"{path}.{json_key}")

    if "allowed" in field:
        _require_json_safe_list(field["allowed"], f"{path}.allowed")

    if "fields" in field:
        if field_type != "object":
            raise ConfigSchemaValidationError(f"{path}.fields is only valid for object fields")
        nested_fields = _require_mapping_list(field["fields"], f"{path}.fields")
        for index, nested_field in enumerate(nested_fields):
            _validate_field(nested_field, path=f"{path}.fields[{index}]", require_identity=True)
    elif field_type == "object":
        raise ConfigSchemaValidationError(f"{path}.fields is required for object fields")

    if "item_schema" in field:
        if field_type != "array":
            raise ConfigSchemaValidationError(f"{path}.item_schema is only valid for array fields")
        item_schema = _require_mapping(field["item_schema"], f"{path}.item_schema")
        _validate_field(item_schema, path=f"{path}.item_schema", require_identity=False)
    elif field_type == "array":
        raise ConfigSchemaValidationError(f"{path}.item_schema is required for array fields")


def _validate_allowed_keys(mapping: Mapping[str, object], allowed_keys: frozenset[str], path: str) -> None:
    unknown_keys = sorted(set(mapping) - allowed_keys)
    if unknown_keys:
        raise ConfigSchemaValidationError(f"{path} has unsupported keys: {', '.join(unknown_keys)}")


def _require_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigSchemaValidationError(f"{path} must be an object")
    return value


def _require_mapping_list(value: object, path: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or not value:
        raise ConfigSchemaValidationError(f"{path} must be a non-empty array")
    mappings: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        mappings.append(_require_mapping(item, f"{path}[{index}]"))
    return mappings


def _require_json_safe_list(value: object, path: str) -> None:
    if not isinstance(value, list) or not value:
        raise ConfigSchemaValidationError(f"{path} must be a non-empty array")
    assert_json_safe(value, path=path)


def _require_text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigSchemaValidationError(f"{path} must be a non-empty string")
    return value.strip()


def _require_positive_int(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigSchemaValidationError(f"{path} must be a positive integer")
    return value


__all__ = [
    "CONFIG_SCHEMA_FIELD_KEYS",
    "CONFIG_SCHEMA_SECTION_KEYS",
    "CONFIG_SCHEMA_TOP_LEVEL_KEYS",
    "SUPPORTED_CONFIG_FIELD_TYPES",
    "assert_json_safe",
    "validate_config_schema",
]
