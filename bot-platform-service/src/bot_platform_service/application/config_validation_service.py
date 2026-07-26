from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Protocol

from bot_platform_service.domain import BotModuleMetadata


class ConfigValidationMetadataRepository(Protocol):
    """Read boundary for persisted bot module config schemas."""

    async def get_module_metadata(self, module_id: str) -> BotModuleMetadata | None:
        """Return persisted metadata for one module id."""


@dataclass(frozen=True, slots=True)
class BotConfigValidationError:
    """Stable field-level validation error."""

    field_path: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class BotConfigValidationResult:
    """Config validation result returned to admin callers."""

    valid: bool
    errors: tuple[BotConfigValidationError, ...] = ()


class BotConfigValidationService:
    """Validate admin-provided instance config against persisted module schemas."""

    def __init__(self, *, repository: ConfigValidationMetadataRepository) -> None:
        self.repository = repository

    async def validate_config(
        self,
        *,
        module_id: str,
        config_schema_version: int,
        config: Mapping[str, object],
    ) -> BotConfigValidationResult:
        """Validate config values without importing strategy code."""

        metadata = await self.repository.get_module_metadata(module_id)
        if metadata is None:
            return _invalid(_error("module_id", "module_not_found", f"Unknown bot module: {module_id}"))
        if metadata.config_schema is None or metadata.config_schema_version is None:
            return _invalid(_error("config_schema", "schema_missing", "Module does not expose a config schema"))
        if metadata.config_schema_version != config_schema_version:
            return _invalid(
                _error(
                    "config_schema_version",
                    "schema_version_mismatch",
                    f"Expected config schema version {metadata.config_schema_version}",
                )
            )

        errors: list[BotConfigValidationError] = []
        for section in _mapping_list(metadata.config_schema.get("sections")):
            for field in _mapping_list(section.get("fields")):
                _validate_field(field, config.get(str(field.get("key"))), str(field.get("key")), errors)

        return BotConfigValidationResult(valid=not errors, errors=tuple(errors))


def _validate_field(
    field: Mapping[str, object],
    value: object,
    field_path: str,
    errors: list[BotConfigValidationError],
) -> None:
    field_type = str(field.get("type"))
    required = field.get("required") is True
    if _is_missing(value):
        if required:
            errors.append(_error(field_path, "required", "Field is required"))
        return

    if field_type in {"string", "enum", "symbol", "timeframe", "secret_ref"}:
        _validate_text(field, value, field_path, errors)
        return
    if field_type == "integer":
        _validate_integer(field, value, field_path, errors)
        return
    if field_type == "decimal":
        _validate_decimal(field, value, field_path, errors)
        return
    if field_type == "boolean":
        if not isinstance(value, bool):
            errors.append(_error(field_path, "invalid_type", "Expected boolean value"))
        return
    if field_type in {"symbol_list", "timeframe_list"}:
        _validate_text_list(field, value, field_path, errors)
        return
    if field_type == "object":
        _validate_object(field, value, field_path, errors)
        return
    if field_type == "array":
        _validate_array(field, value, field_path, errors)


def _validate_text(
    field: Mapping[str, object],
    value: object,
    field_path: str,
    errors: list[BotConfigValidationError],
) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(_error(field_path, "invalid_type", "Expected non-empty string value"))
        return
    _validate_allowed(field, value, field_path, errors)


def _validate_integer(
    field: Mapping[str, object],
    value: object,
    field_path: str,
    errors: list[BotConfigValidationError],
) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        errors.append(_error(field_path, "invalid_type", "Expected integer value"))
        return
    _validate_numeric_bounds(field, Decimal(value), field_path, errors)


def _validate_decimal(
    field: Mapping[str, object],
    value: object,
    field_path: str,
    errors: list[BotConfigValidationError],
) -> None:
    if not isinstance(value, str | int):
        errors.append(_error(field_path, "invalid_type", "Expected decimal string value"))
        return
    try:
        decimal_value = Decimal(str(value))
    except InvalidOperation:
        errors.append(_error(field_path, "invalid_decimal", "Expected valid decimal value"))
        return
    _validate_numeric_bounds(field, decimal_value, field_path, errors)


def _validate_text_list(
    field: Mapping[str, object],
    value: object,
    field_path: str,
    errors: list[BotConfigValidationError],
) -> None:
    if not isinstance(value, list):
        errors.append(_error(field_path, "invalid_type", "Expected array of strings"))
        return
    if field.get("required") is True and not value:
        errors.append(_error(field_path, "required", "Field must contain at least one value"))
        return
    for index, item in enumerate(value):
        item_path = f"{field_path}[{index}]"
        if not isinstance(item, str) or not item.strip():
            errors.append(_error(item_path, "invalid_type", "Expected non-empty string value"))
            continue
        _validate_allowed(field, item, item_path, errors)


def _validate_object(
    field: Mapping[str, object],
    value: object,
    field_path: str,
    errors: list[BotConfigValidationError],
) -> None:
    if not isinstance(value, Mapping):
        errors.append(_error(field_path, "invalid_type", "Expected object value"))
        return
    for nested_field in _mapping_list(field.get("fields")):
        nested_key = str(nested_field.get("key"))
        _validate_field(nested_field, value.get(nested_key), f"{field_path}.{nested_key}", errors)


def _validate_array(
    field: Mapping[str, object],
    value: object,
    field_path: str,
    errors: list[BotConfigValidationError],
) -> None:
    if not isinstance(value, list):
        errors.append(_error(field_path, "invalid_type", "Expected array value"))
        return
    item_schema = field.get("item_schema")
    if not isinstance(item_schema, Mapping):
        return
    for index, item in enumerate(value):
        _validate_field(item_schema, item, f"{field_path}[{index}]", errors)


def _validate_allowed(
    field: Mapping[str, object],
    value: str,
    field_path: str,
    errors: list[BotConfigValidationError],
) -> None:
    allowed = field.get("allowed")
    if isinstance(allowed, list) and value not in {str(item) for item in allowed}:
        errors.append(_error(field_path, "not_allowed", f"Value must be one of: {', '.join(str(item) for item in allowed)}"))


def _validate_numeric_bounds(
    field: Mapping[str, object],
    value: Decimal,
    field_path: str,
    errors: list[BotConfigValidationError],
) -> None:
    min_value = _decimal_or_none(field.get("min"))
    max_value = _decimal_or_none(field.get("max"))
    if min_value is not None and value < min_value:
        errors.append(_error(field_path, "min_value", f"Value must be greater than or equal to {min_value}"))
    if max_value is not None and value > max_value:
        errors.append(_error(field_path, "max_value", f"Value must be less than or equal to {max_value}"))


def _decimal_or_none(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def _mapping_list(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _is_missing(value: object) -> bool:
    return value is None or value == ""


def _invalid(*errors: BotConfigValidationError) -> BotConfigValidationResult:
    return BotConfigValidationResult(valid=False, errors=errors)


def _error(field_path: str, code: str, message: str) -> BotConfigValidationError:
    return BotConfigValidationError(field_path=field_path, code=code, message=message)
