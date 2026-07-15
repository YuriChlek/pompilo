from __future__ import annotations

import ast
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[2]
ADMIN_METADATA_PATH = SERVICE_ROOT / "src" / "bot_platform_service" / "application" / "admin_metadata_service.py"
FORBIDDEN_IMPORTS = {
    "asyncpg",
    "sqlalchemy",
    "bot_platform_service.persistence",
    "bot_platform_service.infrastructure",
    "bot_platform_service.trading_bots",
    "spot_grid_bot",
    "spot-greenwich-bot",
    "spot_greenwich_bot",
}


def _direct_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_phase_8_admin_metadata_boundary_exists() -> None:
    assert ADMIN_METADATA_PATH.exists()


def test_phase_8_admin_metadata_boundary_does_not_import_strategy_or_persistence_code() -> None:
    imports = _direct_imports(ADMIN_METADATA_PATH)
    violations = {
        imported
        for imported in imports
        for forbidden in FORBIDDEN_IMPORTS
        if imported == forbidden or imported.startswith(f"{forbidden}.")
    }

    assert violations == set()


def test_phase_8_admin_metadata_boundary_exports_required_contracts() -> None:
    source = ADMIN_METADATA_PATH.read_text(encoding="utf-8")
    required = (
        "AdminMetadataActor",
        "AdminMetadataAccessPolicy",
        "AdminMetadataService",
        "list_modules",
        "get_module_detail",
        "get_config_schema",
        "list_active_module_metadata",
        "get_module_metadata",
    )

    assert [phrase for phrase in required if phrase not in source] == []
