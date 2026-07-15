from __future__ import annotations

import ast
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[2]
LIFECYCLE_PATH = SERVICE_ROOT / "src" / "bot_platform_service" / "application" / "bot_instance_lifecycle_service.py"
FORBIDDEN_IMPORT_ROOTS = {
    "asyncpg",
    "sqlalchemy",
    "spot_grid_bot",
    "spot-greenwich-bot",
    "spot_greenwich_bot",
    "bot_platform_service.infrastructure",
    "bot_platform_service.persistence",
}


def test_lifecycle_service_has_no_persistence_infrastructure_or_legacy_imports() -> None:
    imports = _direct_imports(LIFECYCLE_PATH)
    forbidden = {
        imported
        for imported in imports
        for forbidden_root in FORBIDDEN_IMPORT_ROOTS
        if imported == forbidden_root or imported.startswith(f"{forbidden_root}.")
    }

    assert forbidden == set()


def test_lifecycle_service_does_not_reference_exchange_or_order_creation_terms() -> None:
    text = LIFECYCLE_PATH.read_text(encoding="utf-8")
    forbidden_text = ("place_order", "cancel_order", "Bybit", "Binance", "exchange_order")

    assert [pattern for pattern in forbidden_text if pattern in text] == []


def _direct_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports
