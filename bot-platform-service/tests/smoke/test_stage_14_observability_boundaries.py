from __future__ import annotations

import ast
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[2]
OBSERVABILITY_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "observability"
FORBIDDEN_IMPORT_ROOTS = {
    "asyncpg",
    "sqlalchemy",
    "spot_grid_bot",
    "spot-greenwich-bot",
    "spot_greenwich_bot",
    "bot_platform_service.infrastructure.bot_modules",
    "bot_platform_service.persistence",
}


def test_observability_layer_has_no_persistence_or_legacy_imports() -> None:
    violations: list[str] = []
    for path in sorted(OBSERVABILITY_ROOT.rglob("*.py")):
        imports = _direct_imports(path)
        forbidden = {
            imported
            for imported in imports
            for forbidden_root in FORBIDDEN_IMPORT_ROOTS
            if imported == forbidden_root or imported.startswith(f"{forbidden_root}.")
        }
        if forbidden:
            violations.append(f"{path.relative_to(SERVICE_ROOT)}: {', '.join(sorted(forbidden))}")

    assert violations == []


def test_observability_layer_does_not_reference_exchange_order_creation() -> None:
    violations: list[str] = []
    forbidden_text = ("place_order", "cancel_order", "create_market_buy_order", "create_market_sell_order", "BybitSpot")
    for path in sorted(OBSERVABILITY_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        found = [pattern for pattern in forbidden_text if pattern in text]
        if found:
            violations.append(f"{path.relative_to(SERVICE_ROOT)}: {', '.join(found)}")

    assert violations == []


def _direct_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports
