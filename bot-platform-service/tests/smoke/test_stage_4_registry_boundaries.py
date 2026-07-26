from __future__ import annotations

import ast
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "registry"
FORBIDDEN_IMPORT_ROOTS = {
    "spot_grid_bot",
    "spot-greenwich-bot",
    "spot_greenwich_bot",
    "bot_platform_service.infrastructure",
    "bot_platform_service.persistence",
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


def test_registry_does_not_import_legacy_bots_or_concrete_adapters() -> None:
    violations: list[str] = []
    for path in sorted(REGISTRY_ROOT.rglob("*.py")):
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
