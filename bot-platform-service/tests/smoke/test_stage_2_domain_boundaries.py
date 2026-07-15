from __future__ import annotations

import ast
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[2]
DOMAIN_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "domain"
FORBIDDEN_IMPORT_ROOTS = {
    "asyncpg",
    "sqlalchemy",
    "spot_grid_bot",
    "spot-greenwich-bot",
    "spot_greenwich_bot",
}


def _module_root(import_name: str) -> str:
    return import_name.split(".", 1)[0]


def _direct_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(_module_root(alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(_module_root(node.module))
    return imports


def test_domain_contracts_do_not_import_forbidden_dependencies() -> None:
    violations: list[str] = []
    for path in sorted(DOMAIN_ROOT.rglob("*.py")):
        forbidden = _direct_imports(path) & FORBIDDEN_IMPORT_ROOTS
        if forbidden:
            relative_path = path.relative_to(SERVICE_ROOT)
            violations.append(f"{relative_path}: {', '.join(sorted(forbidden))}")

    assert violations == []
