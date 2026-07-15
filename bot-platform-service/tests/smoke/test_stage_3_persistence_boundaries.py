from __future__ import annotations

import ast
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[2]
PERSISTENCE_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "persistence"
FORBIDDEN_IMPORT_ROOTS = {
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


def test_persistence_does_not_import_legacy_bots() -> None:
    violations: list[str] = []
    for path in sorted(PERSISTENCE_ROOT.rglob("*.py")):
        forbidden = _direct_imports(path) & FORBIDDEN_IMPORT_ROOTS
        if forbidden:
            violations.append(f"{path.relative_to(SERVICE_ROOT)}: {', '.join(sorted(forbidden))}")

    assert violations == []


def test_alembic_stage_3_files_exist() -> None:
    required = [
        SERVICE_ROOT / "alembic.ini",
        SERVICE_ROOT / "alembic" / "env.py",
        SERVICE_ROOT / "alembic" / "versions" / "20260714_0001_create_bot_platform_schema.py",
        SERVICE_ROOT / "alembic" / "versions" / "20260715_0002_extend_bot_module_metadata.py",
    ]

    missing = [str(path.relative_to(SERVICE_ROOT)) for path in required if not path.exists()]
    assert missing == []
