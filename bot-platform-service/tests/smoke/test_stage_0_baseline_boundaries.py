from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = REPO_ROOT / "bot-platform-service"
PACKAGE_ROOT = SERVICE_ROOT / "src" / "bot_platform_service"
ADAPTER_ROOT = PACKAGE_ROOT / "infrastructure" / "bot_modules"
BASELINE_DOC = SERVICE_ROOT / "docs" / "stage_0_baseline.md"
FORBIDDEN_DIRECT_IMPORT_ROOTS = {
    "spot_grid_bot",
    "spot-greenwich-bot",
    "spot_greenwich_bot",
}


def _module_root(import_name: str) -> str:
    return import_name.split(".", 1)[0]


def _iter_python_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _is_allowed_adapter_file(path: Path) -> bool:
    try:
        path.relative_to(ADAPTER_ROOT)
    except ValueError:
        return False
    return True


def _direct_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(_module_root(alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(_module_root(node.module))
    return imports


def test_stage_0_baseline_doc_exists() -> None:
    assert BASELINE_DOC.exists()
    content = BASELINE_DOC.read_text(encoding="utf-8")
    assert "spot_grid_bot" in content
    assert "spot-greenwich-bot" in content
    assert "CLI modes" in content
    assert "Application ports" in content
    assert "Market-data dependencies" in content
    assert "Execution and private exchange dependencies" in content
    assert "Signal-generation dependencies" in content


def test_platform_core_does_not_import_legacy_bots_directly() -> None:
    violations: list[str] = []
    for path in _iter_python_files(PACKAGE_ROOT):
        if _is_allowed_adapter_file(path):
            continue
        forbidden = _direct_imports(path) & FORBIDDEN_DIRECT_IMPORT_ROOTS
        if forbidden:
            relative_path = path.relative_to(SERVICE_ROOT)
            violations.append(f"{relative_path}: {', '.join(sorted(forbidden))}")

    assert violations == []
