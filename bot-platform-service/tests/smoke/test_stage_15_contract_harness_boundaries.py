from __future__ import annotations

import ast
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[2]
TESTING_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "testing"
FORBIDDEN_IMPORT_ROOTS = {
    "asyncpg",
    "sqlalchemy",
    "spot_grid_bot",
    "spot-greenwich-bot",
    "spot_greenwich_bot",
    "bot_platform_service.infrastructure.bot_modules",
    "bot_platform_service.persistence",
}


def test_contract_harness_has_no_persistence_or_legacy_imports() -> None:
    violations: list[str] = []
    for path in sorted(TESTING_ROOT.rglob("*.py")):
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


def test_contract_harness_checks_for_exchange_and_db_construction_terms() -> None:
    text = (TESTING_ROOT / "bot_module_contracts.py").read_text(encoding="utf-8")

    assert "place_order" in text
    assert "create_engine(" in text
    assert "BinanceMarketDataSynchronizer" in text


def _direct_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports
