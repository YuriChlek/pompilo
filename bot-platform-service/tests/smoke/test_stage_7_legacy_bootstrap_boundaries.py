from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_BOOTSTRAPS = (
    REPO_ROOT / "spot_grid_bot" / "application" / "platform_bootstrap.py",
    REPO_ROOT / "spot-greenwich-bot" / "application" / "platform_bootstrap.py",
)
STANDALONE_ENTRYPOINTS = (
    REPO_ROOT / "spot_grid_bot" / "main.py",
    REPO_ROOT / "spot-greenwich-bot" / "main.py",
)
FORBIDDEN_PLATFORM_BOOTSTRAP_TEXT = (
    "BybitSpot",
    "BybitSpotExecutor",
    "BybitSpotExecutionService",
    "BinanceMarketDataSynchronizer",
    "DatabaseMarketDataProvider",
    "MultiTimeframeMarketDataProvider",
    "create_engine",
    "create_connection",
    "place_order",
    "cancel_order",
    "HTTP(",
)


def test_platform_bootstraps_do_not_import_private_exchange_or_db_constructors() -> None:
    violations: list[str] = []
    for path in LEGACY_BOOTSTRAPS:
        text = path.read_text(encoding="utf-8")
        found = [pattern for pattern in FORBIDDEN_PLATFORM_BOOTSTRAP_TEXT if pattern in text]
        if found:
            violations.append(f"{path.relative_to(REPO_ROOT)}: {', '.join(found)}")
    assert violations == []


def test_platform_bootstraps_do_not_import_concrete_legacy_infrastructure_clients() -> None:
    violations: list[str] = []
    for path in LEGACY_BOOTSTRAPS:
        imports = _direct_imports(path)
        forbidden = {
            imported
            for imported in imports
            if imported.startswith("infrastructure.bybit")
            or imported.startswith("infrastructure.binance")
            or imported.startswith("infrastructure.execution")
            or imported.startswith("infrastructure.market_data")
            or imported.startswith("infrastructure.db")
        }
        if forbidden:
            violations.append(f"{path.relative_to(REPO_ROOT)}: {', '.join(sorted(forbidden))}")
    assert violations == []


def test_standalone_entrypoints_do_not_import_platform_bootstrap() -> None:
    violations: list[str] = []
    for path in STANDALONE_ENTRYPOINTS:
        text = path.read_text(encoding="utf-8")
        if "platform_bootstrap" in text:
            violations.append(str(path.relative_to(REPO_ROOT)))
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
