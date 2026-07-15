from __future__ import annotations

import ast
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = SERVICE_ROOT / "src" / "bot_platform_service" / "trading_bots" / "spot_grid" / "adapter.py"
LEGACY_PACKAGE_PATH = SERVICE_ROOT / "src" / "bot_platform_service" / "infrastructure" / "bot_modules"
FORBIDDEN_TEXT = (
    "BybitSpot",
    "BybitSpotExecutionService",
    "BybitSpotExchange",
    "BinanceMarketDataSynchronizer",
    "run_binance_candle_sync",
    "ensure_candle_tables",
    "DatabaseMarketDataProvider",
    "PostgresStateStore",
    "place_order",
    "cancel_order",
    "HTTP(",
)


def test_spot_grid_platform_adapter_does_not_reference_private_exchange_or_direct_sync() -> None:
    text = ADAPTER_PATH.read_text(encoding="utf-8")
    found = [pattern for pattern in FORBIDDEN_TEXT if pattern in text]

    assert found == []


def test_spot_grid_platform_adapter_does_not_import_legacy_root_bot_packages() -> None:
    imports = _direct_imports(ADAPTER_PATH)
    forbidden = {
        imported
        for imported in imports
        if imported.startswith("spot_grid_bot")
        or imported.startswith("spot-greenwich-bot")
        or imported.startswith("spot_greenwich_bot")
    }

    assert forbidden == set()


def test_spot_grid_legacy_infrastructure_adapter_package_is_removed() -> None:
    assert not LEGACY_PACKAGE_PATH.exists()


def _direct_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports
