from __future__ import annotations

import ast
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[2]
MARKET_DATA_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "infrastructure" / "market_data"
FORBIDDEN_TEXT = (
    "spot_grid_bot",
    "spot-greenwich-bot",
    "_candles_trading_data",
    "BinanceMarketDataSynchronizer",
    "run_binance",
    "backfill",
)


def _direct_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_market_data_bridge_does_not_import_legacy_bots_or_market_data_service_package() -> None:
    violations: list[str] = []
    for path in sorted(MARKET_DATA_ROOT.rglob("*.py")):
        imports = _direct_imports(path)
        forbidden = {
            imported
            for imported in imports
            if imported.startswith("spot_grid_bot")
            or imported.startswith("spot_greenwich_bot")
            or imported.startswith("market_data_service")
        }
        if forbidden:
            violations.append(f"{path.relative_to(SERVICE_ROOT)}: {', '.join(sorted(forbidden))}")

    assert violations == []


def test_market_data_bridge_does_not_reference_legacy_candle_sync_or_tables() -> None:
    violations: list[str] = []
    for path in sorted(MARKET_DATA_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        found = [pattern for pattern in FORBIDDEN_TEXT if pattern in text]
        if found:
            violations.append(f"{path.relative_to(SERVICE_ROOT)}: {', '.join(found)}")

    assert violations == []
