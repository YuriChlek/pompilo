from __future__ import annotations

import ast
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = SERVICE_ROOT / "src" / "bot_platform_service"
SIGNALS_ROOT = SRC_ROOT / "infrastructure" / "signals"
FORBIDDEN_TEXT = (
    "place_order",
    "cancel_order",
    "create_order",
    "create_market_buy_order",
    "create_market_sell_order",
    "BybitSpot",
    "BybitSpotExecutor",
    "BinanceMarketDataSynchronizer",
    "HTTP(",
)


def test_signal_publisher_does_not_reference_exchange_or_private_clients() -> None:
    violations: list[str] = []
    for path in sorted(SIGNALS_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        found = [pattern for pattern in FORBIDDEN_TEXT if pattern in text]
        if found:
            violations.append(f"{path.relative_to(SERVICE_ROOT)}: {', '.join(found)}")

    assert violations == []


def test_signal_publisher_does_not_import_bot_adapters_or_legacy_bots() -> None:
    violations: list[str] = []
    forbidden_roots = {
        "spot_grid_bot",
        "spot-greenwich-bot",
        "spot_greenwich_bot",
        "bot_platform_service.infrastructure.bot_modules",
    }
    for path in sorted(SIGNALS_ROOT.rglob("*.py")):
        imports = _direct_imports(path)
        forbidden = {
            imported
            for imported in imports
            for forbidden_root in forbidden_roots
            if imported == forbidden_root or imported.startswith(f"{forbidden_root}.")
        }
        if forbidden:
            violations.append(f"{path.relative_to(SERVICE_ROOT)}: {', '.join(sorted(forbidden))}")

    assert violations == []


def test_bot_platform_source_does_not_contain_exchange_order_creation_calls() -> None:
    violations: list[str] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        found = [
            pattern
            for pattern in ("place_order(", "cancel_order(", "create_market_buy_order(", "create_market_sell_order(")
            if pattern in text
        ]
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
