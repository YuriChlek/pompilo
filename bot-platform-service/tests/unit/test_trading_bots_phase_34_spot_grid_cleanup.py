from __future__ import annotations

import ast
from pathlib import Path

from bot_platform_service.trading_bots.spot_grid.config_schema import CONFIG_SCHEMA
from bot_platform_service.trading_bots.spot_grid.domain import (
    POSITION_INTENT_ENTRY_EXAMPLE,
    POSITION_INTENT_EXIT_EXAMPLE,
)


SERVICE_ROOT = Path(__file__).resolve().parents[2]
SPOT_GRID_SOURCE_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "trading_bots" / "spot_grid"
POSITION_INTENT_DOC = SERVICE_ROOT / "docs" / "spot_grid_position_intent_contract.md"


def test_phase_34_spot_grid_module_matches_target_structure() -> None:
    expected_top_level = {
        "__init__.py",
        "adapter.py",
        "application",
        "bot_config.py",
        "config_schema.py",
        "domain",
        "infrastructure",
        "manifest.py",
    }
    actual_top_level = {path.name for path in SPOT_GRID_SOURCE_ROOT.iterdir() if path.name != "__pycache__"}

    assert expected_top_level <= actual_top_level
    assert (SPOT_GRID_SOURCE_ROOT / "application" / "trading_cycle_service.py").exists()
    assert (SPOT_GRID_SOURCE_ROOT / "application" / "ports.py").exists()
    assert (SPOT_GRID_SOURCE_ROOT / "infrastructure" / "platform_snapshot_adapter.py").exists()


def test_phase_34_legacy_grid_level_rollback_path_is_removed() -> None:
    source_paths = tuple(SPOT_GRID_SOURCE_ROOT.rglob("*.py"))
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in source_paths)

    assert not (SPOT_GRID_SOURCE_ROOT / "grid_level.py").exists()
    assert "spot_grid.grid_level" not in source_text
    assert "rollback-only" not in source_text
    assert "compatibility path" not in source_text


def test_phase_34_source_boundary_covers_entire_spot_grid_package() -> None:
    forbidden_text = (
        "spot_grid_bot",
        "from domain.",
        "import domain.",
        "BybitSpot",
        "BybitSpotExecutionService",
        "BybitSpotExchange",
        "BinanceMarketDataSynchronizer",
        "run_binance_candle_sync",
        "ensure_candle_tables",
        "DatabaseMarketDataProvider",
        "PostgresStateStore",
        "ccxt",
        "pybit",
        "fetch_balance",
        "get_wallet_balance",
        "get_positions",
        "place_order",
        "cancel_order",
        "create_order",
        "HTTP(",
        "os.getenv",
        "os.environ",
        "environ[",
        ".env",
    )
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in SPOT_GRID_SOURCE_ROOT.rglob("*.py"))

    assert [term for term in forbidden_text if term in source_text] == []


def test_phase_34_spot_grid_package_has_no_legacy_or_exchange_imports() -> None:
    forbidden_prefixes = ("spot_grid_bot", "spot-greenwich-bot", "spot_greenwich_bot", "ccxt", "pybit")
    imports = {
        imported
        for path in SPOT_GRID_SOURCE_ROOT.rglob("*.py")
        for imported in _direct_imports(path)
        if imported.startswith(forbidden_prefixes)
    }

    assert imports == set()


def test_phase_34_public_config_schema_describes_current_signal_intent_flow() -> None:
    schema_text = repr(CONFIG_SCHEMA)

    assert "skeleton" not in schema_text.lower()
    assert "future grid planning" not in schema_text.lower()
    assert "Signal-only risk limits for Spot Grid target intents." in schema_text


def test_phase_34_position_intent_docs_match_current_reason_codes() -> None:
    doc = POSITION_INTENT_DOC.read_text(encoding="utf-8")

    assert "rsi_buy_passed" in doc
    assert "rsi_sell_passed" in doc
    assert "rsi_oversold" not in doc
    assert "rsi_overbought" not in doc
    assert POSITION_INTENT_ENTRY_EXAMPLE["reason_codes"] == ("range_buy", "rsi_buy_passed")
    assert POSITION_INTENT_EXIT_EXAMPLE["reason_codes"] == (
        "range_take_profit",
        "rsi_sell_passed",
        "no_loss_passed",
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
