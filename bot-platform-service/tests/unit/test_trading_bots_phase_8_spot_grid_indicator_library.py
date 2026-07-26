from __future__ import annotations

import ast
from decimal import Decimal
from pathlib import Path

import pytest

from bot_platform_service.trading_bots.spot_grid.domain import (
    INDICATOR_LIBRARY_DECISION,
    REJECTED_INDICATOR_LIBRARY_CANDIDATES,
    SELECTED_INDICATOR_LIBRARY,
    SELECTED_INDICATOR_LIBRARY_DEPENDENCY,
    normalize_indicator_decimal,
)


SERVICE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SERVICE_ROOT.parent
PYPROJECT_PATH = SERVICE_ROOT / "pyproject.toml"
INDICATOR_DOC_PATH = SERVICE_ROOT / "docs" / "spot_grid_indicator_library.md"
SPOT_GRID_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "trading_bots" / "spot_grid"


def test_phase_8_indicator_library_choice_is_documented() -> None:
    doc = INDICATOR_DOC_PATH.read_text(encoding="utf-8")

    assert SELECTED_INDICATOR_LIBRARY == "stock-indicators"
    assert SELECTED_INDICATOR_LIBRARY_DEPENDENCY == "stock-indicators>=1.3"
    assert INDICATOR_LIBRARY_DECISION.selected == SELECTED_INDICATOR_LIBRARY
    assert INDICATOR_LIBRARY_DECISION.dependency == SELECTED_INDICATOR_LIBRARY_DEPENDENCY
    assert REJECTED_INDICATOR_LIBRARY_CANDIDATES == ("pandas", "pandas-ta", "ta")
    assert "Use `stock-indicators` as the selected indicator library" in doc
    assert "Do not use package-local hand-rolled implementations" in doc
    assert "Normalize every library output into `Decimal`" in doc


def test_phase_8_selected_indicator_dependency_is_declared_without_extra_dataframe_dependency() -> None:
    pyproject = PYPROJECT_PATH.read_text(encoding="utf-8")

    assert '"stock-indicators>=1.3"' in pyproject
    assert '"pandas' not in pyproject
    assert '"pandas-ta' not in pyproject
    assert '"ta' not in pyproject


def test_phase_8_lightweight_files_do_not_import_heavy_indicator_libraries() -> None:
    heavy_terms = (
        "stock_indicators",
        "pandas",
        "pandas_ta",
        "numpy",
        "from ta",
        "import ta",
    )
    lightweight_files = (
        SPOT_GRID_ROOT / "manifest.py",
        SPOT_GRID_ROOT / "config_schema.py",
        SPOT_GRID_ROOT / "bot_config.py",
    )

    violations: list[str] = []
    for path in lightweight_files:
        text = path.read_text(encoding="utf-8")
        found = [term for term in heavy_terms if term in text]
        if found:
            violations.append(f"{path.relative_to(REPO_ROOT)}: {', '.join(found)}")

    assert violations == []


def test_phase_8_indicator_boundary_normalizes_values_to_decimal() -> None:
    assert normalize_indicator_decimal(Decimal("1.25"), field_name="ema20") == Decimal("1.25")
    assert normalize_indicator_decimal("2.50", field_name="atr14") == Decimal("2.50")
    assert normalize_indicator_decimal(3, field_name="rsi14") == Decimal("3")
    assert normalize_indicator_decimal(None, field_name="ema200") is None


def test_phase_8_indicator_boundary_rejects_float_unless_explicitly_external() -> None:
    with pytest.raises(TypeError, match="ema20 must not be float"):
        normalize_indicator_decimal(1.25, field_name="ema20")

    assert normalize_indicator_decimal(1.25, field_name="ema20", allow_external_float=True) == Decimal("1.25")


def test_phase_8_indicator_policy_module_does_not_import_selected_heavy_library() -> None:
    imported_modules = _direct_imports(SPOT_GRID_ROOT / "domain" / "indicator_policy.py")

    assert "stock_indicators" not in imported_modules
    assert "pandas" not in imported_modules
    assert "pandas_ta" not in imported_modules
    assert "numpy" not in imported_modules


def _direct_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports
