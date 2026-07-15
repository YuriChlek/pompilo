from __future__ import annotations

from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = SERVICE_ROOT / "src" / "bot_platform_service"


def test_stage_1_required_skeleton_files_exist() -> None:
    required_paths = [
        SERVICE_ROOT / "pyproject.toml",
        SERVICE_ROOT / "STANDARDS.md",
        PACKAGE_ROOT / "__init__.py",
        PACKAGE_ROOT / "main.py",
        SERVICE_ROOT / "tests" / "unit",
        SERVICE_ROOT / "tests" / "smoke",
    ]

    missing = [str(path.relative_to(SERVICE_ROOT)) for path in required_paths if not path.exists()]
    assert missing == []


def test_stage_1_required_package_layers_exist() -> None:
    required_packages = [
        "application",
        "config",
        "domain",
        "infrastructure",
        "observability",
        "persistence",
        "registry",
        "trading_bots",
        "workers",
    ]

    missing = []
    for package in required_packages:
        package_path = PACKAGE_ROOT / package
        init_file = package_path / "__init__.py"
        if not package_path.is_dir() or not init_file.exists():
            missing.append(package)

    assert missing == []
