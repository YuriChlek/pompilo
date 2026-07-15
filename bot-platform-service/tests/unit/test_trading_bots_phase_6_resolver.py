from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

import bot_platform_service.trading_bots as trading_bots
from bot_platform_service.application.bot_run_orchestration_service import BotModuleResolver
from bot_platform_service.domain import BotModuleMetadata, BotModuleStatus
from bot_platform_service.registry import BotModuleResolutionError, PersistedBotModuleResolver


class _FakeMetadataRepository:
    def __init__(self, metadata: BotModuleMetadata | None) -> None:
        self.metadata = metadata
        self.calls: list[str] = []

    async def get_active_module_metadata(self, module_id: str) -> BotModuleMetadata | None:
        self.calls.append(module_id)
        if self.metadata is None or self.metadata.module_id != module_id:
            return None
        return self.metadata


def test_resolver_loads_fixture_adapter_from_persisted_metadata(tmp_path: Path) -> None:
    _write_adapter_package(tmp_path, "spot_grid")
    repository = _FakeMetadataRepository(_metadata("spot_grid"))
    resolver = PersistedBotModuleResolver(repository)

    with _trading_bots_path(tmp_path):
        adapter = asyncio.run(resolver.resolve("spot_grid"))
        second_adapter = asyncio.run(resolver.resolve("spot_grid"))

    assert adapter is not None
    assert adapter.module_id == "spot_grid"
    assert type(adapter).__name__ == "SpotGridAdapter"
    assert second_adapter is not adapter
    assert repository.calls == ["spot_grid", "spot_grid"]


def test_resolver_returns_none_when_active_metadata_is_missing() -> None:
    resolver = PersistedBotModuleResolver(_FakeMetadataRepository(None))

    adapter = asyncio.run(resolver.resolve("spot_grid"))

    assert adapter is None


def test_resolver_redacts_missing_adapter_class_metadata() -> None:
    resolver = PersistedBotModuleResolver(
        _FakeMetadataRepository(
            _metadata(
                "spot_grid",
                adapter_path="bot_platform_service.trading_bots.spot_grid.adapter",
                adapter_class=None,
            )
        )
    )

    with pytest.raises(BotModuleResolutionError) as exc_info:
        asyncio.run(resolver.resolve("spot_grid"))

    message = str(exc_info.value)
    assert message == "Adapter metadata is incomplete"
    assert "spot_grid.adapter" not in message


def test_resolver_redacts_invalid_adapter_import_path() -> None:
    path = "bot_platform_service.domain.spot_grid_adapter"
    resolver = PersistedBotModuleResolver(
        _FakeMetadataRepository(_metadata("spot_grid", adapter_path=path))
    )

    with pytest.raises(BotModuleResolutionError) as exc_info:
        asyncio.run(resolver.resolve("spot_grid"))

    message = str(exc_info.value)
    assert message == "Adapter metadata is invalid"
    assert path not in message


def test_resolver_redacts_missing_adapter_class(tmp_path: Path) -> None:
    _write_adapter_package(tmp_path, "spot_grid", class_name="DifferentAdapter")
    resolver = PersistedBotModuleResolver(_FakeMetadataRepository(_metadata("spot_grid")))

    with _trading_bots_path(tmp_path), pytest.raises(BotModuleResolutionError) as exc_info:
        asyncio.run(resolver.resolve("spot_grid"))

    message = str(exc_info.value)
    assert message == "Adapter class could not be loaded"
    assert "SpotGridAdapter" not in message
    assert "bot_platform_service.trading_bots.spot_grid.adapter" not in message


def test_resolver_does_not_require_runtime_module_id_list() -> None:
    resolver_source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "bot_platform_service"
        / "registry"
        / "module_resolver.py"
    ).read_text(encoding="utf-8")

    assert "SUPPORTED_BOTS" not in resolver_source


def test_resolver_matches_current_orchestration_protocol() -> None:
    resolver = PersistedBotModuleResolver(_FakeMetadataRepository(None))

    accepted: BotModuleResolver = resolver

    assert accepted is resolver


def _metadata(
    module_id: str,
    *,
    adapter_path: str | None = None,
    adapter_class: str | None = "SpotGridAdapter",
) -> BotModuleMetadata:
    return BotModuleMetadata(
        module_id=module_id,
        display_name="Spot Grid",
        version="1.0.0",
        adapter_path=adapter_path or f"bot_platform_service.trading_bots.{module_id}.adapter",
        adapter_class=adapter_class,
        status=BotModuleStatus.ACTIVE,
        manifest={"module_id": module_id},
        config_schema_version=1,
        config_schema={"schema_version": 1, "sections": []},
    )


def _write_adapter_package(tmp_path: Path, module_id: str, *, class_name: str = "SpotGridAdapter") -> None:
    package_dir = tmp_path / module_id
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "adapter.py").write_text(
        "\n".join(
            [
                f"class {class_name}:",
                f"    module_id = '{module_id}'",
                "",
                "    async def validate_config(self, config):",
                "        return None",
                "",
                "    async def initialize(self, context):",
                "        return None",
                "",
                "    async def dry_run(self, request):",
                "        return None",
                "",
                "    async def run_once(self, request):",
                "        return None",
                "",
                "    async def start(self, request):",
                "        return None",
                "",
                "    async def stop(self, instance_id):",
                "        return None",
                "",
                "    async def health(self, instance_id):",
                "        return None",
                "",
            ]
        ),
        encoding="utf-8",
    )


@contextmanager
def _trading_bots_path(tmp_path: Path) -> Iterator[None]:
    original_path = list(trading_bots.__path__)
    trading_bots.__path__[:] = [str(tmp_path)]
    _clear_test_modules()
    try:
        yield
    finally:
        trading_bots.__path__[:] = original_path
        _clear_test_modules()


def _clear_test_modules() -> None:
    prefix = "bot_platform_service.trading_bots."
    for module_name in list(sys.modules):
        if module_name.startswith(prefix):
            del sys.modules[module_name]
