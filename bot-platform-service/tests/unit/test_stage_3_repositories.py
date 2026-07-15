from __future__ import annotations

import asyncio
from decimal import Decimal

from bot_platform_service.domain import (
    BotHealthStatus,
    BotInstanceConfig,
    BotManifest,
    BotMode,
    BotModuleStatus,
    BotPermission,
    BotSignal,
    BotSignalSide,
    BotSignalType,
    BotTriggerType,
)
from bot_platform_service.domain.enums import BotSignalPublishStatus
from bot_platform_service.persistence.repositories import (
    BotAuditEventRepository,
    BotInstanceRepository,
    BotModuleRepository,
    BotRunRepository,
    BotSignalRepository,
)


class _FakeMappings:
    def __init__(self, row: dict[str, object] | None, rows: list[dict[str, object]] | None = None) -> None:
        self.row = row
        self.rows = rows if rows is not None else ([] if row is None else [row])

    def first(self) -> dict[str, object] | None:
        return self.row

    def all(self) -> list[dict[str, object]]:
        return self.rows


class _FakeResult:
    def __init__(
        self,
        rowcount: int,
        scalar_value: object | None = None,
        mapping_row: dict[str, object] | None = None,
        mapping_rows: list[dict[str, object]] | None = None,
    ) -> None:
        self.rowcount = rowcount
        self._scalar_value = scalar_value
        self._mapping_row = mapping_row
        self._mapping_rows = mapping_rows

    def scalar_one(self) -> object:
        if self._scalar_value is None:
            raise AssertionError("No scalar value configured")
        return self._scalar_value

    def mappings(self) -> _FakeMappings:
        return _FakeMappings(self._mapping_row, self._mapping_rows)


class _FakeConnection:
    def __init__(
        self,
        rowcounts: list[int] | None = None,
        scalar_value: object | None = None,
        mapping_row: dict[str, object] | None = None,
        mapping_rows: list[dict[str, object]] | None = None,
    ) -> None:
        self.rowcounts = list(rowcounts or [1])
        self.scalar_value = scalar_value
        self.mapping_row = mapping_row
        self.mapping_rows = mapping_rows
        self.statements: list[object] = []

    async def execute(self, statement):
        self.statements.append(statement)
        rowcount = self.rowcounts.pop(0) if self.rowcounts else 1
        scalar_value = self.scalar_value if self.scalar_value is not None and len(self.statements) > 1 else None
        return _FakeResult(rowcount, scalar_value, self.mapping_row, self.mapping_rows)


def test_module_registration_is_duplicate_safe() -> None:
    connection = _FakeConnection([1])
    repository = BotModuleRepository(connection)

    changed = asyncio.run(
        repository.register_module(_manifest(), adapter_path="bot_platform_service.infrastructure.bot_modules.spot_grid_adapter")
    )

    assert changed is True
    assert len(connection.statements) == 1


def test_module_registration_accepts_extended_metadata() -> None:
    connection = _FakeConnection([1])
    repository = BotModuleRepository(connection)

    changed = asyncio.run(
        repository.register_module(
            _manifest(),
            adapter_path="bot_platform_service.trading_bots.spot_grid.adapter",
            adapter_class="SpotGridAdapter",
            config_schema=_config_schema(),
        )
    )

    assert changed is True
    assert len(connection.statements) == 1


def test_module_repository_loads_active_metadata_for_runtime_resolution() -> None:
    connection = _FakeConnection(
        mapping_row={
            "module_id": "spot_grid",
            "display_name": "Spot Grid",
            "version": "1.0.0",
            "adapter_path": "bot_platform_service.trading_bots.spot_grid.adapter",
            "adapter_class": "SpotGridAdapter",
            "status": BotModuleStatus.ACTIVE.value,
            "manifest_json": {"module_id": "spot_grid"},
            "config_schema_version": 1,
            "config_schema_json": _config_schema(),
        }
    )
    repository = BotModuleRepository(connection)

    metadata = asyncio.run(repository.get_active_module_metadata("spot_grid"))

    assert metadata is not None
    assert metadata.module_id == "spot_grid"
    assert metadata.adapter_path == "bot_platform_service.trading_bots.spot_grid.adapter"
    assert metadata.adapter_class == "SpotGridAdapter"
    assert metadata.config_schema == _config_schema()
    assert len(connection.statements) == 1


def test_module_repository_returns_none_when_active_metadata_is_missing() -> None:
    connection = _FakeConnection(mapping_row=None)
    repository = BotModuleRepository(connection)

    metadata = asyncio.run(repository.get_active_module_metadata("spot_grid"))

    assert metadata is None
    assert len(connection.statements) == 1


def test_module_repository_lists_active_metadata_for_admin_boundary() -> None:
    connection = _FakeConnection(
        mapping_rows=[
            _module_metadata_row("spot_grid", "Spot Grid"),
            _module_metadata_row("spot_greenwich", "Spot Greenwich"),
        ]
    )
    repository = BotModuleRepository(connection)

    modules = asyncio.run(repository.list_active_module_metadata())

    assert [module.module_id for module in modules] == ["spot_grid", "spot_greenwich"]
    assert modules[0].config_schema == _config_schema()
    assert len(connection.statements) == 1


def test_module_repository_loads_any_persisted_metadata_for_admin_detail() -> None:
    connection = _FakeConnection(mapping_row=_module_metadata_row("spot_grid", "Spot Grid"))
    repository = BotModuleRepository(connection)

    metadata = asyncio.run(repository.get_module_metadata("spot_grid"))

    assert metadata is not None
    assert metadata.module_id == "spot_grid"
    assert metadata.display_name == "Spot Grid"
    assert metadata.config_schema_version == 1
    assert len(connection.statements) == 1


def test_instance_repository_creates_instance_and_config() -> None:
    connection = _FakeConnection([1, 1, 1])
    repository = BotInstanceRepository(connection)
    config = BotInstanceConfig(
        instance_id="instance-1",
        module_id="spot_grid_bot",
        mode=BotMode.DRY_RUN,
        symbols=("ETHUSDT",),
        timeframes=("1h", "4h"),
        config_schema_version=1,
    )

    created = asyncio.run(repository.create_instance(config))
    config_added = asyncio.run(
        repository.add_config(
            config_id="config-1",
            instance_id=config.instance_id,
            config_schema_version=1,
            config_json={"risk": "low"},
            config_hash="hash-1",
            actor_type="system",
            actor_id="stage-3",
        )
    )

    assert created is True
    assert config_added is True
    assert len(connection.statements) == 3


def test_instance_repository_upserts_state_permission_and_secret_ref() -> None:
    connection = _FakeConnection([1, 1, 1])
    repository = BotInstanceRepository(connection)

    state_saved = asyncio.run(
        repository.upsert_runtime_state(
            state_id="state-1",
            instance_id="instance-1",
            namespace="runtime",
            state_key="ETHUSDT",
            state_json={"status": "ok"},
            state_hash="state-hash",
        )
    )
    permission_saved = asyncio.run(
        repository.upsert_permission(
            permission_id="permission-1",
            instance_id="instance-1",
            permission=BotPermission.PUBLISH_SIGNALS,
            enabled=True,
        )
    )
    secret_saved = asyncio.run(
        repository.upsert_secret_ref(
            secret_ref_id="secret-1",
            instance_id="instance-1",
            secret_name="telegram",
            provider="vault",
            provider_ref="secret/path",
            status="ACTIVE",
        )
    )

    assert (state_saved, permission_saved, secret_saved) == (True, True, True)
    assert len(connection.statements) == 3


def test_run_repository_records_run_event_and_health() -> None:
    connection = _FakeConnection([1, 1, 1])
    repository = BotRunRepository(connection)

    run_created = asyncio.run(
        repository.create_run(
            run_id="run-1",
            instance_id="instance-1",
            module_id="spot_grid_bot",
            trigger_type=BotTriggerType.EVENT,
            idempotency_key="instance-1|snapshot-1",
        )
    )
    event_created = asyncio.run(
        repository.append_run_event(
            event_id="event-1",
            run_id="run-1",
            instance_id="instance-1",
            module_id="spot_grid_bot",
            event_type="STARTED",
            payload_json={},
        )
    )
    health_created = asyncio.run(
        repository.record_health_check(
            health_check_id="health-1",
            instance_id="instance-1",
            module_id="spot_grid_bot",
            status=BotHealthStatus.HEALTHY,
            details_json={},
        )
    )

    assert (run_created, event_created, health_created) == (True, True, True)
    assert len(connection.statements) == 3


def test_signal_publish_returns_inserted_signal_id() -> None:
    connection = _FakeConnection([1])
    repository = BotSignalRepository(connection)

    signal_id = asyncio.run(repository.publish_signal(signal_id="signal-1", run_id="run-1", signal=_signal()))

    assert signal_id == "signal-1"
    assert len(connection.statements) == 1


def test_signal_publish_duplicate_returns_existing_signal_id() -> None:
    connection = _FakeConnection([0, 1], scalar_value="existing-signal")
    repository = BotSignalRepository(connection)

    signal_id = asyncio.run(
        repository.publish_signal(
            signal_id="signal-1",
            run_id="run-1",
            signal=_signal(),
            status=BotSignalPublishStatus.PUBLISHED,
        )
    )

    assert signal_id == "existing-signal"
    assert len(connection.statements) == 2


def test_audit_event_repository_is_duplicate_safe() -> None:
    connection = _FakeConnection([1])
    repository = BotAuditEventRepository(connection)

    created = asyncio.run(
        repository.append_audit_event(
            event_id="audit-1",
            event_type="CONFIG_CHANGED",
            actor_type="system",
            actor_id="stage-3",
            payload_json={"redacted": True},
        )
    )

    assert created is True
    assert len(connection.statements) == 1


def _manifest() -> BotManifest:
    return BotManifest(
        module_id="spot_grid_bot",
        display_name="Spot Grid Bot",
        version="1.0.0",
        supported_modes=(BotMode.DRY_RUN, BotMode.SIGNAL_ONLY),
        required_timeframes=("1h", "4h"),
        required_market_data=("candles", "snapshots"),
        supports_multi_symbol=True,
        config_schema_version=1,
        status=BotModuleStatus.ACTIVE,
    )


def _config_schema() -> dict[str, object]:
    return {
        "schema_version": 1,
        "sections": [
            {
                "key": "market_data",
                "label": "Market Data",
                "fields": [
                    {
                        "key": "symbols",
                        "type": "symbol_list",
                        "label": "Symbols",
                        "default": ["ETHUSDT"],
                    }
                ],
            }
        ],
    }


def _module_metadata_row(module_id: str, display_name: str) -> dict[str, object]:
    return {
        "module_id": module_id,
        "display_name": display_name,
        "version": "1.0.0",
        "adapter_path": f"bot_platform_service.trading_bots.{module_id}.adapter",
        "adapter_class": "FixtureAdapter",
        "status": BotModuleStatus.ACTIVE.value,
        "manifest_json": {
            "module_id": module_id,
            "display_name": display_name,
            "version": "1.0.0",
            "supported_modes": ["dry_run", "signal_only"],
            "required_timeframes": ["1h"],
            "required_market_data": ["snapshots"],
            "supports_multi_symbol": True,
            "config_schema_version": 1,
            "status": BotModuleStatus.ACTIVE.value,
        },
        "config_schema_version": 1,
        "config_schema_json": _config_schema(),
    }


def _signal() -> BotSignal:
    return BotSignal.build(
        instance_id="instance-1",
        module_id="spot_grid_bot",
        symbol="ETHUSDT",
        timeframe="1h",
        snapshot_id="snapshot-1",
        signal_type=BotSignalType.ENTRY,
        side=BotSignalSide.BUY,
        confidence=Decimal("0.75"),
        reason="range_entry",
        payload_schema="spot_grid.range_entry",
        payload_schema_version=1,
        payload={"price": Decimal("100")},
    )
