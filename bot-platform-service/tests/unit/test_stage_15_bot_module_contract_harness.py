from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from bot_platform_service.domain import BotInstanceConfig, BotManifest, BotMode, BotModuleStatus
from bot_platform_service.testing import BotModuleContractCase, BotModuleContractHarness, ContractViolation


FIXTURE_SOURCE = Path(__file__).resolve().parents[1] / "fixtures" / "fixture_bot_module.py"
_fixture_module_spec = importlib.util.spec_from_file_location("stage15_fixture_bot_module", FIXTURE_SOURCE)
if _fixture_module_spec is None or _fixture_module_spec.loader is None:
    raise RuntimeError("Unable to load fixture bot module")
_fixture_module = importlib.util.module_from_spec(_fixture_module_spec)
_fixture_module_spec.loader.exec_module(_fixture_module)
BadRunResultBotModule = _fixture_module.BadRunResultBotModule
FixtureBotModule = _fixture_module.FixtureBotModule


def test_fixture_bot_module_passes_full_contract_harness() -> None:
    case = BotModuleContractCase(
        manifest=_manifest(),
        adapter_path="bot_platform_service.trading_bots.fixture.adapter",
        config=_config(),
        module=FixtureBotModule(),
        source_paths=(FIXTURE_SOURCE,),
    )

    BotModuleContractHarness().assert_contract(case)


def test_contract_harness_rejects_non_structured_run_result() -> None:
    case = BotModuleContractCase(
        manifest=_manifest(),
        adapter_path="bot_platform_service.trading_bots.fixture.adapter",
        config=_config(),
        module=BadRunResultBotModule(),
        source_paths=(FIXTURE_SOURCE,),
    )

    with pytest.raises(ContractViolation, match="dry_run must return BotRunResult"):
        BotModuleContractHarness().assert_market_data_and_signal_dtos(case)


def test_contract_harness_rejects_manifest_config_mismatch() -> None:
    case = BotModuleContractCase(
        manifest=_manifest(module_id="other"),
        adapter_path="bot_platform_service.trading_bots.fixture.adapter",
        config=_config(),
        module=FixtureBotModule(),
    )

    with pytest.raises(ContractViolation, match="manifest module_id"):
        BotModuleContractHarness().assert_manifest(case)


def test_contract_harness_rejects_forbidden_source_terms(tmp_path) -> None:
    source = tmp_path / "bad_adapter.py"
    source.write_text("def run(exchange):\n    exchange.place_order({})\n", encoding="utf-8")

    with pytest.raises(ContractViolation, match="place_order"):
        BotModuleContractHarness().assert_no_forbidden_source_terms((source,))


def _manifest(module_id: str = "fixture") -> BotManifest:
    return BotManifest(
        module_id=module_id,
        display_name="Fixture Bot",
        version="1.0.0",
        supported_modes=(BotMode.DRY_RUN, BotMode.NOTIFICATION_ONLY, BotMode.SIGNAL_ONLY),
        required_timeframes=("1h",),
        required_market_data=("snapshots",),
        supports_multi_symbol=False,
        config_schema_version=1,
        status=BotModuleStatus.ACTIVE,
    )


def _config() -> BotInstanceConfig:
    return BotInstanceConfig(
        instance_id="fixture-instance",
        module_id="fixture",
        mode=BotMode.DRY_RUN,
        symbols=("ETHUSDT",),
        timeframes=("1h",),
        config_schema_version=1,
        config={"fixture": True},
    )
