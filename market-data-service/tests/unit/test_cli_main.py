from __future__ import annotations

import io
import json
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from market_data_service import main as cli_main
from market_data_service.application.services.candles_get_fetch_service import CandlesGetItemResult, CandlesGetResult
from market_data_service.application.services.symbol_operations_service import SymbolStatus, SymbolsEnableResult
from market_data_service.application.services.symbol_registry_sync_service import SymbolRegistrySyncResult
from market_data_service.config.settings import HttpSettings
from market_data_service.domain.enums import MarketDataSource
from market_data_service.application.services.candle_collection_service import CollectionResult


class CliMainTests(unittest.TestCase):
    def test_help_exits_successfully(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            cli_main.main(["--help"])

        self.assertEqual(raised.exception.code, 0)

    def test_collect_help_exits_successfully(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            cli_main.main(["collect", "--help"])

        self.assertEqual(raised.exception.code, 0)

    def test_invalid_cli_arguments_exit_with_error(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            cli_main.main(["collect", "--invalid-argument-foo-bar"])

        self.assertNotEqual(raised.exception.code, 0)

    def test_candles_get_help_exits_successfully(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            cli_main.main(["candles:get", "--help"])

        self.assertEqual(raised.exception.code, 0)

    def test_candles_get_requires_period(self) -> None:
        for args in (
            ["candles:get"],
        ):
            with self.subTest(args=args):
                with self.assertRaises(SystemExit) as raised:
                    cli_main.main(args)
                self.assertNotEqual(raised.exception.code, 0)

    def test_candles_get_accepts_valid_providers(self) -> None:
        async def mock_run(command):
            return _result(provider=command.provider, symbols=command.symbols, timeframes=command.timeframes)

        for provider in ("auto", "binance", "bybit"):
            with self.subTest(provider=provider):
                with patch("market_data_service.runtime.maintenance.run_candles_get", side_effect=mock_run) as run:
                    exit_code = cli_main.main(
                        [
                            "candles:get",
                            "--period",
                            "1d",
                            "--provider",
                            provider,
                        ]
                    )
                self.assertEqual(exit_code, cli_main.EXIT_SUCCESS)
                run.assert_called_once()

    def test_candles_get_rejects_invalid_provider(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            cli_main.main(
                [
                    "candles:get",
                    "--period",
                    "1d",
                    "--provider",
                    "invalid-provider",
                ]
            )
        self.assertNotEqual(raised.exception.code, 0)

    def test_candles_get_accepts_optional_symbols_and_timeframes(self) -> None:
        async def mock_run(command):
            return _result(provider=command.provider, symbols=command.symbols, timeframes=command.timeframes)

        with patch("market_data_service.runtime.maintenance.run_candles_get", side_effect=mock_run) as run:
            exit_code = cli_main.main(
                [
                    "candles:get",
                    "--period",
                    "100d",
                    "--symbols",
                    "BTCUSDT,ETHUSDT",
                    "--timeframes",
                    "1h,4h",
                ]
            )
        self.assertEqual(exit_code, cli_main.EXIT_SUCCESS)
        run.assert_called_once()

    def test_candles_get_prints_json_output(self) -> None:
        async def mock_run(command):
            return _result(provider=command.provider, symbols=command.symbols, timeframes=command.timeframes)

        with patch("market_data_service.runtime.maintenance.run_candles_get", side_effect=mock_run):
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = cli_main.main(
                    [
                        "candles:get",
                        "--period",
                        "1d",
                        "--symbols",
                        "BTCUSDT",
                        "--timeframes",
                        "1h",
                        "--provider",
                        "binance",
                    ]
                )

        self.assertEqual(exit_code, cli_main.EXIT_SUCCESS)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["provider"], "binance")
        self.assertEqual(payload["symbols"], ["BTCUSDT"])
        self.assertEqual(payload["timeframes"], ["1h"])
        self.assertEqual(payload["fetched_count"], 3)
        self.assertEqual(payload["inserted_count"], 2)
        self.assertEqual(payload["skipped_duplicate_count"], 1)
        self.assertEqual(payload["unresolved_symbols"], [])
        self.assertEqual(payload["items"][0]["source"], "BINANCE_SPOT")
        self.assertEqual(payload["items"][0]["fetched_count"], 3)
        self.assertEqual(payload["items"][0]["inserted_count"], 2)
        self.assertEqual(payload["items"][0]["skipped_duplicate_count"], 1)

    def test_candles_get_rejects_invalid_timeframe(self) -> None:
        exit_code = cli_main.main(
            [
                "candles:get",
                "--period",
                "1y",
                "--timeframes",
                "15m",
            ]
        )
        self.assertEqual(exit_code, cli_main.EXIT_ERROR)

    def test_candles_get_rejects_invalid_period(self) -> None:
        for period in ("", "0d", "d", "1q", "-1d"):
            with self.subTest(period=period):
                try:
                    exit_code = cli_main.main(
                        [
                            "candles:get",
                            "--period",
                            period,
                        ]
                    )
                except SystemExit as exc:
                    exit_code = exc.code
                self.assertNotEqual(exit_code, cli_main.EXIT_SUCCESS)

    def test_candles_get_period_builds_range_back_from_now(self) -> None:
        now = datetime(2026, 7, 16, 12, 30, tzinfo=UTC)
        args = cli_main._build_parser().parse_args(
            [
                "candles:get",
                "--period",
                "100d",
                "--symbols",
                "btcusdt,ethusdt",
                "--timeframes",
                "1h,4h",
                "--provider",
                "binance",
            ]
        )

        range_from, range_to, symbols, timeframes, provider = cli_main._build_candles_get_contract(
            args,
            now_provider=lambda: now,
        )

        self.assertEqual(range_from, now - timedelta(days=100))
        self.assertEqual(range_to, now)
        self.assertEqual(symbols, ("BTCUSDT", "ETHUSDT"))
        self.assertEqual(timeframes, ("1h", "4h"))
        self.assertEqual(provider, "binance")

    def test_candles_get_period_units(self) -> None:
        self.assertEqual(cli_main._parse_period_duration("12h"), timedelta(hours=12))
        self.assertEqual(cli_main._parse_period_duration("1d"), timedelta(days=1))
        self.assertEqual(cli_main._parse_period_duration("2w"), timedelta(weeks=2))
        self.assertEqual(cli_main._parse_period_duration("3mo"), timedelta(days=90))
        self.assertEqual(cli_main._parse_period_duration("1y"), timedelta(days=365))

    def test_candles_get_uses_configured_symbols_and_timeframes_when_omitted(self) -> None:
        async def mock_run(command):
            self.assertEqual(command.symbols, ("ADAUSDT", "SOLUSDT"))
            self.assertEqual(command.timeframes, ("1h", "1d"))
            return _result(provider=command.provider, symbols=command.symbols, timeframes=command.timeframes)

        env = {
            "MARKET_DATA_PROVIDER_SYMBOLS": "adausdt, solusdt",
            "MARKET_DATA_TIMEFRAMES": "1h,1d",
        }
        with patch.dict("os.environ", env, clear=False):
            with patch("market_data_service.runtime.maintenance.run_candles_get", side_effect=mock_run) as run:
                exit_code = cli_main.main(["candles:get", "--period", "1d"])

        self.assertEqual(exit_code, cli_main.EXIT_SUCCESS)
        run.assert_called_once()

    def test_symbols_status_uses_configured_symbols_and_prints_table(self) -> None:
        async def mock_run(*, symbols, timeframes):
            self.assertEqual(symbols, ("BTCUSDT", "HYPEUSDT"))
            self.assertEqual(timeframes, ("1h", "4h"))
            return (
                SymbolStatus(
                    symbol="BTCUSDT",
                    provider="BINANCE",
                    status="ready",
                    candles=True,
                    snapshots=True,
                    bot_subscribed=False,
                ),
            )

        env = {
            "MARKET_DATA_PROVIDER_SYMBOLS": "btcusdt,hype/usdt",
            "MARKET_DATA_TIMEFRAMES": "1h,4h",
        }
        with patch.dict("os.environ", env, clear=False):
            with patch.object(cli_main, "run_symbols_status", side_effect=mock_run) as run:
                with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    exit_code = cli_main.main(["symbols:status"])

        self.assertEqual(exit_code, cli_main.EXIT_SUCCESS)
        self.assertIn("BTCUSDT", stdout.getvalue())
        self.assertIn("bot_subscribed", stdout.getvalue())
        run.assert_called_once()

    def test_symbols_status_can_print_json(self) -> None:
        async def mock_run(*, symbols, timeframes):
            return (
                SymbolStatus(
                    symbol=symbols[0],
                    provider="BYBIT",
                    status="ready",
                    candles=True,
                    snapshots=True,
                    bot_subscribed=True,
                ),
            )

        with patch.object(cli_main, "run_symbols_status", side_effect=mock_run):
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = cli_main.main(["symbols:status", "--symbols", "hype/usdt", "--timeframes", "1h", "--json"])

        self.assertEqual(exit_code, cli_main.EXIT_SUCCESS)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload[0]["symbol"], "HYPEUSDT")
        self.assertEqual(payload[0]["provider"], "BYBIT")

    def test_symbols_enable_runs_bootstrap_and_optional_bot_subscription(self) -> None:
        async def mock_run(*, symbols, timeframes, bootstrap_ticks, bot_instance_id):
            self.assertEqual(symbols, ("BTCUSDT", "HYPEUSDT"))
            self.assertEqual(timeframes, ("1h",))
            self.assertEqual(bootstrap_ticks, 2)
            self.assertEqual(bot_instance_id, "instance-1")
            return SymbolsEnableResult(
                registry_sync=SymbolRegistrySyncResult(market_symbols=2, provider_symbols=2),
                bootstrap_ticks=2,
                bot_instance_updated=True,
                statuses=(
                    SymbolStatus(
                        symbol="BTCUSDT",
                        provider="BINANCE",
                        status="ready",
                        candles=True,
                        snapshots=True,
                        bot_subscribed=True,
                    ),
                ),
            )

        with patch.object(cli_main, "run_symbols_enable", side_effect=mock_run) as run:
            exit_code = cli_main.main(
                [
                    "symbols:enable",
                    "btcusdt,hype/usdt",
                    "--timeframes",
                    "1h",
                    "--bootstrap-ticks",
                    "2",
                    "--bot-instance-id",
                    "instance-1",
                ]
            )

        self.assertEqual(exit_code, cli_main.EXIT_SUCCESS)
        run.assert_called_once()

    def test_removed_runtime_commands_are_not_registered(self) -> None:
        for command in (
            "scheduler",
            "outbox-publisher",
            "db:revision",
            "scheduler:run-once",
            "sync:run-next",
            "outbox:publish-once",
            "symbols:sync",
            "backfill",
            "gaps:scan",
            "outbox:replay",
        ):
            with self.subTest(command=command):
                with self.assertRaises(SystemExit) as raised:
                    cli_main.main([command])
                self.assertNotEqual(raised.exception.code, cli_main.EXIT_SUCCESS)

    def test_serve_runs_http_server(self) -> None:
        async def run_server():
            return None

        with patch.object(cli_main, "run_http_server", side_effect=run_server) as server:
            exit_code = cli_main.main(["serve"])

        self.assertEqual(exit_code, cli_main.EXIT_SUCCESS)
        server.assert_called_once_with()

    def test_healthcheck_uses_local_readiness_endpoint(self) -> None:
        class Response:
            status_code = 200
            text = "ok"

        with patch.object(cli_main, "load_http_settings", return_value=HttpSettings(host="0.0.0.0", port=8010)):
            with patch("httpx.get", return_value=Response()) as get:
                exit_code = cli_main.main(["healthcheck"])

        self.assertEqual(exit_code, cli_main.EXIT_SUCCESS)
        get.assert_called_once_with("http://127.0.0.1:8010/health/ready", timeout=5.0)

    def test_db_check_validates_database_configuration(self) -> None:
        with patch.object(cli_main, "get_database_url", return_value="postgresql+asyncpg://user:pass@db/app"):
            exit_code = cli_main.main(["db:check"])

        self.assertEqual(exit_code, cli_main.EXIT_SUCCESS)

    def test_collect_once_runs_collect_maintenance_command(self) -> None:
        async def mock_run_once():
            return CollectionResult(scheduled_count=5, processed_count=3, failed_count=0, published_count=3)

        with patch.object(cli_main, "run_collect_once", side_effect=mock_run_once) as collect_once:
            exit_code = cli_main.main(["collect", "--once"])

        self.assertEqual(exit_code, cli_main.EXIT_SUCCESS)
        collect_once.assert_called_once_with()


def _result(
    *,
    provider: str = "binance",
    symbols: tuple[str, ...] = ("BTCUSDT",),
    timeframes: tuple[str, ...] = ("1h",),
) -> CandlesGetResult:
    return CandlesGetResult(
        from_time=datetime(2026, 7, 13, 8, tzinfo=UTC),
        to_time=datetime(2026, 7, 14, 8, tzinfo=UTC),
        symbols=symbols,
        timeframes=timeframes,
        provider=provider,
        total_fetched_count=3,
        total_inserted_count=2,
        total_skipped_duplicate_count=1,
        unresolved_symbols=(),
        items=(
            CandlesGetItemResult(
                source=MarketDataSource.BINANCE_SPOT,
                canonical_symbol=symbols[0],
                provider_symbol=symbols[0].replace("/", "").upper(),
                timeframe=timeframes[0],
                fetched_count=3,
                inserted_count=2,
                skipped_duplicate_count=1,
            ),
        ),
    )


class CliMainLongRunningCollectTests(unittest.TestCase):
    def test_collect_long_running_runs_http_server(self) -> None:
        async def run_server():
            return None

        with patch("market_data_service.runtime.http_server.run_http_server", side_effect=run_server) as server:
            exit_code = cli_main.main(["collect"])

        self.assertEqual(exit_code, cli_main.EXIT_SUCCESS)
        server.assert_called_once_with()
