from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from market_data_service import main as cli_main
from market_data_service.application.services.backfill_command_service import BackfillCommandResult
from market_data_service.application.services.gap_scan_service import GapScanResult
from market_data_service.application.services.outbox_replay_service import OutboxReplayResult
from market_data_service.application.services.outbox_publisher_service import OutboxPublishBatchResult
from market_data_service.config.settings import HttpSettings
from market_data_service.application.services.symbol_registry_sync_service import SymbolRegistrySyncResult
from market_data_service.domain.scheduler_models import SchedulerTickResult
from market_data_service.runtime.maintenance import SyncNextJobResult


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
        for provider in ("auto", "binance", "bybit"):
            with self.subTest(provider=provider):
                exit_code = cli_main.main(
                    [
                        "candles:get",
                        "--period",
                        "1d",
                        "--provider",
                        provider,
                    ]
                )
                self.assertEqual(exit_code, cli_main.EXIT_UNSUPPORTED)

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
        self.assertEqual(exit_code, cli_main.EXIT_UNSUPPORTED)

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

    def test_backfill_rejects_invalid_timeframe(self) -> None:
        exit_code = cli_main.main(
            [
                "backfill",
                "--symbol",
                "ETHUSDT",
                "--timeframe",
                "15m",
                "--from",
                "2026-07-14T00:00:00Z",
                "--to",
                "2026-07-14T02:00:00Z",
            ]
        )
        self.assertEqual(exit_code, cli_main.EXIT_ERROR)

    def test_gaps_scan_rejects_invalid_timeframe(self) -> None:
        exit_code = cli_main.main(
            [
                "gaps:scan",
                "--timeframe",
                "15m",
                "--from",
                "2026-07-14T00:00:00Z",
                "--to",
                "2026-07-14T03:00:00Z",
            ]
        )
        self.assertEqual(exit_code, cli_main.EXIT_ERROR)

    def test_unimplemented_runtime_commands_return_explicit_exit_code(self) -> None:
        for command in ("scheduler", "outbox-publisher", "collect", "db:revision"):
            with self.subTest(command=command):
                self.assertEqual(cli_main.main([command]), cli_main.EXIT_UNSUPPORTED)

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

    def test_symbols_sync_runs_maintenance_command(self) -> None:
        async def run_sync():
            return SymbolRegistrySyncResult(market_symbols=7, provider_symbols=7)

        with patch.object(cli_main, "run_symbols_sync", side_effect=run_sync) as sync:
            exit_code = cli_main.main(["symbols:sync"])

        self.assertEqual(exit_code, cli_main.EXIT_SUCCESS)
        sync.assert_called_once_with()

    def test_symbols_sync_returns_error_on_configuration_failure(self) -> None:
        async def run_sync():
            raise ValueError("database host is not configured")

        with patch.object(cli_main, "run_symbols_sync", side_effect=run_sync):
            exit_code = cli_main.main(["symbols:sync"])

        self.assertEqual(exit_code, cli_main.EXIT_ERROR)

    def test_scheduler_run_once_runs_maintenance_command(self) -> None:
        async def run_once():
            return SchedulerTickResult(created_count=1, skipped_count=0)

        with patch.object(cli_main, "run_scheduler_once", side_effect=run_once) as scheduler:
            exit_code = cli_main.main(["scheduler:run-once"])

        self.assertEqual(exit_code, cli_main.EXIT_SUCCESS)
        scheduler.assert_called_once_with()

    def test_sync_run_next_runs_one_pending_sync_job(self) -> None:
        async def run_next():
            return SyncNextJobResult(job_found=False)

        with patch.object(cli_main, "run_sync_next_job", side_effect=run_next) as sync_next:
            exit_code = cli_main.main(["sync:run-next"])

        self.assertEqual(exit_code, cli_main.EXIT_SUCCESS)
        sync_next.assert_called_once_with()

    def test_outbox_publish_once_runs_one_outbox_batch(self) -> None:
        async def publish_once():
            return OutboxPublishBatchResult(
                fetched_count=1,
                published_count=1,
                retry_count=0,
                failed_count=0,
                publisher_lag_seconds=0.0,
            )

        with patch.object(cli_main, "run_outbox_publish_once", side_effect=publish_once) as publish:
            exit_code = cli_main.main(["outbox:publish-once"])

        self.assertEqual(exit_code, cli_main.EXIT_SUCCESS)
        publish.assert_called_once_with()

    def test_backfill_runs_maintenance_command_with_required_arguments(self) -> None:
        async def run_job(command):
            self.assertEqual(command.provider_symbol, "ETHUSDT")
            self.assertEqual(command.timeframe, "1h")
            self.assertEqual(command.batch_size_candles, 25)
            self.assertEqual(command.max_concurrency, 1)
            return BackfillCommandResult(
                requested_count=2,
                skipped_duplicate_count=0,
                chunk_count=2,
                batch_size_candles=25,
                max_concurrency=1,
            )

        with patch.object(cli_main, "run_backfill", side_effect=run_job) as backfill:
            exit_code = cli_main.main(
                [
                    "backfill",
                    "--symbol",
                    "ETHUSDT",
                    "--timeframe",
                    "1h",
                    "--from",
                    "2026-07-14T00:00:00Z",
                    "--to",
                    "2026-07-14T02:00:00Z",
                    "--batch-size",
                    "25",
                ]
            )

        self.assertEqual(exit_code, cli_main.EXIT_SUCCESS)
        backfill.assert_called_once()

    def test_backfill_requires_range_arguments(self) -> None:
        exit_code = cli_main.main(["backfill"])

        self.assertEqual(exit_code, cli_main.EXIT_ERROR)

    def test_gaps_scan_runs_read_only_by_default(self) -> None:
        async def run_scan(command):
            self.assertEqual(command.provider_symbols, ("ETHUSDT",))
            self.assertEqual(command.timeframes, ("1h",))
            self.assertFalse(command.create_backfill)
            return GapScanResult(reports=(), create_backfill=False)

        with patch.object(cli_main, "run_gaps_scan", side_effect=run_scan) as scan:
            exit_code = cli_main.main(
                [
                    "gaps:scan",
                    "--symbol",
                    "ETHUSDT",
                    "--timeframe",
                    "1h",
                    "--from",
                    "2026-07-14T00:00:00Z",
                    "--to",
                    "2026-07-14T03:00:00Z",
                ]
            )

        self.assertEqual(exit_code, cli_main.EXIT_SUCCESS)
        scan.assert_called_once()

    def test_gaps_scan_create_backfill_requires_explicit_flag(self) -> None:
        async def run_scan(command):
            self.assertTrue(command.create_backfill)
            return GapScanResult(reports=(), create_backfill=True)

        with patch.object(cli_main, "run_gaps_scan", side_effect=run_scan):
            exit_code = cli_main.main(
                [
                    "gaps:scan",
                    "--from",
                    "2026-07-14T00:00:00Z",
                    "--to",
                    "2026-07-14T03:00:00Z",
                    "--create-backfill",
                ]
            )

        self.assertEqual(exit_code, cli_main.EXIT_SUCCESS)

    def test_outbox_replay_runs_recovery_command(self) -> None:
        async def replay(command):
            self.assertEqual(command.from_id, "event-1")
            self.assertEqual(command.to_id, "event-9")
            self.assertTrue(command.dry_run)
            return OutboxReplayResult(
                matched_count=3,
                publishable_count=2,
                published_count=0,
                skipped_published_count=1,
                dry_run=True,
            )

        with patch.object(cli_main, "run_outbox_replay", side_effect=replay) as run:
            exit_code = cli_main.main(
                [
                    "outbox:replay",
                    "--from-id",
                    "event-1",
                    "--to-id",
                    "event-9",
                    "--dry-run",
                ]
            )

        self.assertEqual(exit_code, cli_main.EXIT_SUCCESS)
        run.assert_called_once()

    def test_outbox_replay_returns_error_on_redis_failure(self) -> None:
        async def replay(command):
            raise RuntimeError("redis unavailable")

        with patch.object(cli_main, "run_outbox_replay", side_effect=replay):
            exit_code = cli_main.main(["outbox:replay"])

        self.assertEqual(exit_code, cli_main.EXIT_ERROR)
