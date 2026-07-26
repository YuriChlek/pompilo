from __future__ import annotations

import argparse
from collections.abc import Sequence


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Run the Trade Execution Service command line interface."""
    parser = argparse.ArgumentParser(prog="trade-execution-service")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("healthcheck", help="Check process readiness.")
    subparsers.add_parser("worker", help="Start the signal execution worker.")

    args = parser.parse_args(list(argv or ()))
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "healthcheck":
        print("trade-execution-service ready")
        return 0
    if args.command == "worker":
        print("Trade Execution worker scaffold started. Order placement is not implemented yet.")
        return 0
    parser.error(f"Unsupported command: {args.command}")
    return 2
