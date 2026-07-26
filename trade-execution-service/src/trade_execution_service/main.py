from __future__ import annotations

import sys

from trade_execution_service.cli import run_cli


def main(argv: list[str] | None = None) -> int:
    """Run the Trade Execution Service CLI."""
    return run_cli([] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
