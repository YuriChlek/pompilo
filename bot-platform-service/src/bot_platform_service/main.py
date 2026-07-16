from __future__ import annotations

import sys

from bot_platform_service.cli import run_cli


def main(argv: list[str] | None = None) -> int:
    """Run the Bot Platform CLI."""
    return run_cli([] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
