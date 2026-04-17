from typing import Any, Dict, Optional

from trading.application.bootstrap import build_live_trading_cycle


async def run_bot(symbol: str, is_test: bool) -> Optional[Dict[str, Any]]:
    """Backward-compatible entrypoint for a single-symbol trading cycle."""
    return await build_live_trading_cycle().run(symbol, is_test)
