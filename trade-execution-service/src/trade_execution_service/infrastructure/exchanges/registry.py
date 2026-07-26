from __future__ import annotations

from trade_execution_service.domain import ExchangeExecutionAdapter
from trade_execution_service.infrastructure.exchanges.binance_adapter import BinanceExecutionAdapter
from trade_execution_service.infrastructure.exchanges.bybit_adapter import BybitExecutionAdapter
from trade_execution_service.infrastructure.exchanges.okx_adapter import OkxExecutionAdapter


def build_exchange_adapters() -> dict[str, ExchangeExecutionAdapter]:
    """Build configured exchange adapters.

    The scaffold returns placeholder adapters. Production implementations should inject
    SDK clients and secret resolution outside domain code.
    """

    return {
        "binance": BinanceExecutionAdapter(),
        "bybit": BybitExecutionAdapter(),
        "okx": OkxExecutionAdapter(),
    }
