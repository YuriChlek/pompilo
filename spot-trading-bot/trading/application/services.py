from __future__ import annotations

from trading.application.ports import MarketDataProvider, PositionExecutor, SignalNotifier
from trading.domain.execution import decide_spot_execution
from trading.domain.signals import generate_spot_signal


class TradingCycleService:
    def __init__(self, market_data_provider: MarketDataProvider, executor: PositionExecutor, notifier: SignalNotifier) -> None:
        self.market_data_provider = market_data_provider
        self.executor = executor
        self.notifier = notifier

    async def run(self, symbol: str, dry_run: bool = False) -> dict:
        candles_df = self.market_data_provider.get_symbol_history(symbol)
        signal = generate_spot_signal(symbol, candles_df)
        position_state = await self.executor.get_position_state(symbol)
        decision = decide_spot_execution(signal, position_state)
        result = await self.executor.execute(decision, position_state, dry_run=dry_run)
        await self.notifier.notify(signal, result)
        return {
            "signal": signal,
            "decision": decision,
            "result": result,
        }
