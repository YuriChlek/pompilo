from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

import websockets

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class PriceDeviationEvent:
    """Live price event that exceeded the configured ATR-based deviation threshold."""

    symbol: str
    live_price: float
    cached_price: float
    atr14: float


class BybitLivePriceMonitor:
    """WebSocket monitor for Bybit spot ticker prices between scheduled cycles."""

    def __init__(
        self,
        *,
        reference_provider,
        on_deviation,
        atr_multiplier: float = 2.0,
        cooldown_seconds: float = 60.0,
        websocket_url: str = "wss://stream.bybit.com/v5/public/spot",
    ) -> None:
        self.reference_provider = reference_provider
        self.on_deviation = on_deviation
        self.atr_multiplier = atr_multiplier
        self.cooldown_seconds = cooldown_seconds
        self.websocket_url = websocket_url
        self._last_alert_at: dict[str, float] = {}

    async def run_forever(self, symbols: list[str]) -> None:
        """Subscribe to live tickers and keep monitoring until cancelled."""
        if not symbols:
            return
        subscribe_payload = json.dumps({"op": "subscribe", "args": [f"tickers.{symbol.upper()}" for symbol in symbols]})
        while True:
            try:
                async with websockets.connect(self.websocket_url, ping_interval=20, ping_timeout=20) as websocket:
                    await websocket.send(subscribe_payload)
                    async for message in websocket:
                        await self.process_message(message)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("live_price_monitor_reconnecting error_type=%s", type(exc).__name__)
                await asyncio.sleep(2.0)

    async def process_message(self, raw_message: str) -> None:
        """Process one raw Bybit WebSocket message."""
        payload = json.loads(raw_message)
        topic = str(payload.get("topic") or "")
        if not topic.startswith("tickers."):
            return
        symbol = topic.split(".", 1)[1].upper()
        data = payload.get("data") or {}
        if isinstance(data, list):
            data = data[0] if data else {}
        last_price_raw = data.get("lastPrice") or data.get("price") or data.get("markPrice")
        if last_price_raw is None:
            return
        await self._handle_price(symbol, float(last_price_raw))

    async def _handle_price(self, symbol: str, live_price: float) -> None:
        reference = self.reference_provider(symbol)
        if reference is None or reference.atr14 <= 0 or reference.cached_price <= 0:
            return
        deviation = abs(live_price - reference.cached_price)
        threshold = reference.atr14 * self.atr_multiplier
        if deviation < threshold:
            return
        now = asyncio.get_running_loop().time()
        last_alert = self._last_alert_at.get(symbol)
        if last_alert is not None and now - last_alert < self.cooldown_seconds:
            return
        self._last_alert_at[symbol] = now
        logger.warning(
            "live_price_deviation_detected symbol=%s live_price=%s cached_price=%s atr14=%s threshold=%s",
            symbol,
            live_price,
            reference.cached_price,
            reference.atr14,
            threshold,
        )
        await self.on_deviation(
            PriceDeviationEvent(
                symbol=symbol,
                live_price=live_price,
                cached_price=reference.cached_price,
                atr14=reference.atr14,
            )
        )
