from __future__ import annotations

from decimal import Decimal, localcontext


class FakeStockIndicatorsRuntime:
    """Test double for the stock-indicators boundary used by Spot Grid tests."""

    def build_quotes(self, candles):
        return tuple(candles)

    def ema_last(self, quotes, length: int):
        values = tuple(quote.close for quote in quotes)
        if not values:
            return None
        with localcontext() as context:
            context.prec = 34
            multiplier = Decimal(2) / Decimal(length + 1)
            current = values[0]
            for value in values[1:]:
                current = (value - current) * multiplier + current
            return +current

    def atr_last(self, quotes, length: int):
        candles = tuple(quotes)
        if not candles:
            return None
        ranges = [candles[0].high - candles[0].low]
        for previous, current in zip(candles, candles[1:]):
            ranges.append(
                max(
                    current.high - current.low,
                    abs(current.high - previous.close),
                    abs(current.low - previous.close),
                )
            )
        return self._mean(tuple(ranges[-length:]))

    def rsi_last(self, quotes, length: int):
        closes = tuple(quote.close for quote in quotes)
        if not closes:
            return None
        if len(closes) == 1:
            return Decimal("50")

        gains = [Decimal("0")]
        losses = [Decimal("0")]
        for previous, current in zip(closes, closes[1:]):
            delta = current - previous
            gains.append(max(delta, Decimal("0")))
            losses.append(max(-delta, Decimal("0")))

        avg_gain = self._mean(tuple(gains[-length:]))
        avg_loss = self._mean(tuple(losses[-length:]))
        if avg_loss <= 0 and avg_gain <= 0:
            return Decimal("50")
        if avg_loss <= 0:
            return Decimal("100")
        with localcontext() as context:
            context.prec = 34
            relative_strength = avg_gain / avg_loss
            return +(Decimal("100") - (Decimal("100") / (Decimal("1") + relative_strength)))

    def volume_sma_last(self, quotes, length: int):
        volumes = tuple(quote.volume for quote in quotes)
        return self._mean(tuple(volumes[-length:])) if volumes else None

    def realized_volatility_last(self, quotes, length: int):
        closes = tuple(quote.close for quote in quotes)
        if not closes:
            return None
        if len(closes) == 1:
            return Decimal("0")

        returns = []
        for previous, current in zip(closes, closes[1:]):
            returns.append(Decimal("0") if previous <= 0 else (current - previous) / previous)
        window = tuple(returns[-length:])
        with localcontext() as context:
            context.prec = 34
            mean = self._mean(window)
            variance = sum((value - mean) * (value - mean) for value in window) / Decimal(len(window))
            return +variance.sqrt()

    @staticmethod
    def _mean(values: tuple[Decimal, ...]) -> Decimal:
        with localcontext() as context:
            context.prec = 34
            return +(sum(values, Decimal("0")) / Decimal(len(values)))
