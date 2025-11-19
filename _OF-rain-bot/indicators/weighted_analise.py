import numpy as np
from utils import TradeSignal

def weighted_signal(market):
    score = {"BUY": 0, "SELL": 0}

    # === CVD тренд (сильний індикатор) ===
    if market.cvd["trend"] == "bullish":
        score["BUY"] += 30
    elif market.cvd["trend"] == "bearish":
        score["SELL"] += 30

    # === Тип свічки ===
    if market.price["type"].startswith("bullish"):
        score["BUY"] += 20
    elif market.price["type"].startswith("bearish"):
        score["SELL"] += 20

    # === VPOC відносно ціни ===
    if market.vpoc_cluster["current_relation"] == "below":
        score["SELL"] += 20
    elif market.vpoc_cluster["current_relation"] == "above":
        score["BUY"] += 20

    # === Обсяг ===
    if market.volume["trend"] == "increasing":
        if market.cvd["trend"] == "bullish":
            score["BUY"] += 15
        else:
            score["SELL"] += 15

    # === RSI ===
    if market.indicators["rsi"] < 35:
        score["BUY"] += 15  # перепроданість → шанс на відскок
    elif market.indicators["rsi"] > 65:
        score["SELL"] += 15  # перекупленість → шанс на відкат

    # === Визначення сигналу ===
    total = score["BUY"] + score["SELL"]
    if total == 0:
        return {"signal": TradeSignal.HOLD, "confidence": 0, "score": score}

    if score["BUY"] > score["SELL"]:
        return {
            "signal": TradeSignal.BUY,
            "confidence": round(score["BUY"] / total * 100, 2),
            "score": score
        }
    elif score["SELL"] > score["BUY"]:
        return {
            "signal": TradeSignal.SELL,
            "confidence": round(score["SELL"] / total * 100, 2),
            "score": score
        }
    else:
        return {"signal": TradeSignal.HOLD, "confidence": 50, "score": score}
