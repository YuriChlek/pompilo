__all__ = [
    "TrendResult",
    "get_of_data",
    "get_of_data_from_dataframe",
    "get_trend_data",
    "get_trend_history",
]


def __getattr__(name):
    """Lazily import indicator APIs to avoid importing optional heavy deps during discovery."""
    if name == "TrendResult":
        from .models import TrendResult

        return TrendResult
    if name in {
        "get_of_data",
        "get_of_data_from_dataframe",
        "get_trend_data",
        "get_trend_history",
    }:
        from .api import get_of_data, get_of_data_from_dataframe, get_trend_data, get_trend_history

        mapping = {
            "get_of_data": get_of_data,
            "get_of_data_from_dataframe": get_of_data_from_dataframe,
            "get_trend_data": get_trend_data,
            "get_trend_history": get_trend_history,
        }
        return mapping[name]
    raise AttributeError(name)
