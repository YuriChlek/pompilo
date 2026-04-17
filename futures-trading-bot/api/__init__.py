__all__ = [
    "run_api",
    "fetch_and_store",
]


def __getattr__(name):
    """Lazily import API helpers to keep package import safe during test discovery."""
    if name in {"run_api", "fetch_and_store"}:
        from .binance_api import fetch_and_store, run_api

        mapping = {
            "run_api": run_api,
            "fetch_and_store": fetch_and_store,
        }
        return mapping[name]
    raise AttributeError(name)
