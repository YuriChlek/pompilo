"""
Compatibility wrapper for the active forecast entrypoint.

Canonical active inference module:
- module_ai/forecast.py

This wrapper exposes the forecast-first path while avoiding the old
decision-returning semantics.
"""

from module_ai.forecast import build_argparser, get_return_forecast, main

__all__ = ["build_argparser", "get_return_forecast", "main"]


if __name__ == "__main__":
    main()
