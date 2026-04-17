"""
Compatibility wrapper for the active training entrypoint.

Canonical active trainer:
- module_ai/train.py

This wrapper is kept only to avoid breaking operator habits while the active
workflow transitions away from versioned filenames.
"""

from module_ai.train import build_argparser, main

__all__ = ["build_argparser", "main"]


if __name__ == "__main__":
    main()
