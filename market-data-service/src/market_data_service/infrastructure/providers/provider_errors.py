from __future__ import annotations


class ProviderAdapterError(Exception):
    """Base class for provider adapter errors."""


class RetryableProviderError(ProviderAdapterError):
    """Provider request can be retried later."""


class NonRetryableProviderError(ProviderAdapterError):
    """Provider request is invalid and should not be retried as-is."""
