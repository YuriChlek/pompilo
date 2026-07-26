from __future__ import annotations

from datetime import UTC, datetime, timedelta
import unittest

from market_data_service.infrastructure.providers.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitBreakerState,
)


class CircuitBreakerTests(unittest.TestCase):
    def test_breaker_opens_after_threshold_and_half_opens_after_timeout(self) -> None:
        now = datetime(2026, 7, 14, 10, tzinfo=UTC)
        breaker = CircuitBreaker(
            CircuitBreakerConfig(failure_threshold=2, recovery_timeout=timedelta(seconds=30)),
            now_provider=lambda: now,
        )

        breaker.record_failure()
        self.assertEqual(breaker.state, CircuitBreakerState.CLOSED)
        breaker.record_failure()
        self.assertEqual(breaker.state, CircuitBreakerState.OPEN)

        with self.assertRaises(CircuitBreakerOpenError):
            breaker.before_call()

        now = datetime(2026, 7, 14, 10, 0, 31, tzinfo=UTC)
        breaker.before_call()
        self.assertEqual(breaker.state, CircuitBreakerState.HALF_OPEN)

        breaker.record_success()
        self.assertEqual(breaker.state, CircuitBreakerState.CLOSED)
        self.assertEqual(breaker.failure_count, 0)

    def test_breaker_rejects_invalid_config(self) -> None:
        with self.assertRaises(ValueError):
            CircuitBreakerConfig(failure_threshold=0)
