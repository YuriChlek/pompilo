from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum


class CircuitBreakerState(StrEnum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerOpenError(RuntimeError):
    """Raised when provider calls are blocked by an open circuit breaker."""


@dataclass(frozen=True, slots=True)
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: timedelta = timedelta(seconds=60)

    def __post_init__(self) -> None:
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if self.recovery_timeout.total_seconds() <= 0:
            raise ValueError("recovery_timeout must be positive")


class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig | None = None, *, now_provider=None) -> None:
        self.config = config or CircuitBreakerConfig()
        self.now_provider = now_provider or (lambda: datetime.now(UTC))
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.opened_at: datetime | None = None

    def before_call(self) -> None:
        if self.state != CircuitBreakerState.OPEN:
            return
        if self.opened_at is None:
            raise CircuitBreakerOpenError("provider circuit breaker is open")
        if self.now_provider() - self.opened_at >= self.config.recovery_timeout:
            self.state = CircuitBreakerState.HALF_OPEN
            return
        raise CircuitBreakerOpenError("provider circuit breaker is open")

    def record_success(self) -> None:
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.opened_at = self.now_provider()
