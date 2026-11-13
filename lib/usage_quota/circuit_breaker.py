"""Circuit breaker for the CustomersDot usage quota API."""

import time
from enum import StrEnum
from typing import Callable

import structlog

__all__ = [
    "CircuitBreaker",
    "CircuitState",
]

log = structlog.stdlib.get_logger("usage_quota")


class CircuitState(StrEnum):
    CLOSED = "closed"  # Calls go through
    OPEN = "open"  # Calls are refused until the cooldown ends
    HALF_OPEN = "half_open"  # One probe call goes through after the cooldown


class CircuitBreaker:
    """Stops calls to an unhealthy dependency for a cooldown period.

    The breaker opens after ``failure_threshold`` consecutive failures. While open,
    ``try_acquire`` returns False until ``recovery_timeout`` seconds have passed. It then
    moves to half-open and lets exactly one probe call through. A successful probe closes
    the breaker, a failed probe opens it again.

    Each acquire returns a token for the current generation, and the breaker starts a new
    generation each time it opens. Completions that pass a token from an older generation
    come from calls that started before the breaker opened, so the breaker ignores them.

    The breaker is not thread-safe. It is meant for a single asyncio event loop, where its
    methods never yield control and so cannot interleave.

    Args:
        failure_threshold: Consecutive failures that open the breaker.
        recovery_timeout: Seconds the breaker stays open before it allows a probe.
        clock: Monotonic clock in seconds. Injectable for tests.
    """

    def __init__(
        self,
        failure_threshold: int,
        recovery_timeout: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if recovery_timeout < 0:
            raise ValueError("recovery_timeout cannot be negative")

        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._clock = clock
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._probe_in_flight = False
        self._generation = 0

    @property
    def state(self) -> CircuitState:
        return self._state

    def try_acquire(self) -> int | None:
        """Return a token if a call may go through now, or None if it may not.

        Every token must be passed to exactly one call to ``record_reachable``,
        ``record_failure`` or ``release``.
        """
        if self._state is CircuitState.CLOSED:
            return self._generation

        if self._state is CircuitState.OPEN:
            if self._clock() - self._opened_at < self.recovery_timeout:
                return None
            self._transition(CircuitState.HALF_OPEN)

        if self._probe_in_flight:
            return None

        self._probe_in_flight = True
        return self._generation

    def record_reachable(self, token: int) -> None:
        """Record that the dependency answered, which closes the breaker."""
        if token != self._generation:
            return

        self._probe_in_flight = False
        self._consecutive_failures = 0
        if self._state is not CircuitState.CLOSED:
            self._transition(CircuitState.CLOSED)

    def record_failure(self, token: int) -> None:
        """Record a failed call, which can open the breaker."""
        if token != self._generation:
            return

        if self._state is CircuitState.HALF_OPEN:
            self._probe_in_flight = False
            self._open()
            return

        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._open()

    def release(self, token: int) -> None:
        """Give back an acquired call that ended without an outcome, for example on cancellation."""
        if token == self._generation:
            self._probe_in_flight = False

    def _open(self) -> None:
        self._opened_at = self._clock()
        self._generation += 1
        self._transition(CircuitState.OPEN)

    def _transition(self, new_state: CircuitState) -> None:
        log_method = log.warning if new_state is CircuitState.OPEN else log.info
        log_method(
            "Usage quota circuit breaker state changed",
            previous_state=self._state.value,
            new_state=new_state.value,
            consecutive_failures=self._consecutive_failures,
            recovery_timeout=self.recovery_timeout,
        )
        self._state = new_state
