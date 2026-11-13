"""Unit tests for the usage quota CircuitBreaker."""

import pytest

from lib.usage_quota.circuit_breaker import CircuitBreaker, CircuitState


class FakeClock:
    """Monotonic clock that only moves when a test advances `now`."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture(name="clock")
def clock_fixture() -> FakeClock:
    return FakeClock()


@pytest.fixture(name="breaker")
def breaker_fixture(clock: FakeClock) -> CircuitBreaker:
    return CircuitBreaker(failure_threshold=3, recovery_timeout=30.0, clock=clock)


def _acquire(breaker: CircuitBreaker) -> int:
    token = breaker.try_acquire()
    assert token is not None
    return token


def _open(breaker: CircuitBreaker) -> None:
    for _ in range(breaker.failure_threshold):
        breaker.record_failure(_acquire(breaker))
    assert breaker.state is CircuitState.OPEN


def _acquire_probe(breaker: CircuitBreaker, clock: FakeClock) -> int:
    _open(breaker)
    clock.now += breaker.recovery_timeout
    return _acquire(breaker)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"failure_threshold": 0, "recovery_timeout": 1.0},
        {"failure_threshold": 1, "recovery_timeout": -1.0},
    ],
)
def test_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        CircuitBreaker(**kwargs)


def test_starts_closed_and_allows_calls(breaker: CircuitBreaker):
    assert breaker.state is CircuitState.CLOSED
    _acquire(breaker)
    _acquire(breaker)


def test_token_increments_only_when_breaker_opens(
    breaker: CircuitBreaker, clock: FakeClock
):
    token = _acquire(breaker)
    breaker.release(token)
    same_generation = _acquire(breaker)
    assert same_generation == token
    breaker.release(same_generation)

    probe = _acquire_probe(breaker, clock)
    assert probe == token + 1

    breaker.record_failure(probe)
    clock.now += breaker.recovery_timeout
    next_probe = _acquire(breaker)
    assert next_probe == probe + 1

    breaker.record_reachable(next_probe)
    after_recovery = _acquire(breaker)
    assert after_recovery == next_probe
    breaker.release(after_recovery)


def test_stays_closed_below_threshold(breaker: CircuitBreaker):
    for _ in range(breaker.failure_threshold - 1):
        breaker.record_failure(_acquire(breaker))

    assert breaker.state is CircuitState.CLOSED
    _acquire(breaker)


def test_opens_after_consecutive_failures(breaker: CircuitBreaker):
    _open(breaker)

    assert breaker.try_acquire() is None


def test_reachable_resets_failure_count(breaker: CircuitBreaker):
    for _ in range(breaker.failure_threshold - 1):
        breaker.record_failure(_acquire(breaker))
    breaker.record_reachable(_acquire(breaker))
    for _ in range(breaker.failure_threshold - 1):
        breaker.record_failure(_acquire(breaker))

    assert breaker.state is CircuitState.CLOSED


@pytest.mark.parametrize(
    "elapsed,expected_allowed", [(29.9, False), (30.0, True), (45.0, True)]
)
def test_allows_probe_only_after_recovery_timeout(
    breaker: CircuitBreaker, clock: FakeClock, elapsed: float, expected_allowed: bool
):
    _open(breaker)
    clock.now += elapsed

    assert (breaker.try_acquire() is not None) is expected_allowed
    expected_state = CircuitState.HALF_OPEN if expected_allowed else CircuitState.OPEN
    assert breaker.state is expected_state


def test_half_open_allows_a_single_probe(breaker: CircuitBreaker, clock: FakeClock):
    _acquire_probe(breaker, clock)

    assert breaker.try_acquire() is None
    assert breaker.state is CircuitState.HALF_OPEN


def test_successful_probe_closes(breaker: CircuitBreaker, clock: FakeClock):
    probe = _acquire_probe(breaker, clock)

    breaker.record_reachable(probe)

    assert breaker.state is CircuitState.CLOSED
    _acquire(breaker)
    _acquire(breaker)


def test_failed_probe_reopens_for_a_new_cooldown(
    breaker: CircuitBreaker, clock: FakeClock
):
    probe = _acquire_probe(breaker, clock)

    breaker.record_failure(probe)

    assert breaker.state is CircuitState.OPEN
    clock.now += breaker.recovery_timeout - 0.1
    assert breaker.try_acquire() is None
    clock.now += 0.1
    _acquire(breaker)


def test_released_probe_allows_another_probe(breaker: CircuitBreaker, clock: FakeClock):
    probe = _acquire_probe(breaker, clock)

    breaker.release(probe)

    assert breaker.state is CircuitState.HALF_OPEN
    _acquire(breaker)


@pytest.mark.parametrize("outcome", ["record_reachable", "record_failure", "release"])
def test_ignores_completion_of_call_that_started_before_opening_while_open(
    breaker: CircuitBreaker, clock: FakeClock, outcome: str
):
    stale = _acquire(breaker)
    _open(breaker)

    getattr(breaker, outcome)(stale)

    assert breaker.state is CircuitState.OPEN
    clock.now += breaker.recovery_timeout - 0.1
    assert breaker.try_acquire() is None
    clock.now += 0.1
    _acquire(breaker)


@pytest.mark.parametrize("outcome", ["record_reachable", "record_failure", "release"])
def test_ignores_completion_of_call_that_started_before_opening_while_half_open(
    breaker: CircuitBreaker, clock: FakeClock, outcome: str
):
    stale = _acquire(breaker)
    probe = _acquire_probe(breaker, clock)

    getattr(breaker, outcome)(stale)

    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.try_acquire() is None
    breaker.record_reachable(probe)
    assert breaker.state is CircuitState.CLOSED


def test_ignores_failure_of_call_that_started_before_the_last_opening(
    breaker: CircuitBreaker, clock: FakeClock
):
    stale = _acquire(breaker)
    breaker.record_reachable(_acquire_probe(breaker, clock))

    for _ in range(breaker.failure_threshold - 1):
        breaker.record_failure(_acquire(breaker))
    breaker.record_failure(stale)

    assert breaker.state is CircuitState.CLOSED
