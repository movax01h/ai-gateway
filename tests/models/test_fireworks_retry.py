import asyncio
import logging
from unittest.mock import patch

import litellm
import pytest
from tenacity import stop_after_delay, wait_fixed

from ai_gateway.models import fireworks_retry
from ai_gateway.models.fireworks_retry import create_fireworks_retry_decorator

# Stand-ins for the production pair (`request_timeout=120` against a 120s budget), scaled
# down so these run in well under a second. Only the ratio matters: one attempt has to be
# able to consume the whole wall-clock budget.
STALL_DURATION = 0.2
CLOCK_BUDGET = 0.1

# Real sleeps are needed because the wall-clock budget measures them, so keep them short.
BACKOFF = 0.02


@pytest.fixture(name="logger")
def logger_fixture():
    return logging.getLogger("test_fireworks_retry")


@pytest.fixture(name="fast_backoff")
def fast_backoff_fixture():
    """Shorten the exponential backoff, which would otherwise dominate the runtime."""
    with patch.object(
        fireworks_retry, "wait_exponential", lambda **_: wait_fixed(BACKOFF)
    ):
        yield


@pytest.fixture(name="shrunken_clock")
def shrunken_clock_fixture():
    """Shrink the wall-clock budget so a real stall can exhaust it quickly."""
    with patch.object(
        fireworks_retry, "_STOP_STATUS_ERRORS", stop_after_delay(CLOCK_BUDGET)
    ):
        yield


def _timeout() -> litellm.Timeout:
    return litellm.Timeout(
        message="Connection timed out",
        model="minimax-m2p7",
        llm_provider="fireworks_ai",
    )


def _service_unavailable() -> litellm.ServiceUnavailableError:
    return litellm.ServiceUnavailableError(
        message="Service temporarily unavailable",
        model="minimax-m2p7",
        llm_provider="fireworks_ai",
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("fast_backoff", "shrunken_clock")
async def test_timeouts_retry_even_when_one_attempt_exceeds_the_clock_budget(logger):
    """A stalled attempt must not consume the whole retry budget.

    ``stop_after_delay`` counts the attempts themselves, so when one attempt lasts as long
    as the budget, the first check already stops the loop and no retry runs. Each stall here
    outlasts the whole clock budget, so a wall-clock stop would yield exactly one attempt.
    """
    attempts = 0

    @create_fireworks_retry_decorator(logger)
    async def stalling_call():
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(STALL_DURATION)
        raise _timeout()

    with pytest.raises(litellm.Timeout):
        await stalling_call()

    assert attempts == 3


@pytest.mark.asyncio
@pytest.mark.usefixtures("fast_backoff")
async def test_timeouts_stop_after_the_attempt_budget(logger):
    """Timeouts get a fixed number of attempts, not an unbounded stream of them."""
    attempts = 0

    @create_fireworks_retry_decorator(logger)
    async def failing_call():
        nonlocal attempts
        attempts += 1
        raise _timeout()

    with pytest.raises(litellm.Timeout):
        await failing_call()

    assert attempts == 3


@pytest.mark.asyncio
@pytest.mark.usefixtures("fast_backoff", "shrunken_clock")
async def test_cold_start_503s_keep_the_wall_clock_budget(logger):
    """Cold starts fail in milliseconds, so the clock must still buy many cheap attempts.

    This is the regression guard for the fix above: budgeting every error by attempt count
    would cut a booting Fireworks instance off long before it is ready.
    """
    attempts = 0

    @create_fireworks_retry_decorator(logger)
    async def failing_call():
        nonlocal attempts
        attempts += 1
        raise _service_unavailable()

    with pytest.raises(litellm.ServiceUnavailableError):
        await failing_call()

    # More than the 3 attempts a connection error gets, because the clock rather than a
    # counter is what bounds this class.
    assert attempts > 3
