from __future__ import annotations

import logging
from typing import Callable, Iterable, Type

import litellm
from tenacity import (
    RetryCallState,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
)

__all__ = ["create_fireworks_retry_decorator"]

DEFAULT_FIREWORKS_ERRORS: tuple[Type[BaseException], ...] = (
    litellm.Timeout,
    litellm.APIError,
    litellm.APIConnectionError,
    litellm.RateLimitError,
    litellm.ServiceUnavailableError,
)

# Transport failures. Each attempt costs a whole request timeout.
_CONNECTION_ERRORS: tuple[Type[BaseException], ...] = (
    litellm.Timeout,
    litellm.APIConnectionError,
)

# Count attempts, not seconds. One timed-out attempt spends a clock budget on its own.
_STOP_CONNECTION_ERRORS = stop_after_attempt(3)

# 503s and rate limits fail in milliseconds, so a clock budget buys many cheap attempts.
_STOP_STATUS_ERRORS = stop_after_delay(120)


def _stop_by_error_class(retry_state: RetryCallState) -> bool:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, _CONNECTION_ERRORS):
        return _STOP_CONNECTION_ERRORS(retry_state)
    return _STOP_STATUS_ERRORS(retry_state)


def create_fireworks_retry_decorator(
    logger: logging.Logger,
    error_types: Iterable[Type[BaseException]] | None = None,
) -> Callable[[Callable], Callable]:
    """Return a tenacity retry decorator with exponential backoff for Fireworks 503 errors.

    Fireworks.ai instances may return 503 when becoming available. This decorator
    implements exponential backoff to handle cold starts gracefully.

    Configuration:
    - Initial wait: 1 second
    - Max wait: 10 seconds
    - Backoff: Exponential (multiplier: 1)
    - Stop: 120 seconds total for status errors, 3 attempts for connection errors

    The stop condition depends on the error class. See ``_stop_by_error_class``.
    """
    errors = tuple(error_types) if error_types else DEFAULT_FIREWORKS_ERRORS

    def _decorator(func: Callable) -> Callable:
        return retry(
            reraise=True,
            stop=_stop_by_error_class,
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(errors),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )(func)

    return _decorator
