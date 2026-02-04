from __future__ import annotations

import logging
from typing import Callable, Iterable, Type

import litellm
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
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
    - Total timeout: 120 seconds
    - Backoff: Exponential (multiplier: 2)
    """
    errors = tuple(error_types) if error_types else DEFAULT_FIREWORKS_ERRORS

    def _decorator(func: Callable) -> Callable:
        return retry(
            reraise=True,
            stop=stop_after_delay(120),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(errors),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )(func)

    return _decorator
