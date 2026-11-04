import asyncio

import httpx
from anthropic import APITimeoutError as AnthropicAPITimeoutError
from openai import APITimeoutError as OpenAIAPITimeoutError

__all__ = ["is_timeout_error"]

# aiohttp's idle-read timer message; litellm surfaces it as a generic
# APIConnectionError, so it can't be classified by exception type alone.
_SOCKET_READ_TIMEOUT_MESSAGE = "Timeout on reading data from socket"

# `litellm.Timeout` subclasses `openai.APITimeoutError`, so it is covered too. Its
# message doesn't necessarily say "timed out", so it needs the type check.
_TIMEOUT_EXCEPTION_TYPES = (
    httpx.TimeoutException,
    asyncio.TimeoutError,
    AnthropicAPITimeoutError,
    OpenAIAPITimeoutError,
)


def is_timeout_error(exc: BaseException | None) -> bool:
    """Best-effort check whether an exception (or its causes) is a timeout.

    Covers httpx, asyncio, Anthropic, OpenAI and litellm timeout classes plus the aiohttp idle-read timeout that
    litellm wraps in ``APIConnectionError``/``MidStreamFallbackError`` with a
    fixed message rather than a distinct type. Walks ``__cause__`` and
    litellm's ``original_exception`` a bounded number of steps to avoid
    reference cycles.
    """
    for _ in range(5):
        if exc is None:
            return False
        if isinstance(exc, _TIMEOUT_EXCEPTION_TYPES):
            return True
        message = str(exc)
        if _SOCKET_READ_TIMEOUT_MESSAGE in message or "timed out" in message.lower():
            # NOTE: "timed out" is a broad heuristic and may over-match
            # exceptions unrelated to the request timeout.
            return True
        exc = exc.__cause__ or getattr(exc, "original_exception", None)
    return False
