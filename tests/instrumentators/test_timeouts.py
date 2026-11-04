import asyncio
from typing import Optional

import httpx
import pytest
from litellm.exceptions import MidStreamFallbackError

from ai_gateway.instrumentators.timeouts import is_timeout_error


def _chained_timeout_error() -> ValueError:
    outer = ValueError("wrapper")
    outer.__cause__ = httpx.ReadTimeout("read timed out")
    return outer


class TestIsTimeoutError:
    @pytest.mark.parametrize(
        ("exc", "expected"),
        [
            (httpx.ReadTimeout("read timed out"), True),
            (asyncio.TimeoutError(), True),
            # litellm surfaces the aiohttp idle-read timeout as a generic
            # connection error carrying a fixed message.
            (
                MidStreamFallbackError(
                    "AnthropicException - Timeout on reading data from socket",
                    model="claude-3",
                    llm_provider="anthropic",
                ),
                True,
            ),
            (_chained_timeout_error(), True),
            (ValueError("boom"), False),
            (None, False),
        ],
    )
    def test_classification(self, exc: Optional[BaseException], expected: bool):
        assert is_timeout_error(exc) is expected

    def test_loop_exhaustion_returns_false(self):
        """A chain of 5+ non-timeout exceptions exhausts the loop and hits the final ``return False``."""
        # Build a chain of 6 plain exceptions so the loop runs all 5 iterations
        # without ever finding a timeout, reaching the ``return False`` after the loop.
        exc: BaseException = ValueError("level-6")
        for i in range(5, 0, -1):
            wrapper = ValueError(f"level-{i}")
            wrapper.__cause__ = exc
            exc = wrapper
        assert is_timeout_error(exc) is False
