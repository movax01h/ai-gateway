import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(name="no_facilitator_key")
def no_facilitator_key_fixture():
    """Keep a developer's real ANTHROPIC_FACILITATOR_KEY out of header assertions."""
    with patch("ai_gateway.models.base.config") as mock_config:
        mock_config.anthropic_facilitator_key = None
        yield mock_config


@pytest.fixture(name="mock_litellm_acompletion_streamed")
def mock_litellm_acompletion_streamed_fixture():
    with patch("ai_gateway.models.litellm.acompletion") as mock_acompletion:
        streamed_response = AsyncMock()
        streamed_response.__aiter__.return_value = iter(
            [
                AsyncMock(
                    choices=[AsyncMock(delta=AsyncMock(content="Streamed content"))]
                )
            ]
        )

        mock_acompletion.return_value = streamed_response

        yield mock_acompletion


@pytest.fixture(name="anthropic_refusal_sse")
def anthropic_refusal_sse_fixture():
    """Build the Anthropic SSE body of a turn ending with ``stop_reason`` (a refusal by default)."""

    def build(stop_details: dict | None, stop_reason: str = "refusal") -> str:
        events: list[dict[str, Any]] = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5-5",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason,
                    "stop_sequence": None,
                    "stop_details": stop_details,
                },
                "usage": {"output_tokens": 0},
            },
            {"type": "message_stop"},
        ]
        return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)

    return build
