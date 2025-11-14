from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.messages.ai import InputTokenDetails, UsageMetadata
from langchain_core.outputs import ChatGenerationChunk

from ai_gateway.models.v2.chat_litellm import ChatLiteLLM, _create_usage_metadata


@pytest.mark.asyncio
async def test_astream_with_stream_options_and_stop_reason():
    """Test that stream_options is added correctly and finish_reason is extracted from stop_reason."""

    message = HumanMessage(content="Hello")

    # Mock the raw LiteLLM response chunks
    mock_chunks = [
        {
            "choices": [
                {
                    "delta": {"role": "assistant", "content": "Hello"},
                    "finish_reason": None,
                    "index": 0,
                }
            ],
            "usage": {},
        },
        {
            "choices": [
                {
                    "delta": {"content": " world"},
                    "finish_reason": None,
                    "index": 0,
                }
            ],
            "usage": {},
        },
        {
            "choices": [
                {
                    "delta": {},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    ]

    async def mock_acompletion(*args, **kwargs):
        for chunk in mock_chunks:
            yield chunk

    with patch(
        "ai_gateway.models.v2.chat_litellm.acompletion_with_retry",
        new=AsyncMock(return_value=mock_acompletion()),
    ):
        chat = ChatLiteLLM(model="gpt-3.5-turbo")

        result = []
        async for item in chat._astream(messages=[message]):
            result.append(item)

        # Verify we got chunks
        assert len(result) == 3

        # Verify the last chunk has finish_reason in response_metadata
        last_chunk = result[-1]
        assert isinstance(last_chunk, ChatGenerationChunk)
        assert isinstance(last_chunk.message, AIMessageChunk)
        assert last_chunk.message.response_metadata.get("finish_reason") == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("token_usage", "expected_usage_metadata"),
    [
        (
            {
                "prompt_tokens": 1,
                "completion_tokens": 2,
            },
            UsageMetadata(input_tokens=1, output_tokens=2, total_tokens=3),
        ),
        (
            {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 4,
            },
            UsageMetadata(
                input_tokens=1,
                output_tokens=2,
                total_tokens=3,
                input_token_details=InputTokenDetails(cache_creation=3, cache_read=4),
            ),
        ),
    ],
)
async def test_create_usage_metadata(token_usage, expected_usage_metadata):
    assert _create_usage_metadata(token_usage) == expected_usage_metadata
