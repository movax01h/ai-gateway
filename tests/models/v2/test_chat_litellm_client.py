from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import InputTokenDetails, UsageMetadata

from ai_gateway.models.v2.chat_litellm import ChatLiteLLM, _create_usage_metadata


@pytest.mark.asyncio
async def test_astream_with_stream_options():
    """Test that stream_options is added correctly to super()._astream call."""
    message = HumanMessage(content="Hello")

    with patch(
        "langchain_community.chat_models.ChatLiteLLM._astream"
    ) as mock_super_astream:

        chat = ChatLiteLLM()

        result = []
        async for item in chat._astream(messages=[message]):
            result.append(item)

        # Assert that the correct stream_options were passed
        mock_super_astream.assert_called_once()

        call_kwargs = mock_super_astream.call_args.kwargs
        assert call_kwargs["stream_options"] == {"include_usage": True}


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
                input_tokens=4,
                output_tokens=2,
                total_tokens=6,
                input_token_details=InputTokenDetails(cache_creation=3, cache_read=4),
            ),
        ),
    ],
)
async def test_create_usage_metadata(token_usage, expected_usage_metadata):
    assert _create_usage_metadata(token_usage) == expected_usage_metadata
