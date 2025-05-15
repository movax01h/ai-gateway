from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

from ai_gateway.models.v2.chat_litellm import ChatLiteLLM


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
