from unittest.mock import AsyncMock, patch

import httpx
import pytest
from litellm.llms.anthropic import common_utils as anthropic_common_utils
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

from ai_gateway.models.v2.chat_litellm import ChatLiteLLM
from ai_gateway.vendor.langchain_litellm.litellm import ChatLiteLLM as _LChatLiteLLM
from lib.context.request_metadata import LLMFinishReason, extract_finish_reason


def _mocked_litellm_client(body: str, hosts: list[str]) -> AsyncHTTPHandler:
    def respond(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=body
        )

    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("custom_llm_provider", "expected_host"),
    [
        ("anthropic", "api.anthropic.com"),
        ("vertex_ai", "aiplatform.googleapis.com"),
    ],
)
@pytest.mark.parametrize(
    "stop_details",
    [
        {
            "type": "refusal",
            "category": "reasoning_extraction",
            "explanation": "reverse engineering or duplicating model outputs",
        },
        None,
    ],
)
async def test_anthropic_refusal_survives_litellm_streaming(
    stop_details, custom_llm_provider, expected_host, anthropic_refusal_sse, monkeypatch
):
    """Drive the real litellm Anthropic stream, so a litellm bump that stops calling the patched parser fails here."""
    monkeypatch.setattr(
        VertexBase,
        "_ensure_access_token",
        lambda *_args, **_kwargs: ("token", "project"),
    )
    hosts: list[str] = []
    client = _mocked_litellm_client(anthropic_refusal_sse(stop_details), hosts)
    chat = ChatLiteLLM(
        model="claude-sonnet-5-5",
        custom_llm_provider=custom_llm_provider,
        api_key="test",
    )

    chunks = [chunk async for chunk in chat.astream("hello", client=client)]
    result = sum(chunks[1:], chunks[0])

    assert len(hosts) == 1 and hosts[0].endswith(expected_host)
    assert result.response_metadata["finish_reason"] == "content_filter"
    assert "stop_reason" not in result.response_metadata
    assert result.response_metadata["stop_details"] == stop_details
    assert (
        extract_finish_reason(result.response_metadata)
        == LLMFinishReason.CONTENT_FILTER
    )


@pytest.mark.asyncio
async def test_anthropic_end_turn_gains_no_refusal_details(anthropic_refusal_sse):
    chat = ChatLiteLLM(
        model="claude-sonnet-5-5", custom_llm_provider="anthropic", api_key="test"
    )
    client = _mocked_litellm_client(
        anthropic_refusal_sse(None, stop_reason="end_turn"), []
    )

    chunks = [chunk async for chunk in chat.astream("hello", client=client)]
    result = sum(chunks[1:], chunks[0])

    assert result.response_metadata["finish_reason"] == "stop"
    assert "stop_reason" not in result.response_metadata
    assert "stop_details" not in result.response_metadata


@pytest.mark.asyncio
async def test_non_anthropic_content_filter_stays_a_content_filter():
    """A Gemini or Bedrock `content_filter` gains no `stop_details`, so it logs no refusal category."""

    async def stream():
        yield {
            "choices": [
                {
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": "content_filter",
                }
            ]
        }

    chat = ChatLiteLLM(model="gemini-2.5-pro", custom_llm_provider="vertex_ai")

    with patch.object(
        _LChatLiteLLM, "acompletion_with_retry", new=AsyncMock(return_value=stream())
    ):
        chunks = [chunk async for chunk in chat.astream("hello")]
    result = sum(chunks[1:], chunks[0])

    assert result.response_metadata["finish_reason"] == "content_filter"
    assert "stop_details" not in result.response_metadata


def test_patch_is_still_needed():
    assert not hasattr(
        anthropic_common_utils, "stop_reason_provider_specific_fields"
    ), (
        "litellm now passes Anthropic refusals through itself, so the patch no longer "
        "applies: delete ai_gateway/models/v2/litellm_refusal_details_patch.py, its "
        "import in chat_litellm.py and this test, and move the streaming specs above "
        "into test_chat_litellm.py."
    )
