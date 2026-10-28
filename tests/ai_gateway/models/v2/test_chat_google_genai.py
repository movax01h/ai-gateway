from unittest.mock import MagicMock, patch

import httpx
import pytest
from google.genai import Client
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_gateway.models.v2.chat_google_genai import (
    ChatGoogleGenerativeAI,
    connect_google_gen_vertex_ai,
)


def test_connect_google_gen_vertex_ai_passes_httpx_async_client():
    """The httpx client is forwarded to `HttpOptions` so it's pooled across resolutions."""
    client = MagicMock(spec=httpx.AsyncClient)

    with patch("ai_gateway.models.v2.chat_google_genai.Client") as mock_client:
        connect_google_gen_vertex_ai(
            project="test-project",
            location="us-central1",
            http_client=client,
        )

    _, kwargs = mock_client.call_args
    assert kwargs["http_options"].httpx_async_client is client


class _Sent(Exception):
    pass


def test_claude_tool_call_history_is_sent_to_gemini_without_claude_thinking():
    """A session switched from Claude to Gemini replays Claude's turns.

    langchain-google-genai 4.2.1 forwarded Claude's thinking signature and left the function call unsigned, and Gemini
    returned 400 for both (#3290).
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-4-argon",
        google_api_key="dummy",
        client=Client(api_key="test"),
        max_retries=0,
    )
    history = [
        HumanMessage("hi"),
        AIMessage(
            [
                {"type": "thinking", "thinking": "", "signature": "c2lnbmF0dXJl"},
                {"type": "tool_use", "id": "t1", "name": "f", "input": {}},
            ],
            tool_calls=[{"id": "t1", "name": "f", "args": {}}],
            response_metadata={"model_provider": "anthropic"},
        ),
        ToolMessage("result", tool_call_id="t1"),
    ]

    with patch.object(llm.client.models, "generate_content", side_effect=_Sent) as send:
        with pytest.raises(_Sent):
            llm.invoke(history)

    # The Claude turn becomes a single function call part: its thinking block is dropped.
    (part,) = send.call_args.kwargs["contents"][1].parts
    assert part.function_call.name == "f"
    assert part.thought_signature  # Gemini 3+ rejects a function call without one
