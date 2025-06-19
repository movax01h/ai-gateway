from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import litellm
import pytest
from google.auth.credentials import Credentials
from langchain_litellm import ChatLiteLLM

from ai_gateway.prompts import Prompt


@pytest.fixture
def model():
    return ChatLiteLLM(
        model="claude-3-sonnet@20240229", custom_llm_provider="vertex_ai", max_retries=3  # type: ignore[call-arg]
    )


@pytest.fixture
def mock_http(mock_http_handler: Mock):
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler",
        return_value=mock_http_handler,
    ) as mock:
        yield mock


@pytest.fixture
def mock_http_handler(response_text: str):
    handler = AsyncMock()
    handler.post.return_value = httpx.Response(status_code=200, text=response_text)
    return handler


@pytest.fixture
def mock_sleep():  # So we don't have to wait
    with patch("asyncio.sleep"):
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize(("response_text"), ['{"error": "something went wrong"}'])
@pytest.mark.usefixtures("mock_sleep")
async def test_ainvoke(
    mock_http: Mock,
    mock_http_handler: AsyncMock,
    prompt: Prompt,
):
    with pytest.raises(litellm.APIConnectionError, match="something went wrong"), patch(
        "google.auth.default",
        return_value=(
            MagicMock(spec=Credentials, token="mock_token"),
            "mock_project_id",
        ),
    ):
        await prompt.ainvoke({"name": "Duo", "content": "What's up?"})

    mock_http.assert_called_once()
    assert mock_http_handler.post.call_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_text"),
    [
        'data:{"type":"error","error":{"details":null,"type":"api_error","message":"something went wrong"}}'
    ],
)
@pytest.mark.usefixtures("mock_sleep")
async def test_astream(
    mock_http: Mock,
    mock_http_handler: AsyncMock,
    prompt: Prompt,
):
    litellm.module_level_aclient = mock_http_handler

    with pytest.raises(
        litellm.InternalServerError, match="something went wrong"
    ), patch(
        "google.auth.default",
        return_value=(
            MagicMock(spec=Credentials, token="mock_token"),
            "mock_project_id",
        ),
    ):
        await anext(prompt.astream({"name": "Duo", "content": "What's up?"}))

    mock_http.assert_not_called()
    assert mock_http_handler.post.call_count == 1
