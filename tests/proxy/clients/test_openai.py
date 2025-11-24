import json

import fastapi
import pytest
from fastapi import status

from ai_gateway.proxy.clients.openai import OpenAIProxyClient
from ai_gateway.proxy.clients.token_usage import TokenUsage


@pytest.fixture(name="openai_client")
def openai_client_fixture(async_client_factory, limits, monkeypatch):
    """Fixture to create an OpenAIProxyClient instance."""
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    return OpenAIProxyClient(async_client_factory(), limits)


@pytest.fixture(name="request_params")
def request_params_fixture():
    """Fixture for common request parameters."""
    return {
        "model": "gpt-5",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hi, how are you?"}],
        "stream": True,
    }


@pytest.fixture(name="completion_request_params")
def completion_request_params_fixture():
    """Fixture for completion API request parameters."""
    return {
        "model": "gpt-5",
        "prompt": "Hello, world!",
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False,
    }


@pytest.fixture(name="request_headers")
def request_headers_fixture():
    """Fixture for common request headers."""
    return {
        "accept": "application/json",
        "content-type": "application/json",
        "user-agent": "test-client",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_url,expected_status",
    [
        ("http://0.0.0.0:5052/v1/proxy/openai/v1/chat/completions", 200),
        ("http://0.0.0.0:5052/v1/proxy/openai/v1/completions", 200),
        ("http://0.0.0.0:5052/v1/proxy/openai/v1/embeddings", 200),
        ("http://0.0.0.0:5052/v1/proxy/openai/v1/models", 200),
        ("http://0.0.0.0:5052/v1/proxy/openai/v1/responses", 200),
    ],
)
async def test_valid_proxy_requests(
    openai_client,
    request_factory,
    request_params,
    request_headers,
    request_url,
    expected_status,
):
    """Test valid proxy requests with different URLs."""
    response = await openai_client.proxy(
        request_factory(
            request_url=request_url,
            request_body=json.dumps(request_params).encode("utf-8"),
            request_headers=request_headers,
        )
    )

    assert isinstance(response, fastapi.Response)
    assert response.status_code == expected_status


@pytest.mark.asyncio
async def test_valid_completions_request(
    openai_client,
    request_factory,
    completion_request_params,
    request_headers,
):
    """Test valid completion requests."""
    response = await openai_client.proxy(
        request_factory(
            request_url="http://0.0.0.0:5052/v1/proxy/openai/v1/completions",
            request_body=json.dumps(completion_request_params).encode("utf-8"),
            request_headers=request_headers,
        )
    )

    assert isinstance(response, fastapi.Response)
    assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_url,expected_status,expected_detail",
    [
        (
            "http://0.0.0.0:5052/v1/proxy/openai/v1/wrongpath",
            status.HTTP_404_NOT_FOUND,
            "Not found",
        ),
        (
            "http://0.0.0.0:5052/v1/proxy/openai/v2/chat/completions",
            status.HTTP_404_NOT_FOUND,
            "Not found",
        ),
    ],
)
async def test_invalid_proxy_requests(
    openai_client,
    request_factory,
    request_params,
    request_headers,
    request_url,
    expected_status,
    expected_detail,
):
    """Test invalid proxy requests that should raise exceptions."""
    with pytest.raises(fastapi.HTTPException) as excinfo:
        await openai_client.proxy(
            request_factory(
                request_url=request_url,
                request_body=json.dumps(request_params).encode("utf-8"),
                request_headers=request_headers,
            )
        )

    assert excinfo.value.status_code == expected_status
    assert excinfo.value.detail == expected_detail


@pytest.mark.asyncio
async def test_missing_model_in_request(
    openai_client,
    request_factory,
    request_headers,
):
    """Test request missing model parameter."""
    request_params_no_model = {
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hi, how are you?"}],
    }

    with pytest.raises(fastapi.HTTPException) as excinfo:
        await openai_client.proxy(
            request_factory(
                request_url="http://0.0.0.0:5052/v1/proxy/openai/v1/chat/completions",
                request_body=json.dumps(request_params_no_model).encode("utf-8"),
                request_headers=request_headers,
            )
        )

    assert excinfo.value.status_code == status.HTTP_400_BAD_REQUEST
    assert excinfo.value.detail == "Failed to extract model name"


@pytest.mark.asyncio
async def test_unsupported_model(
    openai_client,
    request_factory,
    request_headers,
):
    """Test request with unsupported model."""
    request_params_unsupported = {
        "model": "unsupported-model",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hi, how are you?"}],
    }

    with pytest.raises(fastapi.HTTPException) as excinfo:
        await openai_client.proxy(
            request_factory(
                request_url="http://0.0.0.0:5052/v1/proxy/openai/v1/chat/completions",
                request_body=json.dumps(request_params_unsupported).encode("utf-8"),
                request_headers=request_headers,
            )
        )

    assert excinfo.value.status_code == status.HTTP_400_BAD_REQUEST
    assert excinfo.value.detail == "Unsupported model"


@pytest.mark.asyncio
async def test_missing_api_key(
    async_client_factory,
    limits,
    request_factory,
    request_params,
    request_headers,
    monkeypatch,
):
    """Test request with missing API key."""
    # Remove the API key from environment
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    openai_client_local = OpenAIProxyClient(async_client_factory(), limits)

    with pytest.raises(fastapi.HTTPException) as excinfo:
        await openai_client_local.proxy(
            request_factory(
                request_url="http://0.0.0.0:5052/v1/proxy/openai/v1/chat/completions",
                request_body=json.dumps(request_params).encode("utf-8"),
                request_headers=request_headers,
            )
        )

    assert excinfo.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert excinfo.value.detail == "API key not found"


@pytest.mark.asyncio
async def test_stream_flag_extraction(
    openai_client,
    request_factory,
    request_headers,
):
    """Test stream flag extraction from request."""
    # Test with stream=True
    stream_params = {
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }

    response = await openai_client.proxy(
        request_factory(
            request_url="http://0.0.0.0:5052/v1/proxy/openai/v1/chat/completions",
            request_body=json.dumps(stream_params).encode("utf-8"),
            request_headers=request_headers,
        )
    )

    assert isinstance(response, fastapi.Response)
    assert response.status_code == 200

    # Test with stream=False (default)
    no_stream_params = {
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    response = await openai_client.proxy(
        request_factory(
            request_url="http://0.0.0.0:5052/v1/proxy/openai/v1/chat/completions",
            request_body=json.dumps(no_stream_params).encode("utf-8"),
            request_headers=request_headers,
        )
    )

    assert isinstance(response, fastapi.Response)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_invalid_json_body(
    openai_client,
    request_factory,
    request_headers,
):
    """Test invalid JSON in request body."""
    with pytest.raises(fastapi.HTTPException) as excinfo:
        await openai_client.proxy(
            request_factory(
                request_url="http://0.0.0.0:5052/v1/proxy/openai/v1/chat/completions",
                request_body=b"invalid json",
                request_headers=request_headers,
            )
        )

    assert excinfo.value.status_code == status.HTTP_400_BAD_REQUEST
    assert excinfo.value.detail == "Invalid JSON"


class TestOpenAITokenUsageExtraction:

    @pytest.fixture(name="openai_client_for_usage")
    def openai_usage_client(self, async_client_factory, limits, monkeypatch):
        """Fixture to create an OpenAIProxyClient instance for usage tests."""
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        return OpenAIProxyClient(async_client_factory(), limits)

    @pytest.mark.parametrize(
        (
            "endpoint,response_body,expected_input,expected_output,"
            "expected_total,expected_cache_read,expected_cache_creation"
        ),
        [
            (
                "/v1/chat/completions",
                {
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30,
                    },
                },
                10,
                20,
                30,
                0,
                0,
            ),
            (
                "/v1/chat/completions",
                {"usage": {"prompt_tokens": 15, "completion_tokens": 25}},
                15,
                25,
                40,
                0,
                0,
            ),
            (
                "/v1/chat/completions",
                {"id": "chatcmpl-123", "object": "chat.completion"},
                0,
                0,
                0,
                0,
                0,
            ),
            (
                "/v1/chat/completions",
                {
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                },
                0,
                0,
                0,
                0,
                0,
            ),
            (
                "/v1/chat/completions",
                {"usage": {"prompt_tokens": 100}},
                100,
                0,
                100,
                0,
                0,
            ),
            (
                "/v1/completions",
                {
                    "id": "cmpl-123",
                    "object": "text_completion",
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 10,
                        "total_tokens": 15,
                    },
                },
                5,
                10,
                15,
                0,
                0,
            ),
            (
                "/v1/embeddings",
                {
                    "object": "list",
                    "data": [],
                    "usage": {"prompt_tokens": 8, "total_tokens": 8},
                },
                8,
                0,
                8,
                0,
                0,
            ),
            (
                "/v1/chat/completions",
                {
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                        "prompt_tokens_details": {"cached_tokens": 30},
                    }
                },
                100,
                50,
                150,
                30,
                0,
            ),
            (
                "/v1/chat/completions",
                {
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    }
                },
                100,
                50,
                150,
                0,
                0,
            ),
            (
                "/v1/chat/completions",
                {
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "prompt_tokens_details": {},
                    }
                },
                100,
                50,
                150,
                0,
                0,
            ),
            (
                "/v1/chat/completions",
                {
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "prompt_tokens_details": "invalid",
                    }
                },
                100,
                50,
                150,
                0,
                0,
            ),
            (
                "/v1/chat/completions",
                {
                    "id": "chatcmpl-CdJZaJTA6UgqxTVQIWH3N8itNUq9Z",
                    "object": "chat.completion",
                    "model": "gpt-5-2025-08-07",
                    "usage": {
                        "prompt_tokens": 38,
                        "completion_tokens": 1024,
                        "total_tokens": 1062,
                        "prompt_tokens_details": {"cached_tokens": 0},
                        "completion_tokens_details": {
                            "reasoning_tokens": 1024,
                            "accepted_prediction_tokens": 0,
                            "rejected_prediction_tokens": 0,
                        },
                    },
                },
                38,
                1024,
                1062,
                0,
                0,
            ),
        ],
        ids=[
            "Complete response with all fields",
            "Missing total_tokens (auto-computed)",
            "Missing usage field entirely",
            "Zero tokens",
            "Partial fields (only prompt_tokens)",
            "Completions endpoint",
            "Embeddings endpoint (no completion tokens)",
            "With cached tokens",
            "Without cached tokens",
            "Empty prompt_tokens_details",
            "Invalid prompt_tokens_details (ignored gracefully)",
            "With reasoning tokens (gpt-5 model)",
        ],
    )
    def test_extract_token_usage(
        self,
        openai_client_for_usage,
        endpoint,
        response_body,
        expected_input,
        expected_output,
        expected_total,
        expected_cache_read,
        expected_cache_creation,
    ):
        usage = openai_client_for_usage._extract_token_usage(endpoint, response_body)

        assert isinstance(usage, TokenUsage)
        assert usage.input_tokens == expected_input
        assert usage.output_tokens == expected_output
        assert usage.total_tokens == expected_total
        assert usage.cache_read_input_tokens == expected_cache_read
        assert usage.cache_creation_input_tokens == expected_cache_creation

    @pytest.mark.parametrize(
        "response_body",
        [
            None,
            {"usage": None},
            {"usage": "invalid"},
        ],
    )
    def test_extract_token_usage_invalid_response(
        self, openai_client_for_usage, response_body
    ):
        """Test extracting token usage from invalid response raises exception."""
        with pytest.raises(fastapi.HTTPException) as excinfo:
            openai_client_for_usage._extract_token_usage(
                "/v1/chat/completions", response_body
            )

        assert excinfo.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to extract token usage" in excinfo.value.detail
