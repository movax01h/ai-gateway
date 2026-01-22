import json
from unittest.mock import Mock

import fastapi
import httpx
import pytest
from fastapi import status

from ai_gateway.proxy.clients.anthropic import AnthropicProxyClient
from ai_gateway.proxy.clients.token_usage import TokenUsage


@pytest.fixture(name="anthropic_client")
def anthropic_client_fixture(async_client_factory, limits, monkeypatch):
    """Fixture to create an AnthropicProxyClient instance."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    return AnthropicProxyClient(async_client_factory(), limits)


@pytest.fixture(name="request_params")
def request_params_fixture():
    """Fixture for common request parameters."""
    return {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hi, how are you?"}],
        "stream": True,
    }


@pytest.fixture(name="request_headers")
def request_headers_fixture():
    """Fixture for common request headers."""
    return {
        "accept": "application/json",
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_url,expected_status",
    [
        ("http://0.0.0.0:5052/v1/proxy/anthropic/v1/messages", 200),
        ("http://0.0.0.0:5052/v1/proxy/anthropic/v1/messages?beta=true", 200),
        (
            "http://0.0.0.0:5052/v1/proxy/anthropic/v1/messages/count_tokens?beta=true",
            200,
        ),
    ],
)
async def test_valid_proxy_requests(
    anthropic_client,
    request_factory,
    request_params,
    request_headers,
    request_url,
    expected_status,
):
    """Test valid proxy requests with different URLs."""
    response = await anthropic_client.proxy(
        request_factory(
            request_url=request_url,
            request_body=json.dumps(request_params).encode("utf-8"),
            request_headers=request_headers,
        )
    )

    assert isinstance(response, fastapi.Response)
    assert response.status_code == expected_status


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_url,expected_status,expected_detail",
    [
        (
            "http://0.0.0.0:5052/v1/proxy/anthropic/v1/wrongpath",
            status.HTTP_404_NOT_FOUND,
            "Not found",
        ),
    ],
)
async def test_invalid_proxy_requests(
    anthropic_client,
    request_factory,
    request_params,
    request_headers,
    request_url,
    expected_status,
    expected_detail,
):
    """Test invalid proxy requests that should raise exceptions."""
    with pytest.raises(fastapi.HTTPException) as excinfo:
        await anthropic_client.proxy(
            request_factory(
                request_url=request_url,
                request_body=json.dumps(request_params).encode("utf-8"),
                request_headers=request_headers,
            )
        )

    assert excinfo.value.status_code == expected_status
    assert excinfo.value.detail == expected_detail


@pytest.mark.asyncio
async def test_count_tokens_endpoint(
    anthropic_client,
    request_factory,
    request_headers,
):
    count_tokens_request_body = {
        "model": "claude-haiku-4-5-20251001",
        "messages": [{"role": "user", "content": "Hi, how are you?"}],
    }

    response = await anthropic_client.proxy(
        request_factory(
            request_url="http://0.0.0.0:5052/v1/proxy/anthropic/v1/messages/count_tokens?beta=true",
            request_body=json.dumps(count_tokens_request_body).encode("utf-8"),
            request_headers=request_headers,
        )
    )

    assert isinstance(response, fastapi.Response)
    assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_headers,upstream_response_headers,expected_upstream_headers,expected_downstream_headers",
    [
        pytest.param(
            {
                "accept": "application/json",
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "prompt-caching-2024-07-31",
            },
            {"content-type": "application/json"},
            {"anthropic-beta": "prompt-caching-2024-07-31"},
            {},
            id="anthropic-beta header forwarded to upstream",
        ),
        pytest.param(
            {
                "accept": "application/json",
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            {
                "content-type": "application/json",
                "anthropic-beta": "prompt-caching-2024-07-31",
            },
            {},
            {"anthropic-beta": "prompt-caching-2024-07-31"},
            id="anthropic-beta header forwarded to downstream",
        ),
        pytest.param(
            {
                "accept": "application/json",
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            {"content-type": "application/json"},
            {},
            {},
            id="request without anthropic-beta header",
        ),
        pytest.param(
            {
                "accept": "application/json",
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "prompt-caching-2024-07-31",
                "x-custom-header": "should-be-filtered",
                "authorization": "Bearer token",
            },
            {
                "content-type": "application/json",
                "anthropic-beta": "prompt-caching-2024-07-31",
                "x-custom-header": "should-not-be-forwarded",
            },
            {"anthropic-beta": "prompt-caching-2024-07-31"},
            {"anthropic-beta": "prompt-caching-2024-07-31"},
            id="disallowed headers filtered correctly",
        ),
    ],
)
async def test_anthropic_beta_header_handling(
    limits,
    monkeypatch,
    request_factory,
    request_params,
    request_headers,
    upstream_response_headers,
    expected_upstream_headers,
    expected_downstream_headers,
):
    """Test anthropic-beta header handling in both directions (upstream and downstream)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    # Create mock response
    response_body = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }

    mock_client = Mock(spec=httpx.AsyncClient)
    mock_response = httpx.Response(
        status_code=200,
        headers=upstream_response_headers,
        json=response_body,
    )
    mock_client.send.return_value = mock_response
    mock_client.build_request = httpx.AsyncClient().build_request

    anthropic_client = AnthropicProxyClient(mock_client, limits)

    # Make request
    response = await anthropic_client.proxy(
        request_factory(
            request_url="http://0.0.0.0:5052/v1/proxy/anthropic/v1/messages",
            request_body=json.dumps(request_params).encode("utf-8"),
            request_headers=request_headers,
        )
    )

    # Verify upstream request headers
    mock_client.send.assert_called_once()
    upstream_request = mock_client.send.call_args[0][0]

    for header, value in expected_upstream_headers.items():
        assert (
            header in upstream_request.headers
        ), f"Expected {header} in upstream headers"
        assert upstream_request.headers[header] == value

    # Verify disallowed headers are NOT in upstream request
    if "x-custom-header" in request_headers:
        assert "x-custom-header" not in upstream_request.headers
    if "authorization" in request_headers:
        assert "authorization" not in upstream_request.headers

    # Verify downstream response headers
    for header, value in expected_downstream_headers.items():
        assert header in response.headers, f"Expected {header} in downstream headers"
        assert response.headers[header] == value

    # Verify disallowed headers are NOT in downstream response
    if "x-custom-header" in upstream_response_headers:
        assert "x-custom-header" not in response.headers

    # Verify that headers NOT in expected sets are absent
    if not expected_upstream_headers.get("anthropic-beta"):
        assert "anthropic-beta" not in upstream_request.headers
    if not expected_downstream_headers.get("anthropic-beta"):
        assert "anthropic-beta" not in response.headers

    # Verify response status
    assert response.status_code == 200


class TestAnthropicTokenUsageExtraction:
    """Test cases for Anthropic token usage extraction."""

    @pytest.fixture
    def anthropic_client_for_usage(self, async_client_factory, limits, monkeypatch):
        """Fixture to create an AnthropicProxyClient instance for usage tests."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        return AnthropicProxyClient(async_client_factory(), limits)

    @pytest.mark.parametrize(
        (
            "endpoint,response_body,expected_input,expected_output,"
            "expected_total,expected_cache_creation,expected_cache_read"
        ),
        [
            (
                "/v1/messages",
                {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello!"}],
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                },
                10,
                20,
                30,
                0,
                0,
            ),
            ("/v1/messages", {"id": "msg_123", "type": "message"}, 0, 0, 0, 0, 0),
            (
                "/v1/messages",
                {"usage": {"input_tokens": 0, "output_tokens": 0}},
                0,
                0,
                0,
                0,
                0,
            ),
            ("/v1/messages", {"usage": {"input_tokens": 50}}, 50, 0, 50, 0, 0),
            ("/v1/messages", {"usage": {"output_tokens": 75}}, 0, 75, 75, 0, 0),
            (
                "/v1/messages?beta=true",
                {
                    "id": "msg_beta_123",
                    "usage": {"input_tokens": 100, "output_tokens": 200},
                },
                100,
                200,
                300,
                0,
                0,
            ),
            (
                "/v1/messages",
                {
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "cache_creation_input_tokens": 25,
                    }
                },
                100,
                50,
                150,
                25,
                0,
            ),
            (
                "/v1/messages",
                {
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "cache_read_input_tokens": 40,
                    }
                },
                100,
                50,
                150,
                0,
                40,
            ),
            (
                "/v1/messages",
                {
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "cache_creation_input_tokens": 20,
                        "cache_read_input_tokens": 30,
                    }
                },
                100,
                50,
                150,
                20,
                30,
            ),
            (
                "/v1/messages",
                {"usage": {"input_tokens": 100, "output_tokens": 50}},
                100,
                50,
                150,
                0,
                0,
            ),
            (
                "/v1/messages/count_tokens?beta=true",
                {"usage": {"input_tokens": 150}},
                150,
                0,
                150,
                0,
                0,
            ),
            (
                "/v1/messages/count_tokens?beta=true",
                {
                    "usage": {
                        "input_tokens": 200,
                        "cache_creation_input_tokens": 50,
                        "cache_read_input_tokens": 30,
                    }
                },
                200,
                0,
                200,
                50,
                30,
            ),
        ],
        ids=[
            "Complete response with all fields",
            "Missing usage field entirely",
            "Zero tokens",
            "Partial fields (only input_tokens)",
            "Only output_tokens",
            "Messages beta endpoint",
            "With cache creation tokens",
            "With cache read tokens",
            "With both cache metrics",
            "Without cache metrics",
            "Count tokens endpoint response",
            "Count tokens endpoint with cache metrics",
        ],
    )
    def test_extract_token_usage(
        self,
        anthropic_client_for_usage,
        endpoint,
        response_body,
        expected_input,
        expected_output,
        expected_total,
        expected_cache_creation,
        expected_cache_read,
    ):
        """Test extracting token usage from various Anthropic responses."""
        usage = anthropic_client_for_usage._extract_token_usage(endpoint, response_body)

        assert isinstance(usage, TokenUsage)
        assert usage.input_tokens == expected_input
        assert usage.output_tokens == expected_output
        assert usage.total_tokens == expected_total
        assert usage.cache_creation_input_tokens == expected_cache_creation
        assert usage.cache_read_input_tokens == expected_cache_read

    @pytest.mark.parametrize(
        "response_body",
        [
            None,
            {"usage": None},
            {"usage": "invalid"},
        ],
    )
    def test_extract_token_usage_invalid_response(
        self, anthropic_client_for_usage, response_body
    ):
        """Test extracting token usage from invalid response raises exception."""
        with pytest.raises(fastapi.HTTPException) as excinfo:
            anthropic_client_for_usage._extract_token_usage(
                "/v1/messages", response_body
            )

        assert excinfo.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to extract token usage" in excinfo.value.detail
