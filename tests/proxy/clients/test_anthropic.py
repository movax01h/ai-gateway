import json

import fastapi
import pytest
from fastapi import status

from ai_gateway.proxy.clients.anthropic import AnthropicProxyClient


@pytest.fixture
def anthropic_client(async_client_factory, limits, monkeypatch):
    """Fixture to create an AnthropicProxyClient instance."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    return AnthropicProxyClient(async_client_factory(), limits)


@pytest.fixture
def request_params():
    """Fixture for common request parameters."""
    return {
        "model": "claude-3-5-haiku-20241022",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hi, how are you?"}],
        "stream": True,
    }


@pytest.fixture
def request_headers():
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
