import json

import fastapi
import pytest
from fastapi import status

from ai_gateway.proxy.clients import OpenAIProxyModelFactory, ProxyClient


@pytest.fixture(name="openai_factory")
def openai_factory_fixture(monkeypatch):
    """Fixture to create an OpenAIProxyModelFactory instance."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return OpenAIProxyModelFactory()


@pytest.fixture(name="proxy_client")
def proxy_client_fixture(limits, internal_event_client, billing_event_client):
    """Fixture to create a ProxyClient instance."""
    return ProxyClient(limits, internal_event_client, billing_event_client)


@pytest.fixture(name="request_params")
def request_params_fixture():
    """Fixture for common request parameters."""
    return {
        "model": "gpt-5-codex",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hi, how are you?"}],
        "stream": True,
    }


@pytest.fixture(name="completion_request_params")
def completion_request_params_fixture():
    """Fixture for completion API request parameters."""
    return {
        "model": "gpt-5-codex",
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
    openai_factory,
    proxy_client,
    request_factory,
    request_params,
    request_headers,
    request_url,
    expected_status,
):
    """Test valid proxy requests with different URLs."""
    request = request_factory(
        request_url=request_url,
        request_body=json.dumps(request_params).encode("utf-8"),
        request_headers=request_headers,
    )

    model = await openai_factory.factory(request)
    response = await proxy_client.proxy(request, model)

    assert isinstance(response, fastapi.Response)
    assert response.status_code == expected_status


@pytest.mark.asyncio
async def test_valid_completions_request(
    openai_factory,
    proxy_client,
    request_factory,
    completion_request_params,
    request_headers,
):
    """Test valid completion requests."""
    request = request_factory(
        request_url="http://0.0.0.0:5052/v1/proxy/openai/v1/completions",
        request_body=json.dumps(completion_request_params).encode("utf-8"),
        request_headers=request_headers,
    )

    model = await openai_factory.factory(request)
    response = await proxy_client.proxy(request, model)

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
    openai_factory,
    request_factory,
    request_params,
    request_headers,
    request_url,
    expected_status,
    expected_detail,
):
    """Test invalid proxy requests that should raise exceptions."""
    request = request_factory(
        request_url=request_url,
        request_body=json.dumps(request_params).encode("utf-8"),
        request_headers=request_headers,
    )

    with pytest.raises(fastapi.HTTPException) as excinfo:
        await openai_factory.factory(request)

    assert excinfo.value.status_code == expected_status
    assert excinfo.value.detail == expected_detail


@pytest.mark.asyncio
async def test_missing_model_in_request(
    openai_factory,
    request_factory,
    request_headers,
):
    """Test request missing model parameter."""
    request_params_no_model = {
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hi, how are you?"}],
    }

    request = request_factory(
        request_url="http://0.0.0.0:5052/v1/proxy/openai/v1/chat/completions",
        request_body=json.dumps(request_params_no_model).encode("utf-8"),
        request_headers=request_headers,
    )

    with pytest.raises(fastapi.HTTPException) as excinfo:
        await openai_factory.factory(request)

    assert excinfo.value.status_code == status.HTTP_400_BAD_REQUEST
    assert excinfo.value.detail == "Failed to extract model name"


@pytest.mark.asyncio
async def test_unsupported_model(
    openai_factory,
    request_factory,
    request_headers,
):
    """Test request with unsupported model."""
    request_params_unsupported = {
        "model": "unsupported-model",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hi, how are you?"}],
    }

    request = request_factory(
        request_url="http://0.0.0.0:5052/v1/proxy/openai/v1/chat/completions",
        request_body=json.dumps(request_params_unsupported).encode("utf-8"),
        request_headers=request_headers,
    )

    with pytest.raises(fastapi.HTTPException) as excinfo:
        await openai_factory.factory(request)

    assert excinfo.value.status_code == status.HTTP_400_BAD_REQUEST
    assert excinfo.value.detail == "Unsupported model"


@pytest.mark.asyncio
async def test_missing_api_key(
    request_factory,
    request_params,
    request_headers,
    monkeypatch,
):
    """Test request with missing API key."""
    # Remove the API key from environment
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    openai_factory_local = OpenAIProxyModelFactory()

    request = request_factory(
        request_url="http://0.0.0.0:5052/v1/proxy/openai/v1/chat/completions",
        request_body=json.dumps(request_params).encode("utf-8"),
        request_headers=request_headers,
    )

    with pytest.raises(fastapi.HTTPException) as excinfo:
        await openai_factory_local.factory(request)

    assert excinfo.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert excinfo.value.detail == "API key not found"


@pytest.mark.asyncio
async def test_stream_flag_extraction(
    openai_factory,
    proxy_client,
    request_factory,
    request_headers,
):
    """Test stream flag extraction from request."""
    # Test with stream=True
    stream_params = {
        "model": "gpt-5-codex",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }

    request = request_factory(
        request_url="http://0.0.0.0:5052/v1/proxy/openai/v1/chat/completions",
        request_body=json.dumps(stream_params).encode("utf-8"),
        request_headers=request_headers,
    )

    model = await openai_factory.factory(request)
    response = await proxy_client.proxy(request, model)

    assert isinstance(response, fastapi.Response)
    assert response.status_code == 200

    # Test with stream=False (default)
    no_stream_params = {
        "model": "gpt-5-codex",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    request = request_factory(
        request_url="http://0.0.0.0:5052/v1/proxy/openai/v1/chat/completions",
        request_body=json.dumps(no_stream_params).encode("utf-8"),
        request_headers=request_headers,
    )

    model = await openai_factory.factory(request)
    response = await proxy_client.proxy(request, model)

    assert isinstance(response, fastapi.Response)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_invalid_json_body(
    openai_factory,
    request_factory,
    request_headers,
):
    """Test invalid JSON in request body."""
    request = request_factory(
        request_url="http://0.0.0.0:5052/v1/proxy/openai/v1/chat/completions",
        request_body=b"invalid json",
        request_headers=request_headers,
    )

    with pytest.raises(fastapi.HTTPException) as excinfo:
        await openai_factory.factory(request)

    assert excinfo.value.status_code == status.HTTP_400_BAD_REQUEST
    assert excinfo.value.detail == "Invalid JSON"
