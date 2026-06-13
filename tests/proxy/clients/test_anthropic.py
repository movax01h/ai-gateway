import json
from unittest.mock import patch

import fastapi
import pytest
from fastapi import status

from ai_gateway.model_selection import ModelSelectionConfig
from ai_gateway.proxy.clients import AnthropicProxyModelFactory, ProxyClient
from ai_gateway.proxy.clients.anthropic import _resolve_api_key


@pytest.fixture(name="anthropic_factory")
def anthropic_factory_fixture(monkeypatch):
    """Fixture to create an AnthropicProxyModelFactory instance."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return AnthropicProxyModelFactory()


@pytest.fixture(name="proxy_client")
def proxy_client_fixture(limits, internal_event_client, billing_event_service):
    """Fixture to create a ProxyClient instance."""
    return ProxyClient(limits, internal_event_client, billing_event_service)


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
@pytest.mark.usefixtures("mock_proxy_async_client")
async def test_valid_proxy_requests(
    anthropic_factory,
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

    model = await anthropic_factory.factory(request)
    response = await proxy_client.proxy(request, model)

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
    anthropic_factory,
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
        await anthropic_factory.factory(request)

    assert excinfo.value.status_code == expected_status
    assert excinfo.value.detail == expected_detail


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_proxy_async_client")
async def test_count_tokens_endpoint(
    anthropic_factory,
    proxy_client,
    request_factory,
    request_headers,
):
    count_tokens_request_body = {
        "model": "claude-haiku-4-5-20251001",
        "messages": [{"role": "user", "content": "Hi, how are you?"}],
    }

    request = request_factory(
        request_url="http://0.0.0.0:5052/v1/proxy/anthropic/v1/messages/count_tokens?beta=true",
        request_body=json.dumps(count_tokens_request_body).encode("utf-8"),
        request_headers=request_headers,
    )

    model = await anthropic_factory.factory(request)
    response = await proxy_client.proxy(request, model)

    assert isinstance(response, fastapi.Response)
    assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_headers,proxy_proxy_upstream_response_headers,expected_upstream_headers,expected_downstream_headers",
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
    mock_proxy_async_client,
    internal_event_client,
    billing_event_service,
    limits,
    monkeypatch,
    request_factory,
    request_params,
    request_headers,
    proxy_proxy_upstream_response_headers,
    expected_upstream_headers,
    expected_downstream_headers,
):
    """Test anthropic-beta header handling in both directions (upstream and downstream)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    anthropic_factory = AnthropicProxyModelFactory()
    proxy_client = ProxyClient(limits, internal_event_client, billing_event_service)

    # Make request
    request = request_factory(
        request_url="http://0.0.0.0:5052/v1/proxy/anthropic/v1/messages",
        request_body=json.dumps(request_params).encode("utf-8"),
        request_headers=request_headers,
    )

    model = await anthropic_factory.factory(request)
    response = await proxy_client.proxy(request, model)

    # Verify upstream request headers
    mock_proxy_async_client.send.assert_called_once()
    upstream_request = mock_proxy_async_client.send.call_args[0][0]

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
    if "x-custom-header" in proxy_proxy_upstream_response_headers:
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
    def anthropic_factory_for_usage(
        self,
        monkeypatch,
    ):
        """Fixture to create an AnthropicProxyModelFactory instance for usage tests."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        return AnthropicProxyModelFactory()


class TestAnthropicApiKeyResolution:
    """Test cases for per-model Anthropic API key resolution."""

    def test_resolves_per_model_api_key(self, monkeypatch):
        """A model with a configured api_key override uses that key."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        monkeypatch.setattr(ModelSelectionConfig, "_instance", None)
        config = ModelSelectionConfig(
            default_models_override={},
            model_params_override={
                "claude_haiku_4_5_20251001": {"api_key": "fable-workspace-key"}
            },
        )

        with patch.object(ModelSelectionConfig, "instance", return_value=config):
            assert (
                _resolve_api_key("claude-haiku-4-5-20251001") == "fable-workspace-key"
            )

    def test_falls_back_to_env_when_no_override(self, monkeypatch):
        """A model without an api_key override falls back to ANTHROPIC_API_KEY."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        monkeypatch.setattr(ModelSelectionConfig, "_instance", None)
        config = ModelSelectionConfig(default_models_override={})

        with patch.object(ModelSelectionConfig, "instance", return_value=config):
            assert _resolve_api_key("claude-haiku-4-5-20251001") == "env-key"

    def test_empty_per_model_api_key_falls_back_to_env(self, monkeypatch):
        """An empty-string api_key override falls back to ANTHROPIC_API_KEY."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        monkeypatch.setattr(ModelSelectionConfig, "_instance", None)
        config = ModelSelectionConfig(
            default_models_override={},
            model_params_override={"claude_haiku_4_5_20251001": {"api_key": ""}},
        )

        with patch.object(ModelSelectionConfig, "instance", return_value=config):
            assert _resolve_api_key("claude-haiku-4-5-20251001") == "env-key"

    def test_raises_when_no_key_available(self, monkeypatch):
        """Missing both per-model key and env var raises HTTP 400."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(ModelSelectionConfig, "_instance", None)
        config = ModelSelectionConfig(default_models_override={})

        with patch.object(ModelSelectionConfig, "instance", return_value=config):
            with pytest.raises(fastapi.HTTPException) as excinfo:
                _resolve_api_key("claude-haiku-4-5-20251001")

        assert excinfo.value.status_code == status.HTTP_400_BAD_REQUEST
        assert excinfo.value.detail == "API key not found"

    @pytest.mark.asyncio
    async def test_factory_forwards_per_model_api_key_to_upstream(
        self,
        mock_proxy_async_client,
        monkeypatch,
        limits,
        internal_event_client,
        billing_event_service,
        request_factory,
        request_headers,
    ):
        """The per-model api_key is forwarded as x-api-key to the upstream request."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        monkeypatch.setattr(ModelSelectionConfig, "_instance", None)
        config = ModelSelectionConfig(
            default_models_override={},
            model_params_override={
                "claude_haiku_4_5_20251001": {"api_key": "fable-workspace-key"}
            },
        )

        request_params = {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }
        request = request_factory(
            request_url="http://0.0.0.0:5052/v1/proxy/anthropic/v1/messages",
            request_body=json.dumps(request_params).encode("utf-8"),
            request_headers=request_headers,
        )

        with patch.object(ModelSelectionConfig, "instance", return_value=config):
            factory = AnthropicProxyModelFactory()
            proxy_client = ProxyClient(
                limits, internal_event_client, billing_event_service
            )
            model = await factory.factory(request)
            assert model.headers_to_upstream["x-api-key"] == "fable-workspace-key"

            await proxy_client.proxy(request, model)

        upstream_request = mock_proxy_async_client.send.call_args[0][0]
        assert upstream_request.headers["x-api-key"] == "fable-workspace-key"
