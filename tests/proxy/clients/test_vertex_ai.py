import json
from unittest.mock import ANY, patch

import fastapi
import pytest
from starlette.datastructures import URL

from ai_gateway.proxy.clients import ProxyClient, VertexAIProxyModelFactory
from ai_gateway.proxy.clients.vertex_ai import PathParams


@pytest.fixture(name="vertex_factory")
def vertex_factory_fixture():
    """Fixture to create a VertexAIProxyModelFactory instance for usage tests."""
    return VertexAIProxyModelFactory(
        endpoint="my-location-aiplatform.googleapis.com",
        project="my-project",
        location="my-location",
    )


@pytest.fixture(name="proxy_client")
def proxy_client_fixture(limits, internal_event_client, billing_event_client):
    """Fixture to create a ProxyClient instance."""
    return ProxyClient(limits, internal_event_client, billing_event_client)


@pytest.mark.asyncio
async def test_valid_proxy_request_text_embedding(
    vertex_factory, proxy_client, request_factory
):
    request_params = {
        "instances": [{"content": "Hello world"}],
    }

    with patch("ai_gateway.proxy.clients.vertex_ai.access_token") as mock_access_token:
        request = request_factory(
            request_url=(
                "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
                "locations/LOCATION/publishers/google/models/text-embedding-005:predict"
            ),
            request_body=json.dumps(request_params).encode("utf-8"),
            request_headers={
                "content-type": "application/json",
            },
        )

        model = await vertex_factory.factory(request)
        response = await proxy_client.proxy(request, model)

        mock_access_token.assert_called_once()

    assert isinstance(response, fastapi.Response)
    assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_url", "expected_upstream_path", "expected_error"),
    [
        (
            "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
            "locations/LOCATION/publishers/google/models/text-embedding-005:predict",
            "https://my-location-aiplatform.googleapis.com/v1/projects/my-project/locations/my-location/publishers/"
            "google/models/text-embedding-005:predict",
            None,
        ),
        (
            "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
            "locations/LOCATION/publishers/google/models/text-embedding-005:predict?alt=sse",
            "https://my-location-aiplatform.googleapis.com/v1/projects/my-project/locations/my-location/publishers/"
            "google/models/text-embedding-005:predict?alt=sse",
            None,
        ),
        (
            "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
            "locations/LOCATION/publishers/google/models/text-embedding-005:predict?alt=unknown",
            "https://my-location-aiplatform.googleapis.com/v1/projects/my-project/locations/my-location/publishers/"
            "google/models/text-embedding-005:predict",
            None,
        ),
        (
            "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
            "locations/LOCATION/publishers/google/models/unknown:predict",
            "",
            "400: Unsupported model",
        ),
        (
            "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
            "locations/LOCATION/publishers/google/models/text-embedding-005:unknown",
            "",
            "404: Not found",
        ),
        (
            "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/unknown/PROJECT/"
            "locations/LOCATION/publishers/google/models/text-embedding-005:predict",
            "",
            "404: Not found",
        ),
        (
            "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
            "unknown/LOCATION/publishers/google/models/text-embedding-005:predict",
            "",
            "404: Not found",
        ),
        (
            "http://0.0.0.0:5052/v1/proxy/vertex-ai/corrupted-path/"
            "text-embedding-005:predict",
            "",
            "404: Not found",
        ),
    ],
)
async def test_request_url(
    async_client,
    vertex_factory,
    proxy_client,
    request_factory,
    request_url,
    expected_upstream_path,
    expected_error,
):

    if expected_error:
        with pytest.raises(fastapi.HTTPException, match=expected_error):
            request = request_factory(request_url=request_url)
            await vertex_factory.factory(request)
    else:
        with patch(
            "ai_gateway.proxy.clients.vertex_ai.access_token"
        ) as mock_access_token:
            request = request_factory(request_url=request_url)
            model = await vertex_factory.factory(request)
            response = await proxy_client.proxy(request, model)

            mock_access_token.assert_called_once()

        assert response.status_code == 200

        async_client.request.assert_called_once_with(
            method="POST",
            url=URL(expected_upstream_path),
            headers=ANY,
            params=ANY,
            json=ANY,
        )


def test_allowed_upstream_models_includes_anthropic():
    """Test allowed models include text-embeddings and Claude models."""
    vertex_factory = VertexAIProxyModelFactory(
        endpoint="test-endpoint",
        project="test-project",
        location="test-location",
    )

    # Access the private method for testing
    from ai_gateway.proxy.clients.vertex_ai import _load_allowed_upstream_models

    allowed_models = _load_allowed_upstream_models()

    assert "text-embedding-005" in allowed_models
    assert "claude-sonnet-4-5@20250929" in allowed_models


class TestPathParams:
    """Test PathParams validation and extraction."""

    @pytest.mark.parametrize(
        ("request_url", "expected_params"),
        [
            # Google models
            (
                "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
                "locations/LOCATION/publishers/google/models/text-embedding-005:predict",
                {
                    "upstream_provider": "google",
                    "model_name": "text-embedding-005",
                    "action": "predict",
                    "sse": "",
                },
            ),
            (
                "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
                "locations/LOCATION/publishers/google/models/gemini-2.5-flash:generateContent",
                {
                    "upstream_provider": "google",
                    "model_name": "gemini-2.5-flash",
                    "action": "generateContent",
                    "sse": "",
                },
            ),
            (
                "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
                "locations/LOCATION/publishers/google/models/gemini-2.5-flash:streamGenerateContent",
                {
                    "upstream_provider": "google",
                    "model_name": "gemini-2.5-flash",
                    "action": "streamGenerateContent",
                    "sse": "",
                },
            ),
            (
                "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
                "locations/LOCATION/publishers/google/models/codestral-2501:streamRawPredict",
                {
                    "upstream_provider": "google",
                    "model_name": "codestral-2501",
                    "action": "streamRawPredict",
                    "sse": "",
                },
            ),
            (
                "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
                "locations/LOCATION/publishers/anthropic/models/claude-sonnet-4@20250514:streamRawPredict",
                {
                    "upstream_provider": "anthropic",
                    "model_name": "claude-sonnet-4@20250514",
                    "action": "streamRawPredict",
                    "sse": "",
                },
            ),
            # With SSE flag
            (
                "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
                "locations/LOCATION/publishers/google/models/text-embedding-005:predict?alt=sse",
                {
                    "upstream_provider": "google",
                    "model_name": "text-embedding-005",
                    "action": "predict",
                    "sse": "?alt=sse",
                },
            ),
            # Model names with dots and hyphens
            (
                "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/my-project-123/"
                "locations/us-central1/publishers/google/models/gemini-1.5-flash-002:serverStreamingPredict",
                {
                    "upstream_provider": "google",
                    "model_name": "gemini-1.5-flash-002",
                    "action": "serverStreamingPredict",
                    "sse": "",
                },
            ),
        ],
    )
    def test_path_params_extraction_success(
        self, request_factory, request_url, expected_params
    ):
        """Test successful extraction of path parameters from various URL formats."""
        request = request_factory(request_url=request_url)
        path_params = PathParams.try_from_request(request)

        assert path_params.upstream_provider == expected_params["upstream_provider"]
        assert path_params.model_name == expected_params["model_name"]
        assert path_params.action == expected_params["action"]
        assert path_params.sse == expected_params["sse"]

    @pytest.mark.parametrize(
        "request_url",
        [
            # Invalid action
            "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
            "locations/LOCATION/publishers/google/models/text-embedding-005:invalidAction",
            # Invalid provider
            "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
            "locations/LOCATION/publishers/unknown/models/text-embedding-005:predict",
            # Malformed path
            "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/corrupted-path/text-embedding-005:predict",
            # Missing model name
            "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
            "locations/LOCATION/publishers/google/models/:predict",
            # Missing action
            "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
            "locations/LOCATION/publishers/google/models/text-embedding-005",
        ],
    )
    def test_path_params_extraction_failure(self, request_factory, request_url):
        """Test that invalid URLs raise HTTPException with 404."""
        request = request_factory(request_url=request_url)
        with pytest.raises(fastapi.HTTPException, match="404: Not found"):
            PathParams.try_from_request(request)


class TestStreamFlagDetection:
    """Test stream flag detection for different providers and actions."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("request_url", "request_body", "expected_stream"),
        [
            # Google models - stream based on action
            (
                "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
                "locations/LOCATION/publishers/google/models/text-embedding-005:predict",
                {"instances": [{"content": "test"}]},
                False,
            ),
            (
                "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
                "locations/LOCATION/publishers/google/models/text-embedding-005:serverStreamingPredict",
                {"instances": [{"content": "test"}]},
                True,
            ),
            (
                "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
                "locations/LOCATION/publishers/google/models/gemini-2.5-flash:streamGenerateContent",
                {"contents": [{"role": "user", "parts": [{"text": "test"}]}]},
                True,
            ),
            (
                "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
                "locations/LOCATION/publishers/google/models/gemini-2.5-flash:generateContent",
                {"contents": [{"role": "user", "parts": [{"text": "test"}]}]},
                False,
            ),
            # Anthropic models - stream based on request body
            (
                "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
                "locations/LOCATION/publishers/anthropic/models/claude-sonnet-4-5@20250929:streamRawPredict",
                {
                    "anthropic_version": "vertex-2023-10-16",
                    "messages": [{"role": "user", "content": "test"}],
                    "stream": True,
                    "max_tokens": 1024,
                },
                True,
            ),
            (
                "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
                "locations/LOCATION/publishers/anthropic/models/claude-sonnet-4-5@20250929:streamRawPredict",
                {
                    "anthropic_version": "vertex-2023-10-16",
                    "messages": [{"role": "user", "content": "test"}],
                    "stream": False,
                    "max_tokens": 1024,
                },
                False,
            ),
        ],
    )
    async def test_stream_flag_detection(
        self,
        vertex_factory,
        request_factory,
        request_url,
        request_body,
        expected_stream,
    ):
        """Test stream flag is correctly detected for different providers and actions."""
        with patch("ai_gateway.proxy.clients.vertex_ai.access_token"):
            request = request_factory(
                request_url=request_url,
                request_body=json.dumps(request_body).encode("utf-8"),
                request_headers={"content-type": "application/json"},
            )

            model = await vertex_factory.factory(request)
            assert model.stream == expected_stream


class TestUpstreamPathGeneration:
    """Test upstream path generation for different providers."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("request_url", "expected_upstream_path_suffix"),
        [
            # Google models
            (
                "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
                "locations/LOCATION/publishers/google/models/text-embedding-005:predict",
                "/publishers/google/models/text-embedding-005:predict",
            ),
            (
                "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
                "locations/LOCATION/publishers/google/models/gemini-2.5-flash:streamGenerateContent",
                "/publishers/google/models/gemini-2.5-flash:streamGenerateContent",
            ),
            # Anthropic models
            (
                "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
                "locations/LOCATION/publishers/anthropic/models/claude-sonnet-4-5@20250929:streamRawPredict",
                "/publishers/anthropic/models/claude-sonnet-4-5@20250929:streamRawPredict",
            ),
            # With SSE flag
            (
                "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
                "locations/LOCATION/publishers/google/models/text-embedding-005:predict?alt=sse",
                "/publishers/google/models/text-embedding-005:predict?alt=sse",
            ),
        ],
    )
    async def test_upstream_path_generation(
        self,
        vertex_factory,
        request_factory,
        request_url,
        expected_upstream_path_suffix,
    ):
        """Test that upstream paths are correctly generated for different providers."""
        with patch("ai_gateway.proxy.clients.vertex_ai.access_token"):
            request_body = {"test": "data"}
            request = request_factory(
                request_url=request_url,
                request_body=json.dumps(request_body).encode("utf-8"),
                request_headers={"content-type": "application/json"},
            )

            model = await vertex_factory.factory(request)
            assert model.upstream_path.endswith(expected_upstream_path_suffix)
            assert model.upstream_path.startswith(
                "/v1/projects/my-project/locations/my-location"
            )
