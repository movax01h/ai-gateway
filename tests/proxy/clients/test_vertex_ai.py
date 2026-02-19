import json
from unittest.mock import ANY, patch

import fastapi
import pytest
from starlette.datastructures import URL

from ai_gateway.proxy.clients import ProxyClient, VertexAIProxyModelFactory


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
