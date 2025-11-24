import json
from unittest.mock import ANY, patch

import fastapi
import pytest
from fastapi import status
from starlette.datastructures import URL

from ai_gateway.proxy.clients.token_usage import TokenUsage
from ai_gateway.proxy.clients.vertex_ai import VertexAIProxyClient


@pytest.mark.asyncio
async def test_valid_proxy_request_text_embedding(
    async_client_factory, limits, request_factory
):
    proxy_client = VertexAIProxyClient(
        project="",
        location="",
        client=async_client_factory(),
        limits=limits,
    )

    request_params = {
        "instances": [{"content": "Hello world"}],
    }

    with patch("ai_gateway.proxy.clients.vertex_ai.access_token") as mock_access_token:
        response = await proxy_client.proxy(
            request_factory(
                request_url=(
                    "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
                    "locations/LOCATION/publishers/google/models/text-embedding-005:predict"
                ),
                request_body=json.dumps(request_params).encode("utf-8"),
                request_headers={
                    "content-type": "application/json",
                },
            )
        )

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
            "/v1/projects/my-project/locations/my-location/publishers/"
            "google/models/text-embedding-005:predict",
            None,
        ),
        (
            "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
            "locations/LOCATION/publishers/google/models/text-embedding-005:predict?alt=sse",
            "/v1/projects/my-project/locations/my-location/publishers/"
            "google/models/text-embedding-005:predict?alt=sse",
            None,
        ),
        (
            "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/"
            "locations/LOCATION/publishers/google/models/text-embedding-005:predict?alt=unknown",
            "/v1/projects/my-project/locations/my-location/publishers/"
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
    async_client_factory,
    limits,
    request_factory,
    request_url,
    expected_upstream_path,
    expected_error,
):
    async_client = async_client_factory()
    proxy_client = VertexAIProxyClient(
        project="my-project",
        location="my-location",
        client=async_client,
        limits=limits,
    )

    if expected_error:
        with pytest.raises(fastapi.HTTPException, match=expected_error):
            await proxy_client.proxy(request_factory(request_url=request_url))
    else:
        with patch(
            "ai_gateway.proxy.clients.vertex_ai.access_token"
        ) as mock_access_token:
            response = await proxy_client.proxy(
                request_factory(request_url=request_url)
            )

            mock_access_token.assert_called_once()

        assert response.status_code == 200

        async_client.build_request.assert_called_once_with(
            "POST",
            URL(expected_upstream_path),
            headers=ANY,
            json=ANY,
        )


class TestVertexAITokenUsageExtraction:
    """Test cases for Vertex AI token usage extraction."""

    @pytest.fixture
    def vertex_client_for_usage(self, async_client_factory, limits):
        """Fixture to create a VertexAIProxyClient instance for usage tests."""
        return VertexAIProxyClient(
            project="test-project",
            location="us-central1",
            client=async_client_factory(),
            limits=limits,
        )

    @pytest.fixture
    def test_path(self):
        """Standard test path for usage extraction."""
        return "/v1/projects/test/locations/us/publishers/google/models/text-embedding-005:predict"

    @pytest.mark.parametrize(
        "response_body,expected_input,expected_output,expected_total",
        [
            (
                {
                    "predictions": [{"embeddings": {"values": [0.1, 0.2, 0.3]}}],
                    "metadata": {
                        "tokenMetadata": {
                            "inputTokenCount": {
                                "totalTokens": 10,
                                "totalBillableCharacters": 50,
                            },
                            "outputTokenCount": {
                                "totalTokens": 20,
                                "totalBillableCharacters": 100,
                            },
                        }
                    },
                },
                10,
                20,
                30,
            ),
            ({"predictions": [{"embeddings": {"values": [0.1, 0.2]}}]}, 0, 0, 0),
            (
                {
                    "predictions": [{"embeddings": {"values": [0.1, 0.2]}}],
                    "metadata": {},
                },
                0,
                0,
                0,
            ),
            (
                {
                    "metadata": {
                        "tokenMetadata": {
                            "inputTokenCount": {"totalTokens": 0},
                            "outputTokenCount": {"totalTokens": 0},
                        }
                    }
                },
                0,
                0,
                0,
            ),
            (
                {
                    "metadata": {
                        "tokenMetadata": {
                            "inputTokenCount": {"totalTokens": 100},
                        }
                    }
                },
                100,
                0,
                100,
            ),
            (
                {
                    "metadata": {
                        "tokenMetadata": {
                            "outputTokenCount": {"totalTokens": 50},
                        }
                    }
                },
                0,
                50,
                50,
            ),
            (
                {
                    "metadata": {
                        "tokenMetadata": {
                            "inputTokenCount": {
                                "totalTokens": 10,
                                "totalBillableCharacters": 999,
                            },
                            "outputTokenCount": {
                                "totalTokens": 20,
                                "totalBillableCharacters": 888,
                            },
                        }
                    }
                },
                10,
                20,
                30,
            ),
        ],
        ids=[
            "Complete response with both input and output tokens",
            "Missing metadata",
            "Missing tokenMetadata",
            "Zero tokens",
            "Only input tokens",
            "Only output tokens",
            "Billable characters should be ignored",
        ],
    )
    def test_extract_token_usage(
        self,
        vertex_client_for_usage,
        test_path,
        response_body,
        expected_input,
        expected_output,
        expected_total,
    ):
        """Test extracting token usage from various Vertex AI responses."""
        usage = vertex_client_for_usage._extract_token_usage(test_path, response_body)

        assert isinstance(usage, TokenUsage)
        assert usage.input_tokens == expected_input
        assert usage.output_tokens == expected_output
        assert usage.total_tokens == expected_total

    def test_extract_token_usage_embedding_model(self, vertex_client_for_usage):
        response_body = {
            "predictions": [
                {
                    "embeddings": {
                        "statistics": {
                            "token_count": 13,
                            "truncated": False,
                        },
                        "values": [0.1, 0.2, 0.3],
                    }
                }
            ],
            "metadata": {"billableCharacterCount": 45},
        }

        usage = vertex_client_for_usage._extract_token_usage(
            "/v1/projects/test/locations/us/publishers/google/models/text-embedding-005:predict",
            response_body,
        )

        assert usage.input_tokens == 13
        assert usage.output_tokens == 0
        assert usage.total_tokens == 13

    def test_extract_token_usage_streaming_endpoint(self, vertex_client_for_usage):
        """Test extracting token usage from streaming endpoint."""
        response_body = {
            "metadata": {
                "tokenMetadata": {
                    "inputTokenCount": {"totalTokens": 15},
                    "outputTokenCount": {"totalTokens": 25},
                }
            }
        }

        usage = vertex_client_for_usage._extract_token_usage(
            (
                "/v1/projects/test/locations/us/publishers/google/models/"
                "text-embedding-005:serverStreamingPredict"
            ),
            response_body,
        )

        assert usage.input_tokens == 15
        assert usage.output_tokens == 25
        assert usage.total_tokens == 40

    @pytest.mark.parametrize(
        "response_body",
        [
            None,
            {"metadata": None},
            {"metadata": {"tokenMetadata": None}},
            {"metadata": {"tokenMetadata": "invalid"}},
        ],
    )
    def test_extract_token_usage_invalid_response(
        self, vertex_client_for_usage, test_path, response_body
    ):
        """Test extracting token usage from invalid response raises exception."""
        with pytest.raises(fastapi.HTTPException) as excinfo:
            vertex_client_for_usage._extract_token_usage(test_path, response_body)

        assert excinfo.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to extract token usage" in excinfo.value.detail

    def test_allowed_upstream_models_includes_anthropic(self, vertex_client_for_usage):
        """Test allowed models include text-embeddings and Claude models."""
        allowed_models = vertex_client_for_usage._allowed_upstream_models()

        assert "text-embedding-005" in allowed_models

        assert "claude-sonnet-4-5@20250929" in allowed_models
