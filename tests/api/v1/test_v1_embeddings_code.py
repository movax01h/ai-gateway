from typing import Any
from unittest.mock import AsyncMock

import litellm
import pytest
from fastapi.testclient import TestClient
from gitlab_cloud_connector import GitLabUnitPrimitive

from ai_gateway.api.v1 import api_router


@pytest.fixture(name="fast_api_router", scope="class")
def fast_api_router_fixture():
    return api_router


@pytest.fixture(name="unit_primitive")
def unit_primitive_fixture():
    return GitLabUnitPrimitive.GENERATE_EMBEDDINGS_CODEBASE


class BaseTestCodeEmbeddings:
    def _build_params(self, model_provider: str, model_identifier: str | None):
        return {
            "model_metadata": {
                "provider": model_provider,
                "identifier": model_identifier,
            },
            "contents": [
                "test content 1",
                "test content 2",
            ],
        }

    def _post_request(
        self, mock_client: TestClient, params: dict[str, Any], route: str | None = None
    ):
        route = route or self._route()
        return mock_client.post(
            route,
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
            },
            json=params,
        )

    def _route(self):
        raise NotImplementedError("Implement in child class.")

    @pytest.mark.parametrize(
        ("model_provider", "model_identifier", "expected_llm_model"),
        [("gitlab", "text_embedding_005_vertex", "text-embedding-005")],
    )
    def test_successful_response(
        self,
        model_provider,
        model_identifier,
        expected_llm_model,
        mock_client: TestClient,
        mock_litellm_aembedding: AsyncMock,
        mock_litellm_aembedding_response: AsyncMock,
    ):
        params = self._build_params(
            model_provider=model_provider, model_identifier=model_identifier
        )
        response = self._post_request(mock_client=mock_client, params=params)

        assert response.status_code == 200

        mock_litellm_aembedding.assert_called_once()
        call_kwargs = mock_litellm_aembedding.call_args[1]
        assert call_kwargs["model"] == expected_llm_model
        assert call_kwargs["input"] == params["contents"]

        response_json = response.json()
        assert response_json["model"] == {
            "engine": "litellm_embedding",
            "name": "text-embedding-005",
        }
        assert response_json["predictions"] == mock_litellm_aembedding_response.data

    def test_successful_response_vertex(
        self,
        mock_client: TestClient,
        mock_litellm_aembedding: AsyncMock,
    ):
        params = self._build_params(
            model_provider="gitlab", model_identifier="text_embedding_005_vertex"
        )
        response = self._post_request(mock_client=mock_client, params=params)

        assert response.status_code == 200

        mock_litellm_aembedding.assert_called_once()
        call_kwargs = mock_litellm_aembedding.call_args[1]
        assert call_kwargs["model"] == "text-embedding-005"
        assert call_kwargs["input"] == params["contents"]
        assert call_kwargs["custom_llm_provider"] == "vertex_ai"
        assert call_kwargs["vertex_ai_location"] == "global"

    def test_unsupported_provider(
        self,
        mock_client: TestClient,
    ):
        params = self._build_params(
            model_provider="not_gitlab", model_identifier="dummy_identifier"
        )
        response = self._post_request(mock_client=mock_client, params=params)

        assert response.status_code == 400
        assert response.json() == {
            "detail": "Only Gitlab-operated models are supported in this endpoint."
        }

    # The model `identifier` must always be given for `provider=gitlab`
    #   since we do not allow fallback models for this endpoint
    # If we need to support other providers and allow `identifier`
    #   to be None, further validation must be added to require
    #   `identifier` to be set if `provider=gitlab`
    def test_missing_model_identifier(
        self,
        mock_client: TestClient,
    ):
        params = self._build_params(model_provider="gitlab", model_identifier=None)
        response = self._post_request(mock_client=mock_client, params=params)

        assert response.status_code == 422

        error_detail = response.json()["detail"][0]
        assert error_detail["loc"] == ["body", "model_metadata", "identifier"]
        assert error_detail["msg"] == "Input should be a valid string"

    def test_bad_request_error(
        self,
        mock_client: TestClient,
        mock_litellm_aembedding: AsyncMock,
    ):
        error_message = "Bad request error from litellm"
        mock_litellm_aembedding.side_effect = litellm.BadRequestError(
            message=error_message, model="test-embedding-model", llm_provider="openai"
        )

        params = self._build_params(
            model_provider="gitlab", model_identifier="text_embedding_005_vertex"
        )
        response = self._post_request(mock_client=mock_client, params=params)

        assert response.status_code == 400
        assert response.json() == {
            "detail": f"litellm.BadRequestError: {error_message}"
        }

    def test_too_many_requests_error(
        self,
        mock_client: TestClient,
        mock_litellm_aembedding: AsyncMock,
    ):
        error_message = "Rate limit error from litellm"
        mock_litellm_aembedding.side_effect = litellm.RateLimitError(
            message=error_message, model="test-embedding-model", llm_provider="openai"
        )

        params = self._build_params(
            model_provider="gitlab", model_identifier="text_embedding_005_vertex"
        )
        response = self._post_request(mock_client=mock_client, params=params)

        assert response.status_code == 429
        assert response.json() == {"detail": f"litellm.RateLimitError: {error_message}"}


class TestCodeEmbeddingsIndex(BaseTestCodeEmbeddings):
    def _route(self):
        return "/embeddings/code_embeddings/index"

    def test_successful_response_base_route(
        self,
        mock_client: TestClient,
        mock_litellm_aembedding: AsyncMock,
        mock_litellm_aembedding_response: AsyncMock,
    ):
        params = self._build_params(
            model_provider="gitlab", model_identifier="text_embedding_005_vertex"
        )
        response = self._post_request(
            mock_client=mock_client, params=params, route="/embeddings/code_embeddings"
        )

        assert response.status_code == 200


class TestCodeEmbeddingsSearch(BaseTestCodeEmbeddings):
    def _route(self):
        return "/embeddings/code_embeddings/search"
