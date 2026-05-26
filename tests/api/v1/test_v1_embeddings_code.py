# pylint: disable=file-naming-for-tests
from json import JSONDecodeError
from typing import Any
from unittest.mock import AsyncMock, patch

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


@pytest.fixture(name="config_values")
def config_values_fixture():
    return {
        "custom_models": {"enabled": True},
    }


@pytest.fixture(name="config_values_custom_models_disabled")
def config_values_custom_models_disabled_fixture(config_values, monkeypatch):
    monkeypatch.setitem(config_values["custom_models"], "enabled", False)
    return config_values


class BaseTestCodeEmbeddings:
    def _build_params(
        self,
        model_provider: str,
        model_identifier: str | None,
        model_name: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        dimensions: int | None = None,
    ):
        model_metadata = {"provider": model_provider}

        if model_identifier:
            model_metadata["identifier"] = model_identifier

        if model_name:
            model_metadata["name"] = model_name

        if endpoint:
            model_metadata["endpoint"] = endpoint

        if api_key:
            model_metadata["api_key"] = api_key

        params: dict[str, Any] = {
            "model_metadata": model_metadata,
            "contents": [
                "test content 1",
                "test content 2",
            ],
        }

        if dimensions:
            params["dimensions"] = dimensions

        return params

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
        ("model_identifier", "expected_llm_model", "expected_custom_llm_provider"),
        [("text_embedding_005_vertex", "text-embedding-005", "vertex_ai")],
    )
    def test_successful_response_provider_gitlab(
        self,
        model_identifier,
        expected_llm_model,
        expected_custom_llm_provider,
        mock_client: TestClient,
        mock_litellm_aembedding: AsyncMock,
        mock_litellm_aembedding_response: AsyncMock,
    ):
        params = self._build_params(
            model_provider="gitlab", model_identifier=model_identifier
        )
        response = self._post_request(mock_client=mock_client, params=params)

        assert response.status_code == 200

        mock_litellm_aembedding.assert_called_once()
        call_kwargs = mock_litellm_aembedding.call_args[1]

        assert call_kwargs["input"] == params["contents"]
        assert "dimensions" not in call_kwargs

        assert call_kwargs["model"] == expected_llm_model
        assert call_kwargs["custom_llm_provider"] == expected_custom_llm_provider

        response_json = response.json()
        assert response_json["model"] == {
            "engine": "litellm_embedding",
            "name": "text-embedding-005",
            "identifier": "text_embedding_005_vertex",
        }
        assert response_json["predictions"] == mock_litellm_aembedding_response.data

    # The `litellm` provider is used for self-hosted models
    @pytest.mark.parametrize(
        (
            "model_identifier",
            "api_key",
            "expected_custom_llm_provider",
            "expected_llm_model",
            "expected_api_key",
        ),
        [
            (
                "openai/test-embedding-002",
                None,
                "openai",
                "test-embedding-002",
                None,
            ),
            (
                "openai/test-embedding-002",
                "test-api-key",
                "openai",
                "test-embedding-002",
                "test-api-key",
            ),
            (
                "test-embedding-001",
                None,
                "custom_openai",
                "test-embedding-001",
                "dummy_key",  # if custom_llm_provider=custom_openai, a dummy key is set
            ),
            (
                "test-embedding-001",
                "test-api-key",
                "custom_openai",
                "test-embedding-001",
                "test-api-key",
            ),
        ],
    )
    def test_successful_response_provider_litellm(
        self,
        model_identifier,
        api_key,
        expected_custom_llm_provider,
        expected_llm_model,
        expected_api_key,
        mock_client: TestClient,
        mock_litellm_aembedding: AsyncMock,
        mock_litellm_aembedding_response: AsyncMock,
    ):
        params = self._build_params(
            model_provider="litellm",
            model_name="embedding",
            model_identifier=model_identifier,
            endpoint="http://test",
            api_key=api_key,
        )
        response = self._post_request(mock_client=mock_client, params=params)

        assert response.status_code == 200

        mock_litellm_aembedding.assert_called_once()
        call_kwargs = mock_litellm_aembedding.call_args[1]

        assert call_kwargs["input"] == params["contents"]
        assert "dimensions" not in call_kwargs

        assert call_kwargs["custom_llm_provider"] == expected_custom_llm_provider
        assert call_kwargs["model"] == expected_llm_model
        assert call_kwargs["api_base"] == "http://test"

        if expected_api_key:
            assert call_kwargs["api_key"] == expected_api_key
        else:
            assert "api_key" not in call_kwargs

        response_json = response.json()
        assert response_json["model"] == {
            "engine": "litellm_embedding",
            "name": "embedding",
            "identifier": model_identifier,
        }
        assert response_json["predictions"] == mock_litellm_aembedding_response.data

    @pytest.mark.parametrize(
        ("model_provider", "model_identifier", "model_name", "endpoint"),
        [
            ("gitlab", "text_embedding_005_vertex", None, None),
            ("litellm", "openai/test-embedding-001", "embedding", "http://test"),
            ("litellm", "test-embedding-001", "embedding", "http://test"),
        ],
    )
    def test_successful_response_with_dimensions(
        self,
        model_provider,
        model_identifier,
        model_name,
        endpoint,
        mock_client: TestClient,
        mock_litellm_aembedding: AsyncMock,
        mock_litellm_aembedding_response: AsyncMock,
    ):
        params = self._build_params(
            model_provider=model_provider,
            model_name=model_name,
            model_identifier=model_identifier,
            endpoint=endpoint,
            dimensions=1024,
        )
        response = self._post_request(mock_client=mock_client, params=params)

        assert response.status_code == 200

        mock_litellm_aembedding.assert_called_once()
        call_kwargs = mock_litellm_aembedding.call_args[1]
        assert call_kwargs["input"] == params["contents"]
        assert call_kwargs["dimensions"] == 1024

        response_json = response.json()
        assert response_json["predictions"] == mock_litellm_aembedding_response.data

    def test_successful_response_with_drop_params(
        self, mock_client, mock_litellm_aembedding, mock_litellm_aembedding_response
    ):
        params = self._build_params(
            model_provider="gitlab",
            model_identifier="text_embedding_005_vertex",
            dimensions=1024,
        )
        params["litellm_drop_params"] = True
        response = self._post_request(mock_client=mock_client, params=params)

        mock_litellm_aembedding.assert_called_once()
        call_kwargs = mock_litellm_aembedding.call_args[1]
        assert call_kwargs["input"] == params["contents"]
        assert call_kwargs["dimensions"] == 1024
        assert call_kwargs["drop_params"] is True

        response_json = response.json()
        assert response_json["predictions"] == mock_litellm_aembedding_response.data

    def test_successful_response_custom_provider_vertex(
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

        assert response.status_code == 422
        assert response.json() == {"detail": "Allowed providers are: gitlab|litellm."}

    # The model `identifier` must always be given for `provider=gitlab|litellm`
    # If we need to support other providers and allow `identifier`
    #   to be None, further validation must be added to require
    #   `identifier` to be set if `provider=gitlab|litellm`
    @pytest.mark.parametrize(
        ("model_provider"),
        [("gitlab"), ("litellm")],
    )
    def test_missing_model_identifier(
        self,
        model_provider,
        mock_client: TestClient,
    ):
        params = self._build_params(
            model_provider=model_provider, model_identifier=None
        )
        response = self._post_request(mock_client=mock_client, params=params)

        assert response.status_code == 422

        error_detail = response.json()["detail"][0]
        assert error_detail["loc"] == ["body", "model_metadata", "identifier"]
        assert error_detail["msg"] == "Field required"

    @pytest.mark.parametrize(
        ("model_name", "model_endpoint", "expected_error_message"),
        [
            (
                None,
                "http://local-endpoint",
                "Model `name` must be set when using `litellm` provider.",
            ),
            (
                "some-name",
                "http://local-endpoint",
                "Model `name` must be 'embedding' when using `litellm` provider.",
            ),
            (
                "embedding",
                None,
                "Model `endpoint` must be set when using `litellm` provider.",
            ),
        ],
    )
    def test_invalid_payload_provider_litellm(
        self,
        model_name,
        model_endpoint,
        expected_error_message,
        mock_client: TestClient,
    ):
        params = self._build_params(
            model_provider="litellm",
            model_identifier="open_ai/test-embedding-model",
            model_name=model_name,
            endpoint=model_endpoint,
        )
        response = self._post_request(mock_client=mock_client, params=params)

        assert response.status_code == 422
        assert response.json() == {"detail": expected_error_message}

    @pytest.mark.usefixtures("config_values_custom_models_disabled")
    def test_endpoint_provided_with_custom_models_disabled(
        self,
        mock_client: TestClient,
    ):
        params = self._build_params(
            model_provider="litellm",
            model_name="embedding",
            model_identifier="openai/test-embedding-model",
            endpoint="http://endpoint",
        )
        response = self._post_request(mock_client=mock_client, params=params)

        assert response.status_code == 422
        assert (
            response.json()["detail"]
            == "specifying custom models endpoint is disabled: api_base is not allowed"
        )

    @pytest.mark.parametrize(
        ("error_class", "error_args", "expected_error_message"),
        [
            (ValueError, ["some error"], "some error"),
            (
                JSONDecodeError,
                ["some error", "doc", 1],
                "some error: line 1 column 2 (char 1)",
            ),
            (
                UnicodeDecodeError,
                ["utf-8", b"", 0, 1, "some error"],
                "'utf-8' codec can't decode bytes in position 0-0: some error",
            ),
        ],
    )
    def test_create_model_metadata_error(
        self,
        error_class,
        error_args,
        expected_error_message,
        mock_client: TestClient,
    ):
        params = self._build_params(
            model_provider="litellm",
            model_identifier="open_ai/test-embedding-model",
            model_name="embedding",
            endpoint="http://local",
        )

        with patch(
            "ai_gateway.api.v1.embeddings.code_embeddings.create_model_metadata",
            side_effect=error_class(*error_args),
        ):
            response = self._post_request(mock_client=mock_client, params=params)

            assert response.status_code == 422

            response_error_message = (
                f"Error creating the model_metadata: {expected_error_message}"
            )
            assert response.json()["detail"] == response_error_message

    def test_no_model_metadata_created(
        self,
        mock_client: TestClient,
    ):
        params = self._build_params(
            model_provider="litellm",
            model_identifier="open_ai/test-embedding-model",
            model_name="embedding",
            endpoint="http://local",
        )

        with patch(
            "ai_gateway.api.v1.embeddings.code_embeddings.create_model_metadata",
            return_value=None,
        ):
            response = self._post_request(mock_client=mock_client, params=params)

            assert response.json()["detail"] == "No model metadata created"

            assert response.status_code == 422

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

    def test_authentication_error(
        self,
        mock_client: TestClient,
        mock_litellm_aembedding: AsyncMock,
    ):
        error_message = "Authentication error from litellm"
        mock_litellm_aembedding.side_effect = litellm.AuthenticationError(
            message=error_message, model="test-embedding-model", llm_provider="openai"
        )

        params = self._build_params(
            model_provider="gitlab", model_identifier="text_embedding_005_vertex"
        )
        response = self._post_request(mock_client=mock_client, params=params)

        assert response.status_code == 401
        assert response.json() == {
            "detail": f"litellm.AuthenticationError: {error_message}"
        }


class TestCodeEmbeddingsIndex(BaseTestCodeEmbeddings):
    def _route(self):
        return "/embeddings/code_embeddings/index"

    @pytest.mark.usefixtures(
        "mock_litellm_aembedding", "mock_litellm_aembedding_response"
    )
    def test_successful_response_base_route(
        self,
        mock_client: TestClient,
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
