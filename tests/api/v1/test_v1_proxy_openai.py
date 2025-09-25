from unittest.mock import patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser, GitLabUnitPrimitive, UserClaims

from ai_gateway.api.v1 import api_router
from ai_gateway.api.v1.proxy.request import (
    EXTENDED_FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS,
)


@pytest.fixture(name="fast_api_router", scope="class")
def fast_api_router_fixture():
    return api_router


@pytest.fixture(name="auth_user")
def auth_user_fixture():
    return CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(
            scopes=EXTENDED_FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS.keys()
        ),
    )


class TestProxyOpenAI:
    @pytest.mark.parametrize(
        "unit_primitive", EXTENDED_FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS.keys()
    )
    def test_successful_chat_completions_request(
        self, mock_client, mock_track_internal_event, mock_detect_abuse, unit_primitive
    ):
        with patch(
            "ai_gateway.proxy.clients.OpenAIProxyClient.proxy",
            return_value={"response": "test"},
        ):
            response = mock_client.post(
                "/proxy/openai/v1/chat/completions",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    "X-Gitlab-Unit-Primitive": unit_primitive,
                },
                json={
                    "model": "gpt-5",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hi, how are you?"}],
                    "stream": False,
                },
            )

        assert response.status_code == 200
        assert response.json() == {"response": "test"}

        mock_track_internal_event.assert_called_once_with(
            f"request_{unit_primitive}",
            category="ai_gateway.api.v1.proxy.openai",
        )

    @pytest.mark.parametrize(
        "unit_primitive", EXTENDED_FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS.keys()
    )
    def test_successful_completions_request(
        self, mock_client, mock_track_internal_event, mock_detect_abuse, unit_primitive
    ):
        with patch(
            "ai_gateway.proxy.clients.OpenAIProxyClient.proxy",
            return_value={"response": "completion test"},
        ):
            response = mock_client.post(
                "/proxy/openai/v1/completions",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    "X-Gitlab-Unit-Primitive": unit_primitive,
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "prompt": "Hello, world!",
                    "max_tokens": 100,
                    "temperature": 0.7,
                },
            )

        assert response.status_code == 200
        assert response.json() == {"response": "completion test"}

        mock_track_internal_event.assert_called_once_with(
            f"request_{unit_primitive}",
            category="ai_gateway.api.v1.proxy.openai",
        )

    @pytest.mark.parametrize(
        "unit_primitive", EXTENDED_FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS.keys()
    )
    def test_successful_embeddings_request(
        self, mock_client, mock_track_internal_event, mock_detect_abuse, unit_primitive
    ):
        with patch(
            "ai_gateway.proxy.clients.OpenAIProxyClient.proxy",
            return_value={"data": [{"embedding": [0.1, 0.2, 0.3]}]},
        ):
            response = mock_client.post(
                "/proxy/openai/v1/embeddings",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    "X-Gitlab-Unit-Primitive": unit_primitive,
                },
                json={
                    "model": "text-embedding-ada-002",
                    "input": "Hello, world!",
                },
            )

        assert response.status_code == 200
        assert response.json() == {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

        mock_track_internal_event.assert_called_once_with(
            f"request_{unit_primitive}",
            category="ai_gateway.api.v1.proxy.openai",
        )

    def test_successful_models_request(
        self, mock_client, mock_track_internal_event, mock_detect_abuse
    ):
        unit_primitive = list(EXTENDED_FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS.keys())[0]

        with patch(
            "ai_gateway.proxy.clients.OpenAIProxyClient.proxy",
            return_value={"data": [{"id": "gpt-5", "object": "model"}]},
        ):
            response = mock_client.post(
                "/proxy/openai/v1/models",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    "X-Gitlab-Unit-Primitive": unit_primitive,
                },
                json={
                    "model": "gpt-5",
                },
            )

        assert response.status_code == 200
        assert response.json() == {"data": [{"id": "gpt-5", "object": "model"}]}

        mock_track_internal_event.assert_called_once_with(
            f"request_{unit_primitive}",
            category="ai_gateway.api.v1.proxy.openai",
        )

    def test_streaming_request(
        self, mock_client, mock_track_internal_event, mock_detect_abuse
    ):
        unit_primitive = list(EXTENDED_FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS.keys())[0]

        with patch(
            "ai_gateway.proxy.clients.OpenAIProxyClient.proxy",
            return_value={"response": "streaming test"},
        ):
            response = mock_client.post(
                "/proxy/openai/v1/chat/completions",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    "X-Gitlab-Unit-Primitive": unit_primitive,
                },
                json={
                    "model": "gpt-5",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                },
            )

        assert response.status_code == 200
        assert response.json() == {"response": "streaming test"}

        mock_track_internal_event.assert_called_once_with(
            f"request_{unit_primitive}",
            category="ai_gateway.api.v1.proxy.openai",
        )


class TestUnauthorizedScopes:
    @pytest.fixture(name="auth_user")
    def auth_user_fixture(self):
        return CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(scopes=["unauthorized_scope"]),
        )

    def test_failed_authorization_scope(self, mock_client):
        with patch("ai_gateway.proxy.clients.OpenAIProxyClient.proxy"):
            response = mock_client.post(
                "/proxy/openai/v1/chat/completions",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    "X-Gitlab-Unit-Primitive": GitLabUnitPrimitive.EXPLAIN_VULNERABILITY,
                },
                json={
                    "model": "gpt-5",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hi, how are you?"}],
                    "stream": False,
                },
            )

        assert response.status_code == 403
        assert response.json() == {
            "detail": "Unauthorized to access explain_vulnerability"
        }

    def test_missing_unit_primitive_header(self, mock_client):
        """Test request missing the required X-Gitlab-Unit-Primitive header."""
        with patch("ai_gateway.proxy.clients.OpenAIProxyClient.proxy"):
            response = mock_client.post(
                "/proxy/openai/v1/chat/completions",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    # Missing X-Gitlab-Unit-Primitive header
                },
                json={
                    "model": "gpt-5",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        assert response.status_code == 400


class TestInvalidRoutes:
    def test_invalid_openai_path(self, mock_client):
        """Test request to invalid OpenAI path."""
        unit_primitive = list(EXTENDED_FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS.keys())[0]

        response = mock_client.post(
            "/proxy/openai/v1/invalid-path",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-Gitlab-Unit-Primitive": unit_primitive,
            },
            json={
                "model": "gpt-5",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        # This should result in a 404 or 500 depending on how the proxy handles it
        assert response.status_code in [404, 500, 502]
