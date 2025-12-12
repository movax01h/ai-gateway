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
            scopes=EXTENDED_FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS.keys(),
            gitlab_instance_id="1",
            extra={
                "gitlab_project_id": "1",
                "gitlab_namespace_id": "1",
                "gitlab_root_namespace_id": "1",
            },
        ),
    )


@pytest.fixture(name="proxy_headers")
def proxy_headers_fixture():
    """Common headers for proxy requests."""
    return {
        "Authorization": "Bearer 12345",
        "X-Gitlab-Authentication-Type": "oidc",
        "X-Gitlab-Instance-Id": "1",
        "X-Gitlab-Project-Id": "1",
        "X-Gitlab-Namespace-Id": "1",
        "x-gitlab-root-namespace-id": "1",
    }


class TestProxyOpenAI:
    @pytest.mark.parametrize(
        "unit_primitive", EXTENDED_FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS.keys()
    )
    def test_successful_chat_completions_request(
        self,
        mock_client,
        mock_track_internal_event,
        mock_detect_abuse,
        unit_primitive,
        proxy_headers,
    ):
        with patch(
            "ai_gateway.proxy.clients.OpenAIProxyClient.proxy",
            return_value={"response": "test"},
        ):
            headers = {**proxy_headers, "X-Gitlab-Unit-Primitive": unit_primitive}
            response = mock_client.post(
                "/proxy/openai/v1/chat/completions",
                headers=headers,
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
        self,
        mock_client,
        mock_track_internal_event,
        mock_detect_abuse,
        unit_primitive,
        proxy_headers,
    ):
        with patch(
            "ai_gateway.proxy.clients.OpenAIProxyClient.proxy",
            return_value={"response": "completion test"},
        ):
            headers = {**proxy_headers, "X-Gitlab-Unit-Primitive": unit_primitive}
            response = mock_client.post(
                "/proxy/openai/v1/completions",
                headers=headers,
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
        self,
        mock_client,
        mock_track_internal_event,
        mock_detect_abuse,
        unit_primitive,
        proxy_headers,
    ):
        with patch(
            "ai_gateway.proxy.clients.OpenAIProxyClient.proxy",
            return_value={"data": [{"embedding": [0.1, 0.2, 0.3]}]},
        ):
            headers = {**proxy_headers, "X-Gitlab-Unit-Primitive": unit_primitive}
            response = mock_client.post(
                "/proxy/openai/v1/embeddings",
                headers=headers,
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
        self, mock_client, mock_track_internal_event, mock_detect_abuse, proxy_headers
    ):
        unit_primitive = list(EXTENDED_FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS.keys())[0]

        with patch(
            "ai_gateway.proxy.clients.OpenAIProxyClient.proxy",
            return_value={"data": [{"id": "gpt-5", "object": "model"}]},
        ):
            headers = {**proxy_headers, "X-Gitlab-Unit-Primitive": unit_primitive}
            response = mock_client.post(
                "/proxy/openai/v1/models",
                headers=headers,
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

    @pytest.mark.parametrize(
        "unit_primitive", EXTENDED_FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS.keys()
    )
    def test_successful_responses_request(
        self,
        mock_client,
        mock_track_internal_event,
        mock_detect_abuse,
        unit_primitive,
        proxy_headers,
    ):
        with patch(
            "ai_gateway.proxy.clients.OpenAIProxyClient.proxy",
            return_value={"response": "responses test"},
        ):
            headers = {**proxy_headers, "X-Gitlab-Unit-Primitive": unit_primitive}
            response = mock_client.post(
                "/proxy/openai/v1/responses",
                headers=headers,
                json={
                    "model": "gpt-5-codex",
                    "input": "Tell me a three sentence bedtime story about a unicorn.",
                },
            )

        assert response.status_code == 200
        assert response.json() == {"response": "responses test"}

        mock_track_internal_event.assert_called_once_with(
            f"request_{unit_primitive}",
            category="ai_gateway.api.v1.proxy.openai",
        )

    def test_streaming_request(
        self, mock_client, mock_track_internal_event, mock_detect_abuse, proxy_headers
    ):
        unit_primitive = list(EXTENDED_FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS.keys())[0]

        with patch(
            "ai_gateway.proxy.clients.OpenAIProxyClient.proxy",
            return_value={"response": "streaming test"},
        ):
            headers = {**proxy_headers, "X-Gitlab-Unit-Primitive": unit_primitive}
            response = mock_client.post(
                "/proxy/openai/v1/chat/completions",
                headers=headers,
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

    def test_failed_authorization_scope(self, mock_client, proxy_headers):
        with patch("ai_gateway.proxy.clients.OpenAIProxyClient.proxy"):
            headers = {
                **proxy_headers,
                "X-Gitlab-Unit-Primitive": GitLabUnitPrimitive.EXPLAIN_VULNERABILITY,
            }
            response = mock_client.post(
                "/proxy/openai/v1/chat/completions",
                headers=headers,
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

    def test_missing_unit_primitive_header(self, mock_client, proxy_headers):
        """Test request missing the required X-Gitlab-Unit-Primitive header."""
        with patch("ai_gateway.proxy.clients.OpenAIProxyClient.proxy"):
            # Remove X-Gitlab-Unit-Primitive from headers
            headers = {
                k: v for k, v in proxy_headers.items() if k != "X-Gitlab-Unit-Primitive"
            }
            response = mock_client.post(
                "/proxy/openai/v1/chat/completions",
                headers=headers,
                json={
                    "model": "gpt-5",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        assert response.status_code == 400


class TestDataMismatch:
    @pytest.mark.parametrize(
        "header_key,header_value,expected_status,expected_detail",
        [
            (
                "X-Gitlab-Instance-Id",
                "999",
                401,
                "Header mismatch",
            ),
            (
                "X-Gitlab-Project-Id",
                "999",
                403,
                "Gitlab project id mismatch",
            ),
            (
                "X-Gitlab-Namespace-Id",
                "999",
                403,
                "Gitlab namespace id mismatch",
            ),
            (
                "x-gitlab-root-namespace-id",
                "999",
                403,
                "Gitlab root namespace id mismatch",
            ),
        ],
    )
    def test_header_mismatch(
        self,
        mock_client,
        proxy_headers,
        header_key,
        header_value,
        expected_status,
        expected_detail,
    ):
        """Test that mismatched headers are rejected with appropriate status codes."""
        with patch("ai_gateway.proxy.clients.OpenAIProxyClient.proxy"):
            headers = {
                **proxy_headers,
                "X-Gitlab-Unit-Primitive": GitLabUnitPrimitive.ASK_BUILD,
            }
            # Override the header with mismatched value
            headers[header_key] = header_value

            response = mock_client.post(
                "/proxy/openai/v1/chat/completions",
                headers=headers,
                json={
                    "model": "gpt-5",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hi, how are you?"}],
                    "stream": False,
                },
            )

            assert response.status_code == expected_status
            response_json = response.json()
            if expected_status == 401:
                assert expected_detail in response_json["error"]
            else:
                assert response_json == {"detail": expected_detail}


class TestInvalidRoutes:
    def test_invalid_openai_path(self, mock_client, proxy_headers):
        """Test request to invalid OpenAI path."""
        unit_primitive = list(EXTENDED_FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS.keys())[0]

        headers = {**proxy_headers, "X-Gitlab-Unit-Primitive": unit_primitive}
        response = mock_client.post(
            "/proxy/openai/v1/invalid-path",
            headers=headers,
            json={
                "model": "gpt-5",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        # This should result in a 404 or 500 depending on how the proxy handles it
        assert response.status_code in [404, 500, 502]
