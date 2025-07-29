from unittest.mock import patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser, GitLabUnitPrimitive, UserClaims

from ai_gateway.api.v1 import api_router


@pytest.fixture(name="fast_api_router", scope="class")
def fast_api_router_fixture():
    return api_router


@pytest.fixture(name="auth_user")
def auth_user_fixture():
    return CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(
            scopes=[
                GitLabUnitPrimitive.EXPLAIN_VULNERABILITY,
            ]
        ),
    )


class TestProxyVertexAI:
    def test_successful_request(
        self, mock_client, mock_track_internal_event, mock_detect_abuse
    ):
        with patch(
            "ai_gateway.proxy.clients.VertexAIProxyClient.proxy",
            return_value={"response": "test"},
        ):
            response = mock_client.post(
                "/proxy/vertex-ai",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    "X-Gitlab-Unit-Primitive": GitLabUnitPrimitive.EXPLAIN_VULNERABILITY,
                },
                json={
                    "model": "claude-3-5-haiku-20241022",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hi, how are you?"}],
                    "stream": "true",
                },
            )

        assert response.status_code == 200
        assert response.json() == {"response": "test"}

        mock_track_internal_event.assert_called_once_with(
            "request_explain_vulnerability",
            category="ai_gateway.api.v1.proxy.vertex_ai",
        )


class TestUnauthorizedScopes:
    @pytest.fixture(name="auth_user")
    def auth_user_fixture(self):
        return CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(scopes=["unauthorized_scope"]),
        )

    def test_failed_authorization_scope(self, mock_client):
        with patch("ai_gateway.proxy.clients.VertexAIProxyClient.proxy"):
            response = mock_client.post(
                "/proxy/vertex-ai",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    "X-Gitlab-Unit-Primitive": GitLabUnitPrimitive.EXPLAIN_VULNERABILITY,
                },
                json={
                    "model": "claude-3-5-haiku-20241022",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hi, how are you?"}],
                    "stream": "true",
                },
            )

        assert response.status_code == 403
        assert response.json() == {
            "detail": "Unauthorized to access explain_vulnerability"
        }
