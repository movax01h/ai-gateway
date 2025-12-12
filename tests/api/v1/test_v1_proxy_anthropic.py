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


class TestProxyAnthropic:
    @pytest.mark.parametrize(
        "unit_primitive", EXTENDED_FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS.keys()
    )
    def test_successful_request(
        self, mock_client, mock_track_internal_event, mock_detect_abuse, unit_primitive
    ):
        with patch(
            "ai_gateway.proxy.clients.AnthropicProxyClient.proxy",
            return_value={"response": "test"},
        ):
            response = mock_client.post(
                "/proxy/anthropic/",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    "X-Gitlab-Unit-Primitive": unit_primitive,
                    "X-Gitlab-Instance-Id": "1",
                    "X-Gitlab-Project-Id": "1",
                    "X-Gitlab-Namespace-Id": "1",
                    "x-gitlab-root-namespace-id": "1",
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
            f"request_{unit_primitive}",
            category="ai_gateway.api.v1.proxy.anthropic",
        )


class TestUnauthorizedScopes:
    @pytest.fixture(name="auth_user")
    def auth_user_fixture(self):
        return CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(scopes=["unauthorized_scope"]),
        )

    def test_failed_authorization_scope(self, mock_client, proxy_headers):
        with patch("ai_gateway.proxy.clients.AnthropicProxyClient.proxy"):
            headers = {
                **proxy_headers,
                "X-Gitlab-Unit-Primitive": GitLabUnitPrimitive.EXPLAIN_VULNERABILITY,
            }
            response = mock_client.post(
                "/proxy/anthropic/",
                headers=headers,
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
        with patch("ai_gateway.proxy.clients.AnthropicProxyClient.proxy"):
            headers = {
                **proxy_headers,
                "X-Gitlab-Unit-Primitive": GitLabUnitPrimitive.ASK_BUILD,
            }
            # Override the header with mismatched value
            headers[header_key] = header_value

            response = mock_client.post(
                "/proxy/anthropic/",
                headers=headers,
                json={
                    "model": "claude-3-5-haiku-20241022",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hi, how are you?"}],
                    "stream": "true",
                },
            )

            assert response.status_code == expected_status
            response_json = response.json()
            if expected_status == 401:
                assert expected_detail in response_json["error"]
            else:
                assert response_json == {"detail": expected_detail}
