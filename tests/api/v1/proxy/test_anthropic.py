from unittest.mock import ANY, AsyncMock, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser, GitLabUnitPrimitive, UserClaims

from ai_gateway.proxy.clients import ProxyModel
from lib.billing_events.client import BillingEvent


@pytest.fixture(name="mock_proxy_model")
def mock_proxy_model_fixture():
    """Mock ProxyModel for testing."""
    return ProxyModel(
        base_url="https://api.anthropic.com",
        model_name="claude-3-5-haiku-20241022",
        upstream_path="/v1/messages",
        stream=False,
        upstream_service="anthropic",
        headers_to_upstream={"x-api-key": "test"},
        allowed_upstream_models=["claude-3-5-haiku-20241022"],
        allowed_headers_to_upstream=[],
        allowed_headers_to_downstream=[],
    )


@pytest.fixture(name="proxy_upstream_response")
def proxy_upstream_response_fixture() -> dict:
    return {
        "model": "claude-3-5-haiku-20241022",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Hello!",
            }
        ],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 1,
            "cache_creation_input_tokens": 2,
            "cache_read_input_tokens": 3,
            "output_tokens": 4,
        },
    }


class TestProxyAnthropic:
    def test_successful_request(
        self,
        mock_client,
        mock_track_internal_event,
        unit_primitive,
        mock_proxy_model,
    ):
        with (
            patch(
                "ai_gateway.proxy.clients.ProxyClient.proxy",
                return_value={"response": "test"},
            ),
            patch(
                "ai_gateway.proxy.clients.AnthropicProxyModelFactory.factory",
                new_callable=AsyncMock,
                return_value=mock_proxy_model,
            ),
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

    @pytest.mark.usefixtures("mock_proxy_async_client")
    def test_billing_event(
        self,
        mock_client,
        mock_track_billing_event,
        unit_primitive,
        mock_proxy_model,
    ):
        with (
            patch(
                "ai_gateway.proxy.clients.AnthropicProxyModelFactory.factory",
                new_callable=AsyncMock,
                return_value=mock_proxy_model,
            ),
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
                    "stream": False,
                },
            )

        assert response.status_code == 200

        mock_track_billing_event.assert_called_once_with(
            ANY,
            BillingEvent.AIGW_PROXY_USE,
            "ai_gateway.proxy.clients.base",
            unit_of_measure="request",
            quantity=1,
            metadata={
                "feature_qualified_name": "ai_gateway_proxy_use",
                "feature_ai_catalog_item": False,
                "execution_environment": "duo_agent_platform",
                "llm_operations": [
                    {
                        "token_count": 10,
                        "model_id": "claude-3-5-haiku-20241022",
                        "model_engine": "anthropic",
                        "model_provider": "anthropic",
                        "prompt_tokens": 6,
                        "completion_tokens": 4,
                        "agent_name": None,
                        "cache_read_tokens": 3,
                        "cache_write_tokens": 2,
                        "operation_type": "standard",
                    }
                ],
                "tool_names": [],
                "orbit_called": False,
            },
        )


class TestUnauthorizedScopes:
    @pytest.fixture(name="auth_user")
    def auth_user_fixture(self):
        return CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(scopes=["unauthorized_scope"]),
        )

    def test_failed_authorization_scope(
        self, mock_client, proxy_headers, mock_proxy_model
    ):
        with (
            patch("ai_gateway.proxy.clients.ProxyClient.proxy"),
            patch(
                "ai_gateway.proxy.clients.AnthropicProxyModelFactory.factory",
                new_callable=AsyncMock,
                return_value=mock_proxy_model,
            ),
        ):
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
        mock_proxy_model,
    ):
        """Test that mismatched headers are rejected with appropriate status codes."""
        with (
            patch("ai_gateway.proxy.clients.ProxyClient.proxy"),
            patch(
                "ai_gateway.proxy.clients.AnthropicProxyModelFactory.factory",
                new_callable=AsyncMock,
                return_value=mock_proxy_model,
            ),
        ):
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


class TestUsageQuotaWithModelName:
    """Tests for usage quota integration with model_name parameter."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ],
    )
    def test_passes_model_name_to_usage_quota_service(
        self,
        mock_client,
        mock_track_internal_event,
        proxy_headers,
        unit_primitive,
        model_name,
    ):
        """Test that model_name from ProxyModel is passed to usage quota service."""
        mock_proxy_model = ProxyModel(
            base_url="https://api.anthropic.com",
            model_name=model_name,
            upstream_path="/v1/messages",
            stream=False,
            upstream_service="anthropic",
            headers_to_upstream={"x-api-key": "test"},
            allowed_upstream_models=[model_name],
            allowed_headers_to_upstream=[],
            allowed_headers_to_downstream=[],
        )

        with (
            patch(
                "ai_gateway.proxy.clients.ProxyClient.proxy",
                return_value={"response": "test"},
            ),
            patch(
                "ai_gateway.proxy.clients.AnthropicProxyModelFactory.factory",
                new_callable=AsyncMock,
                return_value=mock_proxy_model,
            ),
            patch(
                "lib.usage_quota.service.UsageQuotaService.execute",
                new_callable=AsyncMock,
            ) as mock_usage_quota_execute,
        ):
            headers = {**proxy_headers, "X-Gitlab-Unit-Primitive": unit_primitive}
            response = mock_client.post(
                "/proxy/anthropic/",
                headers=headers,
                json={
                    "model": model_name,
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

            assert response.status_code == 200

            # Verify usage quota service was called with model metadata
            mock_usage_quota_execute.assert_called_once()
            call_args = mock_usage_quota_execute.call_args
            model_metadata = call_args[1]["model_metadata"]
            assert model_metadata is not None
            assert model_metadata.name == model_name
