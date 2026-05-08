from unittest.mock import ANY, AsyncMock, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser, GitLabUnitPrimitive, UserClaims

from ai_gateway.proxy.clients import ProxyModel
from lib.billing_events.client import BillingEvent


@pytest.fixture(name="mock_proxy_model")
def mock_proxy_model_fixture():
    """Mock ProxyModel for testing."""
    return ProxyModel(
        base_url="https://api.openai.com",
        model_name="gpt-5",
        upstream_path="/v1/chat/completions",
        stream=False,
        upstream_service="openai",
        headers_to_upstream={"Authorization": "Bearer test"},
        allowed_upstream_models=["gpt-5"],
        allowed_headers_to_upstream=[],
        allowed_headers_to_downstream=[],
    )


@pytest.fixture(name="proxy_upstream_response")
def proxy_upstream_response_fixture() -> dict:
    return {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hi there! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        },
    }


class TestProxyOpenAI:
    def test_successful_chat_completions_request(
        self,
        mock_client,
        mock_track_internal_event,
        unit_primitive,
        proxy_headers,
        mock_proxy_model,
    ):
        with (
            patch(
                "ai_gateway.proxy.clients.ProxyClient.proxy",
                return_value={"response": "test"},
            ),
            patch(
                "ai_gateway.proxy.clients.OpenAIProxyModelFactory.factory",
                new_callable=AsyncMock,
                return_value=mock_proxy_model,
            ),
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

    def test_successful_completions_request(
        self,
        mock_client,
        mock_track_internal_event,
        unit_primitive,
        proxy_headers,
        mock_proxy_model,
    ):
        with (
            patch(
                "ai_gateway.proxy.clients.ProxyClient.proxy",
                return_value={"response": "completion test"},
            ),
            patch(
                "ai_gateway.proxy.clients.OpenAIProxyModelFactory.factory",
                new_callable=AsyncMock,
                return_value=mock_proxy_model,
            ),
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

    def test_successful_embeddings_request(
        self,
        mock_client,
        mock_track_internal_event,
        unit_primitive,
        proxy_headers,
        mock_proxy_model,
    ):
        with (
            patch(
                "ai_gateway.proxy.clients.ProxyClient.proxy",
                return_value={"data": [{"embedding": [0.1, 0.2, 0.3]}]},
            ),
            patch(
                "ai_gateway.proxy.clients.OpenAIProxyModelFactory.factory",
                new_callable=AsyncMock,
                return_value=mock_proxy_model,
            ),
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
        self,
        mock_client,
        mock_track_internal_event,
        proxy_headers,
        unit_primitive,
        mock_proxy_model,
    ):
        with (
            patch(
                "ai_gateway.proxy.clients.ProxyClient.proxy",
                return_value={"data": [{"id": "gpt-5", "object": "model"}]},
            ),
            patch(
                "ai_gateway.proxy.clients.OpenAIProxyModelFactory.factory",
                new_callable=AsyncMock,
                return_value=mock_proxy_model,
            ),
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

    def test_successful_responses_request(
        self,
        mock_client,
        mock_track_internal_event,
        unit_primitive,
        proxy_headers,
        mock_proxy_model,
    ):
        with (
            patch(
                "ai_gateway.proxy.clients.ProxyClient.proxy",
                return_value={"response": "responses test"},
            ),
            patch(
                "ai_gateway.proxy.clients.OpenAIProxyModelFactory.factory",
                new_callable=AsyncMock,
                return_value=mock_proxy_model,
            ),
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
        self,
        mock_client,
        mock_track_internal_event,
        proxy_headers,
        unit_primitive,
        mock_proxy_model,
    ):
        with (
            patch(
                "ai_gateway.proxy.clients.ProxyClient.proxy",
                return_value={"response": "streaming test"},
            ),
            patch(
                "ai_gateway.proxy.clients.OpenAIProxyModelFactory.factory",
                new_callable=AsyncMock,
                return_value=mock_proxy_model,
            ),
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

    @pytest.mark.usefixtures("mock_proxy_async_client")
    def test_billing_event(
        self,
        mock_client,
        mock_track_billing_event,
        unit_primitive,
        proxy_headers,
        mock_proxy_model,
    ):
        with (
            patch(
                "ai_gateway.proxy.clients.OpenAIProxyModelFactory.factory",
                new_callable=AsyncMock,
                return_value=mock_proxy_model,
            ),
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

        mock_track_billing_event.assert_called_once_with(
            ANY,
            event=BillingEvent.AIGW_PROXY_USE,
            category="ai_gateway.proxy.clients.base",
            unit_of_measure="request",
            quantity=1,
            metadata={
                "llm_operations": [
                    {
                        "token_count": 3,
                        "model_id": "gpt-5",
                        "model_engine": "openai",
                        "model_provider": "openai",
                        "prompt_tokens": 1,
                        "completion_tokens": 2,
                        "agent_name": None,
                        "cache_read_tokens": 0,
                        "cache_write_tokens": 0,
                        "operation_type": "standard",
                    }
                ],
                "feature_qualified_name": "ai_gateway_proxy_use",
                "feature_ai_catalog_item": False,
            },
        )


class TestUnauthorizedScopes:
    @pytest.fixture(name="auth_user")
    def auth_user_fixture(self):
        return CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(scopes=["unauthorized_scope"]),
        )

    def test_failed_authorization_scope(self, mock_client, proxy_headers):
        with patch("ai_gateway.proxy.clients.ProxyClient.proxy"):
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
        with patch("ai_gateway.proxy.clients.ProxyClient.proxy"):
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
        with patch("ai_gateway.proxy.clients.ProxyClient.proxy"):
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
    def test_invalid_openai_path(self, mock_client, proxy_headers, unit_primitive):
        """Test request to invalid OpenAI path."""
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


class TestUsageQuotaWithModelName:
    """Tests for usage quota integration with model_name parameter."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "gpt-4o",
            "gpt-4-turbo-2024-04-09",
            "gpt-3.5-turbo",
            "text-embedding-ada-002",
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
            base_url="https://api.openai.com",
            model_name=model_name,
            upstream_path="/v1/chat/completions",
            stream=False,
            upstream_service="openai",
            headers_to_upstream={"Authorization": "Bearer test"},
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
                "ai_gateway.proxy.clients.OpenAIProxyModelFactory.factory",
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
                "/proxy/openai/v1/chat/completions",
                headers=headers,
                json={
                    "model": model_name,
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
