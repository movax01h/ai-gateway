import json
import time
from unittest.mock import ANY, AsyncMock, Mock, patch

import httpx
import pytest
from gitlab_cloud_connector import CloudConnectorUser, GitLabUnitPrimitive, UserClaims

from ai_gateway.proxy.clients import ProxyModel
from lib.billing_events.client import BillingEvent


@pytest.fixture(name="mock_proxy_model")
def mock_proxy_model_fixture():
    """Mock ProxyModel for testing."""
    return ProxyModel(
        base_url="https://vertex-ai.googleapis.com",
        model_name="text-embedding-005",
        upstream_path="/v1/projects/PROJECT/locations/LOCATION/publishers/google/models/text-embedding-005:predict",
        stream=False,
        upstream_service="vertex-ai",
        headers_to_upstream={"Authorization": "Bearer test"},
        allowed_upstream_models=["text-embedding-005"],
        allowed_headers_to_upstream=[],
        allowed_headers_to_downstream=[],
    )


@pytest.fixture(name="proxy_upstream_response")
def proxy_upstream_response_fixture() -> dict:
    return {
        "predictions": [
            {"embeddings": {"statistics": {"token_count": 2}, "values": []}}
        ]
    }


class TestProxyVertexAI:
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
                "ai_gateway.proxy.clients.VertexAIProxyModelFactory.factory",
                new_callable=AsyncMock,
                return_value=mock_proxy_model,
            ),
        ):
            response = mock_client.post(
                "/proxy/vertex-ai/",
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
            category="ai_gateway.api.v1.proxy.vertex_ai",
        )

    @pytest.mark.usefixtures("mock_proxy_async_client")
    def test_billing_event(
        self,
        mock_client,
        mock_send_billing_event,
        unit_primitive,
        mock_proxy_model,
    ):
        with (
            patch(
                "ai_gateway.proxy.clients.VertexAIProxyModelFactory.factory",
                new_callable=AsyncMock,
                return_value=mock_proxy_model,
            ),
        ):
            response = mock_client.post(
                "/proxy/vertex-ai/",
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
                },
            )

        assert response.status_code == 200

        mock_send_billing_event.assert_called_once_with(
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
                        "token_count": 2,
                        "model_id": "text-embedding-005",
                        "model_engine": "vertex-ai",
                        "model_provider": "vertex-ai",
                        "model_vendor": "vertex-ai",
                        "prompt_tokens": 2,
                        "completion_tokens": 0,
                        "agent_name": None,
                        "cache_read_tokens": 0,
                        "cache_write_tokens": 0,
                        "operation_type": "standard",
                    }
                ],
                "tool_names": [],
                "orbit_called": False,
            },
        )

    @pytest.mark.parametrize(
        ("action", "content_type", "body_prefix"),
        [
            ("generateContent", "application/json", ""),
            ("streamGenerateContent?alt=sse", "text/event-stream", "data: "),
        ],
    )
    def test_gemini_billing_event(
        self,
        mock_client,
        mock_send_billing_event,
        mock_proxy_async_client,
        unit_primitive,
        action,
        content_type,
        body_prefix,
    ):
        """A Gemini response is sent to the global endpoint and billed with its token usage."""
        upstream_response = {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "ready"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 7,
                "candidatesTokenCount": 3,
                "totalTokenCount": 10,
            },
        }
        mock_proxy_async_client.send.return_value = httpx.Response(
            status_code=200,
            headers={"Content-Type": content_type},
            content=f"{body_prefix}{json.dumps(upstream_response)}\r\n\r\n".encode(),
            request=Mock(),
        )

        # LiteLLM logs a finished stream after the response ends. Keep one event loop open, as the server does, so
        # that work is not cancelled with the request.
        with patch("ai_gateway.proxy.clients.vertex_ai.access_token"), mock_client:
            response = mock_client.post(
                "/proxy/vertex-ai/v1/projects/PROJECT/locations/LOCATION/"
                f"publishers/google/models/gemini-4-argon:{action}",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    "X-Gitlab-Unit-Primitive": unit_primitive,
                    "X-Gitlab-Instance-Id": "1",
                    "X-Gitlab-Project-Id": "1",
                    "X-Gitlab-Namespace-Id": "1",
                    "x-gitlab-root-namespace-id": "1",
                },
                json={"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]},
            )
            deadline = time.monotonic() + 5
            while not mock_send_billing_event.called and time.monotonic() < deadline:
                time.sleep(0.01)

        assert response.status_code == 200
        assert json.dumps(upstream_response) in response.text

        upstream_url = mock_proxy_async_client.send.call_args[0][0].url
        assert upstream_url.host == "aiplatform.googleapis.com"
        assert "/locations/global/publishers/google/models/gemini-4-argon:" in str(
            upstream_url
        )

        mock_send_billing_event.assert_called_once()
        operations = mock_send_billing_event.call_args.kwargs["metadata"][
            "llm_operations"
        ]
        assert [
            (op["model_id"], op["prompt_tokens"], op["completion_tokens"])
            for op in operations
        ] == [("gemini-4-argon", 7, 3)]


class TestUnauthorizedScopes:
    @pytest.fixture(name="auth_user")
    def auth_user_fixture(self):
        return CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(scopes=["unauthorized_scope"]),
        )

    def test_failed_authorization_scope(self, mock_client, mock_proxy_model):
        with (
            patch("ai_gateway.proxy.clients.ProxyClient.proxy"),
            patch(
                "ai_gateway.proxy.clients.VertexAIProxyModelFactory.factory",
                new_callable=AsyncMock,
                return_value=mock_proxy_model,
            ),
        ):
            response = mock_client.post(
                "/proxy/vertex-ai/",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    "X-Gitlab-Unit-Primitive": GitLabUnitPrimitive.EXPLAIN_VULNERABILITY,
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
                "ai_gateway.proxy.clients.VertexAIProxyModelFactory.factory",
                new_callable=AsyncMock,
                return_value=mock_proxy_model,
            ),
        ):
            headers = {
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-Gitlab-Unit-Primitive": GitLabUnitPrimitive.ASK_BUILD,
                "X-Gitlab-Instance-Id": "1",
                "X-Gitlab-Project-Id": "1",
                "X-Gitlab-Namespace-Id": "1",
                "x-gitlab-root-namespace-id": "1",
            }
            # Override the header with mismatched value
            headers[header_key] = header_value

            response = mock_client.post(
                "/proxy/vertex-ai/",
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
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "claude-3-5-sonnet-20241022",
        ],
    )
    @pytest.mark.usefixtures("mock_track_internal_event")
    def test_passes_model_name_to_usage_quota_service(
        self,
        mock_client,
        unit_primitive,
        model_name,
    ):
        """Test that model_name from ProxyModel is passed to usage quota service."""
        mock_proxy_model = ProxyModel(
            base_url="https://vertex-ai.googleapis.com",
            model_name=model_name,
            upstream_path="/v1/test",
            stream=False,
            upstream_service="vertex-ai",
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
                "ai_gateway.proxy.clients.VertexAIProxyModelFactory.factory",
                new_callable=AsyncMock,
                return_value=mock_proxy_model,
            ),
            patch(
                "lib.usage_quota.service.UsageQuotaService.execute",
                new_callable=AsyncMock,
            ) as mock_usage_quota_execute,
        ):
            response = mock_client.post(
                "/proxy/vertex-ai/",
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
