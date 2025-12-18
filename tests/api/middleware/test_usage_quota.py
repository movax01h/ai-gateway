import json
from unittest.mock import AsyncMock, patch

import pytest
from starlette.requests import Request
from starlette.responses import Response

from ai_gateway.api.middleware.usage_quota import UsageQuotaMiddleware
from lib.billing_events.context import UsageQuotaEventContext
from lib.internal_events.context import EventContext
from lib.usage_quota.errors import (
    UsageQuotaConnectionError,
    UsageQuotaHTTPError,
    UsageQuotaTimeoutError,
)


@pytest.fixture(name="internal_event_context")
def internal_event_context_fixture():
    return EventContext(
        environment="test",
        source="duo_chat",
        realm="saas",
        deployment_type="saas",
        instance_id="4398e2e6-012d-49d9-bada-d419458fe75f",
        unique_instance_id="4398e2e6-012d-49d9-bada-d419458fe75f",
        feature_enablement_type="duo_pro",
        host_name="gitlab.local",
        instance_version="17.5.0",
        global_user_id="user-123",
        user_id=None,
        project_id=789,
        ultimate_parent_namespace_id=456,
        namespace_id=456,
        correlation_id="12345",
    )


@pytest.fixture(name="usage_quota_event_context")
def usage_quota_event_context_fixture(internal_event_context):
    return UsageQuotaEventContext.from_internal_event(internal_event_context)


@pytest.fixture(autouse=True)
def mock_context(internal_event_context):
    with patch(
        "ai_gateway.api.middleware.usage_quota.current_event_context"
    ) as mock_ctx:
        mock_ctx.get.return_value = internal_event_context
        yield


@pytest.fixture(name="usage_quota_middleware")
def usage_quota_middleware_fixture(mock_app):
    """Create a UsageQuotaMiddleware instance for testing."""
    middleware = UsageQuotaMiddleware(
        app=mock_app,
        customersdot_url="https://customers.gitlab.local",
        customersdot_api_user="customersdot_api_user",
        customersdot_api_token="customersdot_api_token",
        skip_endpoints=["/health", "/metrics"],
        environment="test",
    )
    return middleware


class TestEndpointSkipping:
    """Tests for endpoint skipping behavior."""

    @pytest.mark.asyncio
    async def test_skips_endpoint_in_skip_list(self, usage_quota_middleware):
        """Test that middleware skips usage quota check for endpoints in skip_endpoints list."""
        request = Request(
            {
                "type": "http",
                "path": "/health",
                "method": "GET",
                "headers": [],
            }
        )
        call_next = AsyncMock(return_value=Response(status_code=200))

        with patch.object(
            usage_quota_middleware.usage_quota_client,
            "check_quota_available",
            new_callable=AsyncMock,
        ) as mock_check:
            response = await usage_quota_middleware.dispatch(request, call_next)

            assert response.status_code == 200
            call_next.assert_called_once_with(request)
            mock_check.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_metrics_endpoint(self, usage_quota_middleware):
        """Test that /metrics endpoint is skipped."""
        request = Request(
            {
                "type": "http",
                "path": "/metrics",
                "method": "GET",
                "headers": [],
            }
        )
        call_next = AsyncMock(return_value=Response(status_code=200))

        with patch.object(
            usage_quota_middleware.usage_quota_client,
            "check_quota_available",
            new_callable=AsyncMock,
        ) as mock_check:
            response = await usage_quota_middleware.dispatch(request, call_next)

            assert response.status_code == 200
            call_next.assert_called_once_with(request)
            mock_check.assert_not_called()


class TestQuotaEnforcement:
    """Tests for quota check enforcement."""

    @pytest.mark.asyncio
    async def test_authorized_request_passes_through(self, usage_quota_middleware):
        """Test that a request with available quota is allowed through."""
        request = Request(
            {"type": "http", "path": "/api/v1/chat", "method": "POST", "headers": []}
        )
        call_next = AsyncMock(return_value=Response(status_code=200))

        with patch.object(
            usage_quota_middleware.usage_quota_client,
            "check_quota_available",
            new_callable=AsyncMock,
            return_value=True,
        ):
            response = await usage_quota_middleware.dispatch(request, call_next)

        assert response.status_code == 200
        call_next.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_no_usage_quota_left_returns_402(self, usage_quota_middleware):
        """Test that a request without quota returns 402 status code."""
        request = Request(
            {"type": "http", "path": "/api/v1/chat", "method": "POST", "headers": []}
        )
        call_next = AsyncMock(return_value=Response(status_code=200))

        with patch.object(
            usage_quota_middleware.usage_quota_client,
            "check_quota_available",
            new_callable=AsyncMock,
            return_value=False,
        ):
            response = await usage_quota_middleware.dispatch(request, call_next)

        assert response.status_code == 402
        call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_usage_quota_left_response_format(self, usage_quota_middleware):
        """Test that 402 error response has correct JSON format."""
        request = Request(
            {"type": "http", "path": "/api/v1/chat", "method": "POST", "headers": []}
        )
        call_next = AsyncMock(return_value=Response(status_code=200))

        with patch.object(
            usage_quota_middleware.usage_quota_client,
            "check_quota_available",
            new_callable=AsyncMock,
            return_value=False,
        ):
            response = await usage_quota_middleware.dispatch(request, call_next)

        assert response.status_code == 402
        content = json.loads(response.body.decode())
        assert content["error"] == "insufficient_credits"
        assert content["error_code"] == "USAGE_QUOTA_EXCEEDED"
        assert "sufficient credits" in content["message"]


class TestFailOpenBehavior:
    """Tests for fail-open behavior on errors."""

    @pytest.mark.parametrize(
        "exception_class",
        [
            UsageQuotaTimeoutError,
            UsageQuotaHTTPError,
            UsageQuotaConnectionError,
        ],
    )
    @pytest.mark.asyncio
    async def test_fails_open_on_usage_quota_errors(
        self, usage_quota_middleware, exception_class
    ):
        """Test that middleware fails open on UsageQuotaError exceptions."""
        request = Request(
            {"type": "http", "path": "/api/v1/chat", "method": "POST", "headers": []}
        )
        call_next = AsyncMock(return_value=Response(status_code=200))

        if exception_class == UsageQuotaHTTPError:
            exception = exception_class(status_code=500)
        else:
            exception = exception_class()

        with (
            patch.object(
                usage_quota_middleware.usage_quota_client,
                "check_quota_available",
                new_callable=AsyncMock,
                side_effect=exception,
            ),
            patch(
                "ai_gateway.api.middleware.usage_quota.USAGE_QUOTA_CHECK_TOTAL"
            ) as mock_metrics,
        ):
            response = await usage_quota_middleware.dispatch(request, call_next)

        assert response.status_code == 200
        call_next.assert_called_once_with(request)
        mock_metrics.labels.assert_called_once_with(result="fail_open", realm="saas")
        mock_metrics.labels.return_value.inc.assert_called_once()


class TestMetrics:
    """Tests for metrics recording."""

    @pytest.mark.asyncio
    async def test_records_allow_metric_on_quota_available(
        self, usage_quota_middleware
    ):
        """Test that 'allow' metric is recorded when quota is available."""
        request = Request(
            {"type": "http", "path": "/api/v1/chat", "method": "POST", "headers": []}
        )
        call_next = AsyncMock(return_value=Response(status_code=200))

        with (
            patch.object(
                usage_quota_middleware.usage_quota_client,
                "check_quota_available",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "ai_gateway.api.middleware.usage_quota.USAGE_QUOTA_CHECK_TOTAL"
            ) as mock_metrics,
        ):
            await usage_quota_middleware.dispatch(request, call_next)

        mock_metrics.labels.assert_called_once_with(result="allow", realm="saas")
        mock_metrics.labels.return_value.inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_records_deny_metric_on_quota_exhausted(self, usage_quota_middleware):
        """Test that 'deny' metric is recorded when quota is exhausted."""
        request = Request(
            {"type": "http", "path": "/api/v1/chat", "method": "POST", "headers": []}
        )
        call_next = AsyncMock()

        with (
            patch.object(
                usage_quota_middleware.usage_quota_client,
                "check_quota_available",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "ai_gateway.api.middleware.usage_quota.USAGE_QUOTA_CHECK_TOTAL"
            ) as mock_metrics,
        ):
            await usage_quota_middleware.dispatch(request, call_next)

        mock_metrics.labels.assert_called_once_with(result="deny", realm="saas")
        mock_metrics.labels.return_value.inc.assert_called_once()
