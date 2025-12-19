from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from duo_workflow_service.interceptors.usage_quota_interceptor import (
    UsageQuotaInterceptor,
)
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
        "duo_workflow_service.interceptors.usage_quota_interceptor.current_event_context"
    ) as mock_ctx:
        mock_ctx.get.return_value = internal_event_context
        yield


@pytest.fixture(name="handler_call_details")
def handler_call_details_fixture():
    """Create a mock for the handler_call_details."""
    details = MagicMock()
    details.invocation_metadata = ()
    return details


@pytest.fixture(name="continuation")
def continuation_fixture():
    """Create a mock for the continuation function."""
    mock_handler = Mock()
    mock_handler.unary_unary = AsyncMock(return_value="success_response")
    mock_handler.unary_stream = None
    mock_handler.stream_unary = None
    mock_handler.stream_stream = None
    mock_handler.request_streaming = False
    mock_handler.response_streaming = False
    mock_handler.request_deserializer = None
    mock_handler.response_serializer = None

    continuation = AsyncMock(return_value=mock_handler)
    return continuation


@pytest.fixture(name="interceptor")
def interceptor_fixture():
    return UsageQuotaInterceptor(
        customersdot_url="https://customers.gitlab.local/",
        customersdot_api_user="aigw@gitlab.local",
        customersdot_api_token="customersdot_api_token",
    )


class TestQuotaEnforcement:
    """Tests for quota check enforcement."""

    @pytest.mark.asyncio
    async def test_authorized_request_continues(
        self, interceptor, continuation, handler_call_details
    ):
        """Test that a request with available quota continues to the handler."""
        with patch.object(
            interceptor.usage_quota_client,
            "check_quota_available",
            new_callable=AsyncMock,
            return_value=True,
        ):
            handler = await interceptor.intercept_service(
                continuation, handler_call_details
            )

            assert handler is not None
            continuation.assert_called_once_with(handler_call_details)

    @pytest.mark.asyncio
    async def test_no_usage_quota_left_aborts_request(
        self, interceptor, continuation, handler_call_details
    ):
        """Test that a request without quota is aborted with RESOURCE_EXHAUSTED."""
        with patch.object(
            interceptor.usage_quota_client,
            "check_quota_available",
            new_callable=AsyncMock,
            return_value=False,
        ):
            handler = await interceptor.intercept_service(
                continuation, handler_call_details
            )

            assert handler is not None
            continuation.assert_not_called()

    @pytest.mark.asyncio
    async def test_abort_handler_uses_resource_exhausted_status(
        self, interceptor, continuation, handler_call_details
    ):
        """Test that abort handler uses RESOURCE_EXHAUSTED gRPC status."""
        with patch.object(
            interceptor.usage_quota_client,
            "check_quota_available",
            new_callable=AsyncMock,
            return_value=False,
        ):
            handler = await interceptor.intercept_service(
                continuation, handler_call_details
            )

            assert handler is not None


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
        self, interceptor, continuation, handler_call_details, exception_class
    ):
        """Test that interceptor fails open on UsageQuotaError exceptions."""
        if exception_class == UsageQuotaHTTPError:
            exception = exception_class(status_code=500)
        else:
            exception = exception_class()

        with (
            patch.object(
                interceptor.usage_quota_client,
                "check_quota_available",
                new_callable=AsyncMock,
                side_effect=exception,
            ),
            patch(
                "duo_workflow_service.interceptors.usage_quota_interceptor.USAGE_QUOTA_CHECK_TOTAL"
            ) as mock_metrics,
        ):
            handler = await interceptor.intercept_service(
                continuation, handler_call_details
            )

        assert handler is not None
        continuation.assert_called_once_with(handler_call_details)
        mock_metrics.labels.assert_called_once_with(result="fail_open", realm="saas")
        mock_metrics.labels.return_value.inc.assert_called_once()


class TestMetrics:
    """Tests for metrics recording."""

    @pytest.mark.asyncio
    async def test_records_allow_metric_on_quota_available(
        self, interceptor, continuation, handler_call_details
    ):
        """Test that 'allow' metric is recorded when quota is available."""
        with (
            patch.object(
                interceptor.usage_quota_client,
                "check_quota_available",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "duo_workflow_service.interceptors.usage_quota_interceptor.USAGE_QUOTA_CHECK_TOTAL"
            ) as mock_metrics,
        ):
            await interceptor.intercept_service(continuation, handler_call_details)

        mock_metrics.labels.assert_called_once_with(result="allow", realm="saas")
        mock_metrics.labels.return_value.inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_records_deny_metric_on_quota_exhausted(
        self, interceptor, continuation, handler_call_details
    ):
        """Test that 'deny' metric is recorded when quota is exhausted."""
        with (
            patch.object(
                interceptor.usage_quota_client,
                "check_quota_available",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "duo_workflow_service.interceptors.usage_quota_interceptor.USAGE_QUOTA_CHECK_TOTAL"
            ) as mock_metrics,
        ):
            await interceptor.intercept_service(continuation, handler_call_details)

        mock_metrics.labels.assert_called_once_with(result="deny", realm="saas")
        mock_metrics.labels.return_value.inc.assert_called_once()
