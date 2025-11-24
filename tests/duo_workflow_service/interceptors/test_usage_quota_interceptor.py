from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

from duo_workflow_service.interceptors.usage_quota_interceptor import (
    UsageQuotaInterceptor,
)
from lib.billing_events.context import UsageQuotaEventContext
from lib.feature_flags.context import current_feature_flag_context
from lib.internal_events.context import EventContext


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


@pytest.fixture(autouse=True)
def stub_feature_flags():
    token = current_feature_flag_context.set({"usage_quota_left_check"})
    yield
    current_feature_flag_context.reset(token)


@pytest.fixture(name="usage_quota_interceptor")
def usage_quota_interceptor_fixture():
    """Create a fixture for the UsageQuotaInterceptor."""
    return UsageQuotaInterceptor


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
    return UsageQuotaInterceptor()


@pytest.mark.asyncio
async def test_feature_flag_disabled_skips_usage_quota_check(
    interceptor, continuation, handler_call_details
):
    """Test that middleware skips usage quota check when feature flag "usage_quota_left_check" is disabled."""
    current_feature_flag_context.set(set())

    with patch.object(
        interceptor, "has_usage_quota_left", new_callable=AsyncMock
    ) as mock_check:
        handler = await interceptor.intercept_service(
            continuation, handler_call_details
        )

        assert handler is not None
        mock_check.assert_not_called()


@pytest.mark.asyncio
async def test_feature_flag_enabled_performs_usage_quota_check(
    interceptor, continuation, handler_call_details
):
    """Test that middleware performs usage quota check when feature flag "usage_quota_left_check" is enabled."""

    with patch.object(
        interceptor, "has_usage_quota_left", new_callable=AsyncMock
    ) as mock_check:
        intercepted_handler = await interceptor.intercept_service(
            continuation, handler_call_details
        )
        assert intercepted_handler is not None
        mock_check.assert_called_once()


@pytest.mark.parametrize(
    "exception_class,exception_kwargs,expected_metric_outcome,expected_metric_status",
    [
        (
            httpx.TimeoutException,
            {"message": "Connection timed out"},
            "timeout",
            "timeout",
        ),
        (
            httpx.HTTPStatusError,
            {
                "message": "Client error",
                "request": MagicMock(),
                "response": MagicMock(status_code=500),
            },
            "http_error",
            "500",
        ),
        (
            httpx.RequestError,
            {"message": "Connection error"},
            "unexpected",
            "client_error",
        ),
    ],
)
@pytest.mark.asyncio
async def test_customersdot_http_exceptions_parameterized(
    interceptor,
    usage_quota_event_context,
    exception_class,
    exception_kwargs,
    expected_metric_outcome,
    expected_metric_status,
):
    """Test that different HTTP-related exceptions are properly raised and metrics recorded."""

    with (
        patch(
            "duo_workflow_service.interceptors.usage_quota_interceptor.httpx.AsyncClient"
        ) as mock_client_cls,
        patch(
            "duo_workflow_service.interceptors.usage_quota_interceptor.USAGE_QUOTA_CUSTOMERSDOT_REQUESTS_TOTAL"
        ) as mock_metrics,
    ):
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        exception = exception_class(**exception_kwargs)
        mock_client.head.side_effect = exception
        mock_client_cls.return_value = mock_client

        with pytest.raises(exception_class):
            await interceptor.has_usage_quota_left(usage_quota_event_context)

        mock_metrics.labels.assert_called_once_with(
            outcome=expected_metric_outcome, status=expected_metric_status
        )
        mock_metrics.labels.return_value.inc.assert_called_once()


@pytest.mark.parametrize(
    "exception_class,exception_kwargs",
    [
        (httpx.TimeoutException, {"message": "Connection timed out"}),
        (
            httpx.HTTPStatusError,
            {
                "message": "Internal server error",
                "request": MagicMock(),
                "response": MagicMock(status_code=500),
            },
        ),
        (httpx.RequestError, {"message": "Connection error"}),
    ],
)
@pytest.mark.asyncio
async def test_dispatch_fails_open_on_http_exceptions_parameterized(
    interceptor,
    continuation,
    handler_call_details,
    usage_quota_event_context,
    exception_class,
    exception_kwargs,
):
    """Test that interceptor allows requests when HTTP exceptions occur (fail-open)."""

    mock_check = AsyncMock(
        kwargs=usage_quota_event_context,
        side_effect=exception_class(**exception_kwargs),
    )

    with (
        patch.object(interceptor, "has_usage_quota_left", mock_check),
        patch(
            "duo_workflow_service.interceptors.usage_quota_interceptor.USAGE_QUOTA_CHECK_TOTAL"
        ) as mock_metrics,
    ):
        intercepted_handler = await interceptor.intercept_service(
            continuation, handler_call_details
        )
        assert intercepted_handler is not None

    mock_metrics.labels.assert_called_once_with(result="fail_open", realm="saas")
    mock_metrics.labels.return_value.inc.assert_called_once()


@pytest.mark.asyncio
async def test_customersdot_unexpected_error_raises_exception(
    usage_quota_event_context,
):
    """Test that unexpected error in CustomersDot call raises Exception."""
    interceptor = UsageQuotaInterceptor()

    with patch(
        "duo_workflow_service.interceptors.usage_quota_interceptor.httpx.AsyncClient"
    ) as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.head.side_effect = RuntimeError("Unexpected error")
        mock_client_cls.return_value = mock_client

        with pytest.raises(RuntimeError):
            await interceptor.has_usage_quota_left(usage_quota_event_context)


@pytest.mark.asyncio
async def test_customersdot_forbidden_error_aborts_request(
    continuation, handler_call_details
):
    """Test that forbidden HTTP code from CustomersDot aborts the request."""
    interceptor = UsageQuotaInterceptor()

    with patch(
        "duo_workflow_service.interceptors.usage_quota_interceptor.httpx.AsyncClient"
    ) as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.head.return_value = MagicMock(status_code=403)
        mock_client_cls.return_value = mock_client

        await interceptor.intercept_service(continuation, handler_call_details)
        assert continuation.call_count == 0
