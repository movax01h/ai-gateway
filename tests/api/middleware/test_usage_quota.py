import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from starlette.requests import Request
from starlette.responses import Response

from ai_gateway.api.middleware.usage_quota import UsageQuotaMiddleware
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
        "ai_gateway.api.middleware.usage_quota.current_event_context"
    ) as mock_ctx:
        mock_ctx.get.return_value = internal_event_context
        yield


@pytest.fixture(autouse=True)
def stub_feature_flags():
    token = current_feature_flag_context.set({"usage_quota_left_check"})
    yield
    current_feature_flag_context.reset(token)


@pytest.fixture(name="usage_quota_middleware")
def usage_quota_middleware_fixture(mock_app):
    """Create an UsageQuotaMiddleware instance for testing."""
    middleware = UsageQuotaMiddleware(
        app=mock_app,
        customersdot_url="https://customers.gitlab.local",
        skip_endpoints=["/health", "/metrics"],
        enabled=True,
        environment="test",
        request_timeout=1.0,
    )
    return middleware


@pytest.mark.asyncio
async def test_skips_endpoint_in_skip_list(usage_quota_middleware):
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
    response = await usage_quota_middleware.dispatch(request, call_next)

    assert response.status_code == 200
    call_next.assert_called_once_with(request)


@pytest.mark.asyncio
async def test_skips_metrics_endpoint(usage_quota_middleware):
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
    response = await usage_quota_middleware.dispatch(request, call_next)

    assert response.status_code == 200
    call_next.assert_called_once_with(request)


@pytest.mark.asyncio
async def test_feature_flag_disabled_skips_usage_quota_check(usage_quota_middleware):
    """Test that middleware skips usage quota check when feature flag is disabled."""
    request = Request(
        {"type": "http", "path": "/api/v1/chat", "method": "POST", "headers": []}
    )
    call_next = AsyncMock(return_value=Response(status_code=200))

    with (
        patch("ai_gateway.api.middleware.usage_quota.is_feature_enabled") as mock_ff,
        patch.object(
            usage_quota_middleware,
            "has_usage_quota_left",
            new_callable=AsyncMock,
        ) as mock_check,
    ):
        mock_ff.return_value = False

        response = await usage_quota_middleware.dispatch(request, call_next)

    assert response.status_code == 200
    call_next.assert_called_once_with(request)
    mock_check.assert_not_called()


@pytest.mark.asyncio
async def test_feature_flag_enabled_performs_usage_quota_check(
    usage_quota_middleware, internal_event_context, usage_quota_event_context
):
    """Test that middleware performs usage quota check when feature flag is enabled."""
    request = Request(
        {"type": "http", "path": "/api/v1/chat", "method": "POST", "headers": []}
    )
    call_next = AsyncMock(return_value=Response(status_code=200))
    mock_check = AsyncMock(return_value=True)

    with (
        patch(
            "ai_gateway.api.middleware.usage_quota.current_event_context"
        ) as mock_ctx,
        patch.object(usage_quota_middleware, "has_usage_quota_left", mock_check),
    ):
        mock_ctx.get.return_value = internal_event_context

        response = await usage_quota_middleware.dispatch(request, call_next)

    assert response.status_code == 200
    call_next.assert_called_once_with(request)
    mock_check.assert_called_once_with(usage_quota_event_context)


@pytest.mark.asyncio
async def test_middleware_disabled_skips_usage_quota_check(mock_app):
    """Test that middleware skips usage quota check when middleware is disabled."""
    middleware = UsageQuotaMiddleware(
        app=mock_app,
        customersdot_url="https://customers.gitlab.local",
        skip_endpoints=[],
        enabled=False,  # disabled
        environment="test",
        request_timeout=1.0,
    )
    request = Request(
        {"type": "http", "path": "/api/v1/chat", "method": "POST", "headers": []}
    )
    call_next = AsyncMock(return_value=Response(status_code=200))
    response = await middleware.dispatch(request, call_next)

    assert response.status_code == 200
    call_next.assert_called_once_with(request)


@pytest.mark.asyncio
async def test_authorized_request_passes_through(usage_quota_middleware):
    """Test that an authorized request is allowed through."""
    request = Request(
        {"type": "http", "path": "/api/v1/chat", "method": "POST", "headers": []}
    )
    call_next = AsyncMock(return_value=Response(status_code=200))
    mock_check = AsyncMock(return_value=True)

    with (patch.object(usage_quota_middleware, "has_usage_quota_left", mock_check),):
        response = await usage_quota_middleware.dispatch(request, call_next)

    assert response.status_code == 200
    call_next.assert_called_once_with(request)


@pytest.mark.asyncio
async def test_no_usage_quota_left_returns_402(usage_quota_middleware):
    """Test that a usage_quota left request returns 402 status code."""
    request = Request(
        {"type": "http", "path": "/api/v1/chat", "method": "POST", "headers": []}
    )
    call_next = AsyncMock(return_value=Response(status_code=200))
    mock_check = AsyncMock(return_value=False)

    with (patch.object(usage_quota_middleware, "has_usage_quota_left", mock_check),):
        response = await usage_quota_middleware.dispatch(request, call_next)

    assert response.status_code == 402
    call_next.assert_not_called()


@pytest.mark.asyncio
async def test_no_usage_quota_left_response_format(usage_quota_middleware):
    """Test that 402 error response has correct JSON format."""
    request = Request(
        {"type": "http", "path": "/api/v1/chat", "method": "POST", "headers": []}
    )
    call_next = AsyncMock(return_value=Response(status_code=200))
    mock_check = AsyncMock(return_value=False)

    with (patch.object(usage_quota_middleware, "has_usage_quota_left", mock_check),):
        response = await usage_quota_middleware.dispatch(request, call_next)

    assert response.status_code == 402
    content = json.loads(response.body.decode())
    assert content["error"] == "insufficient_credits"
    assert content["error_code"] == "USAGE_QUOTA_EXCEEDED"
    assert "sufficient credits" in content["message"]


@pytest.mark.asyncio
async def test_dispatch_fails_open_on_timeout(usage_quota_middleware):
    """Test that dispatch allows request when timeout occurs (fail-open)."""
    request = Request(
        {"type": "http", "path": "/api/v1/chat", "method": "POST", "headers": []}
    )
    call_next = AsyncMock(return_value=Response(status_code=200))
    mock_check = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

    with (patch.object(usage_quota_middleware, "has_usage_quota_left", mock_check),):
        response = await usage_quota_middleware.dispatch(request, call_next)

    assert response.status_code == 200
    call_next.assert_called_once_with(request)


@pytest.mark.asyncio
async def test_customersdot_success_returns_true(usage_quota_event_context):
    """Test that CustomersDot 200 response returns True (entitled)."""
    middleware = UsageQuotaMiddleware(
        app=AsyncMock(),
        customersdot_url="https://customers.gitlab.local",
        skip_endpoints=[],
        enabled=True,
        environment="test",
        request_timeout=1.0,
    )

    mock_response = MagicMock(status_code=200)
    with patch(
        "ai_gateway.api.middleware.usage_quota.httpx.AsyncClient"
    ) as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.head.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = await middleware.has_usage_quota_left(usage_quota_event_context)

    assert result is True


@pytest.mark.asyncio
async def test_customersdot_denied_status_returns_false(usage_quota_event_context):
    """Test that CustomersDot 402 responses return False (not entitled)."""
    UsageQuotaMiddleware.has_usage_quota_left.cache.clear()

    middleware = UsageQuotaMiddleware(
        app=AsyncMock(),
        customersdot_url="https://customers.gitlab.local",
        skip_endpoints=[],
        enabled=True,
        environment="test",
        request_timeout=1.0,
    )

    mock_response = MagicMock(status_code=402)

    with patch(
        "ai_gateway.api.middleware.usage_quota.httpx.AsyncClient"
    ) as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.head.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = await middleware.has_usage_quota_left(usage_quota_event_context)
        assert result is False

    UsageQuotaMiddleware.has_usage_quota_left.cache.clear()


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
    usage_quota_event_context,
    exception_class,
    exception_kwargs,
    expected_metric_outcome,
    expected_metric_status,
):
    """Test that different HTTP-related exceptions are properly raised and metrics recorded."""
    middleware = UsageQuotaMiddleware(
        app=AsyncMock(),
        customersdot_url="https://customers.gitlab.local",
        skip_endpoints=[],
        enabled=True,
        environment="test",
        request_timeout=1.0,
    )

    with (
        patch(
            "ai_gateway.api.middleware.usage_quota.httpx.AsyncClient"
        ) as mock_client_cls,
        patch(
            "ai_gateway.api.middleware.usage_quota.USAGE_QUOTA_CUSTOMERSDOT_REQUESTS_TOTAL"
        ) as mock_metrics,
    ):
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        exception = exception_class(**exception_kwargs)
        mock_client.head.side_effect = exception
        mock_client_cls.return_value = mock_client

        with pytest.raises(exception_class):
            await middleware.has_usage_quota_left(usage_quota_event_context)

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
    usage_quota_middleware, exception_class, exception_kwargs
):
    """Test that dispatch allows requests when HTTP exceptions occur (fail-open)."""
    request = Request(
        {"type": "http", "path": "/api/v1/chat", "method": "POST", "headers": []}
    )
    call_next = AsyncMock(return_value=Response(status_code=200))
    mock_check = AsyncMock(side_effect=exception_class(**exception_kwargs))

    with (
        patch.object(usage_quota_middleware, "has_usage_quota_left", mock_check),
        patch(
            "ai_gateway.api.middleware.usage_quota.USAGE_QUOTA_CHECK_TOTAL"
        ) as mock_metrics,
    ):
        response = await usage_quota_middleware.dispatch(request, call_next)

    assert response.status_code == 200
    call_next.assert_called_once_with(request)

    mock_metrics.labels.assert_called_once_with(result="fail_open", realm="saas")
    mock_metrics.labels.return_value.inc.assert_called_once()


@pytest.mark.asyncio
async def test_customersdot_unexpected_error_raises_exception(
    usage_quota_event_context,
):
    """Test that unexpected error in CustomersDot call raises Exception."""
    middleware = UsageQuotaMiddleware(
        app=AsyncMock(),
        customersdot_url="https://customers.gitlab.local",
        skip_endpoints=[],
        enabled=True,
        environment="test",
        request_timeout=1.0,
    )

    with patch(
        "ai_gateway.api.middleware.usage_quota.httpx.AsyncClient"
    ) as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.head.side_effect = RuntimeError("Unexpected error")
        mock_client_cls.return_value = mock_client

        with pytest.raises(RuntimeError):
            await middleware.has_usage_quota_left(usage_quota_event_context)
