import os
from unittest.mock import AsyncMock, patch

import pytest
from starlette.requests import Request

from ai_gateway.api.middleware import DistributedTraceMiddleware


@pytest.fixture
def distributed_trace_middleware_development(mock_app):
    return DistributedTraceMiddleware(
        mock_app, skip_endpoints=["/health"], environment="development"
    )


@pytest.fixture
def distributed_trace_middleware_test(mock_app):
    return DistributedTraceMiddleware(
        mock_app, skip_endpoints=["/health"], environment="test"
    )


@pytest.mark.asyncio
async def test_middleware_distributed_trace_enabled_in_development(
    distributed_trace_middleware_development,
):
    """Test that langsmith tracing is enabled in development environment with langsmith-trace header."""
    current_run_id = "20240808T090953171943Z18dfa1db-1dfc-4a48-aaf8-a139960955ce"
    request = Request(
        {
            "type": "http",
            "path": "/api/endpoint",
            "headers": [
                (b"langsmith-trace", current_run_id.encode()),
            ],
        }
    )
    scope = request.scope
    receive = AsyncMock()
    send = AsyncMock()

    # pylint: disable=direct-environment-variable-reference
    with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"}):
        with patch(
            "ai_gateway.api.middleware.base.tracing_context"
        ) as mock_tracing_context:
            await distributed_trace_middleware_development(scope, receive, send)

            mock_tracing_context.assert_called_once_with(
                parent=current_run_id, enabled=True
            )
    # pylint: enable=direct-environment-variable-reference

    distributed_trace_middleware_development.app.assert_called_once_with(
        scope, receive, send
    )


@pytest.mark.asyncio
async def test_middleware_distributed_trace_disabled_in_development_without_header(
    distributed_trace_middleware_development,
):
    """Test that langsmith tracing is disabled in development environment without langsmith-trace header."""
    request = Request(
        {
            "type": "http",
            "path": "/api/endpoint",
            "headers": [],
        }
    )
    scope = request.scope
    receive = AsyncMock()
    send = AsyncMock()

    # pylint: disable=direct-environment-variable-reference
    with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"}):
        with patch(
            "ai_gateway.api.middleware.base.tracing_context"
        ) as mock_tracing_context:
            await distributed_trace_middleware_development(scope, receive, send)

            # tracing_context should be called with parent=None and enabled=True in development
            mock_tracing_context.assert_called_once_with(parent=None, enabled=True)
    # pylint: enable=direct-environment-variable-reference

    distributed_trace_middleware_development.app.assert_called_once_with(
        scope, receive, send
    )


@pytest.mark.asyncio
async def test_middleware_distributed_trace_disabled_in_non_development_environment(
    distributed_trace_middleware_test,
):
    """Test that langsmith tracing is disabled when environment is not development."""
    current_run_id = "20240808T090953171943Z18dfa1db-1dfc-4a48-aaf8-a139960955ce"
    request = Request(
        {
            "type": "http",
            "path": "/api/endpoint",
            "headers": [
                (b"langsmith-trace", current_run_id.encode()),
            ],
        }
    )
    scope = request.scope
    receive = AsyncMock()
    send = AsyncMock()

    # pylint: disable=direct-environment-variable-reference
    with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"}):
        with patch(
            "ai_gateway.api.middleware.base.tracing_context"
        ) as mock_tracing_context:
            await distributed_trace_middleware_test(scope, receive, send)

            # tracing_context is called but with enabled=False in non-development environments
            mock_tracing_context.assert_called_once_with(
                parent=current_run_id, enabled=False
            )
    # pylint: enable=direct-environment-variable-reference

    distributed_trace_middleware_test.app.assert_called_once_with(scope, receive, send)
