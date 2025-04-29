from unittest.mock import AsyncMock, patch

import pytest
from starlette.requests import Request

from ai_gateway.api.middleware import DistributedTraceMiddleware


@pytest.fixture
def distributed_trace_middleware(mock_app):
    return DistributedTraceMiddleware(
        mock_app, skip_endpoints=["/health"], environment="development"
    )


@pytest.mark.asyncio
async def test_middleware_distributed_trace(distributed_trace_middleware):
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

    with patch(
        "ai_gateway.api.middleware.base.tracing_context"
    ) as mock_tracing_context:
        await distributed_trace_middleware(scope, receive, send)

        mock_tracing_context.assert_called_once_with(parent=current_run_id)

    distributed_trace_middleware.app.assert_called_once_with(scope, receive, send)
