from unittest.mock import AsyncMock, patch

import pytest
from starlette.requests import Request
from starlette_context import request_cycle_context

from ai_gateway.api.middleware.access_log import AccessLogMiddleware


@pytest.fixture(name="access_log_middleware")
def access_log_middleware_fixture(mock_app):
    return AccessLogMiddleware(mock_app, skip_endpoints=["/health"])


@pytest.mark.asyncio
async def test_middleware_non_http_request(access_log_middleware):
    scope = {"type": "websocket"}
    receive = AsyncMock()
    send = AsyncMock()

    await access_log_middleware(scope, receive, send)

    access_log_middleware.app.assert_called_once_with(scope, receive, send)


@pytest.mark.asyncio
async def test_middleware_skip_path(access_log_middleware):
    request = Request(
        {
            "type": "http",
            "path": "/health",
            "headers": [],
        }
    )
    scope = request.scope
    receive = AsyncMock()
    send = AsyncMock()

    await access_log_middleware(scope, receive, send)

    access_log_middleware.app.assert_called_once_with(scope, receive, send)


@pytest.mark.asyncio
async def test_middleware_logs_request(access_log_middleware):
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/endpoint",
            "query_string": b"",
            "http_version": "1.1",
            "headers": [
                (b"user-agent", b"TestAgent"),
            ],
            "server": ("testserver", 80),
        }
    )
    scope = request.scope
    receive = AsyncMock()
    send = AsyncMock()

    async def mock_app(_scope, _receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    access_log_middleware.app = mock_app

    with (
        request_cycle_context({}),
        patch("ai_gateway.api.middleware.access_log.access_logger") as mock_logger,
    ):
        await access_log_middleware(scope, receive, send)
        assert mock_logger.info.called


@pytest.mark.asyncio
async def test_middleware_handles_exception(access_log_middleware):
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/endpoint",
            "query_string": b"",
            "http_version": "1.1",
            "headers": [],
            "server": ("testserver", 80),
        }
    )
    scope = request.scope
    receive = AsyncMock()
    send = AsyncMock()

    async def mock_app_raises(_scope, _receive, _send):
        raise ValueError("something went wrong")

    access_log_middleware.app = mock_app_raises

    with (
        request_cycle_context({}),
        patch("ai_gateway.api.middleware.access_log.log_exception"),
        patch("ai_gateway.api.middleware.access_log.access_logger"),
    ):
        with pytest.raises(ValueError, match="something went wrong"):
            await access_log_middleware(scope, receive, send)


@pytest.mark.asyncio
async def test_middleware_adds_process_time_header(access_log_middleware):
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/endpoint",
            "query_string": b"",
            "http_version": "1.1",
            "headers": [],
            "server": ("testserver", 80),
        }
    )
    scope = request.scope
    receive = AsyncMock()
    sent_messages = []

    async def capture_send(message):
        sent_messages.append(message)

    async def mock_app(_scope, _receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    access_log_middleware.app = mock_app

    with (
        request_cycle_context({}),
        patch("ai_gateway.api.middleware.access_log.access_logger"),
    ):
        await access_log_middleware(scope, receive, capture_send)

    response_start = next(
        m for m in sent_messages if m["type"] == "http.response.start"
    )
    header_names = [h[0].decode() for h in response_start["headers"]]
    assert "x-process-time" in header_names
