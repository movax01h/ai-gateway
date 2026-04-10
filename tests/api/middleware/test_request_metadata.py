from unittest.mock import AsyncMock, patch

import pytest

from ai_gateway.api.middleware import RequestMetadataMiddleware


@pytest.fixture(name="middleware")
def middleware_fixture(mock_app):
    return RequestMetadataMiddleware(mock_app)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "header_value,expected",
    [
        ("true", True),
        ("false", False),
        ("True", True),
        ("FALSE", False),
    ],
)
async def test_is_gitlab_team_member(middleware, header_value, expected):
    scope = {
        "type": "http",
        "path": "/api/endpoint",
        "headers": [(b"x-gitlab-is-team-member", header_value.encode())],
    }
    receive = AsyncMock()
    send = AsyncMock()

    with patch(
        "ai_gateway.api.middleware.request_metadata.is_gitlab_team_member"
    ) as mock_team_member:
        await middleware(scope, receive, send)

        mock_team_member.set.assert_called_once_with(expected)

    middleware.app.assert_called_once_with(scope, receive, send)


@pytest.mark.asyncio
async def test_is_gitlab_team_member_missing_resets_to_none(middleware):
    scope = {
        "type": "http",
        "path": "/api/endpoint",
        "headers": [(b"other-header", b"value")],
    }
    receive = AsyncMock()
    send = AsyncMock()

    with patch(
        "ai_gateway.api.middleware.request_metadata.is_gitlab_team_member"
    ) as mock_team_member:
        await middleware(scope, receive, send)

        mock_team_member.set.assert_called_once_with(None)

    middleware.app.assert_called_once_with(scope, receive, send)


@pytest.mark.asyncio
async def test_non_http_scope_passthrough(middleware):
    scope = {"type": "websocket", "path": "/ws"}
    receive = AsyncMock()
    send = AsyncMock()

    with patch(
        "ai_gateway.api.middleware.request_metadata.is_gitlab_team_member"
    ) as mock_team_member:
        await middleware(scope, receive, send)

        mock_team_member.set.assert_not_called()

    middleware.app.assert_called_once_with(scope, receive, send)
