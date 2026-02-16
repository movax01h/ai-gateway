from unittest.mock import AsyncMock, patch

import pytest
from starlette.requests import Request
from starlette_context import context, request_cycle_context

from ai_gateway.api.middleware.headers import (
    X_GITLAB_CLIENT_NAME,
    X_GITLAB_CLIENT_TYPE,
    X_GITLAB_CLIENT_VERSION,
    X_GITLAB_FEATURE_ENABLED_BY_NAMESPACE_IDS_HEADER,
    X_GITLAB_FEATURE_ENABLEMENT_TYPE_HEADER,
    X_GITLAB_GLOBAL_USER_ID_HEADER,
    X_GITLAB_HOST_NAME_HEADER,
    X_GITLAB_INSTANCE_ID_HEADER,
    X_GITLAB_INTERFACE,
    X_GITLAB_NAMESPACE_ID,
    X_GITLAB_PROJECT_ID,
    X_GITLAB_REALM_HEADER,
    X_GITLAB_ROOT_NAMESPACE_ID,
    X_GITLAB_SAAS_DUO_PRO_NAMESPACE_IDS_HEADER,
    X_GITLAB_TEAM_MEMBER_HEADER,
    X_GITLAB_VERSION_HEADER,
)
from ai_gateway.api.middleware.internal_event import InternalEventMiddleware
from lib.internal_events import EventContext


@pytest.fixture(name="internal_event_middleware")
def internal_event_middleware_fixture(mock_app):
    return InternalEventMiddleware(
        mock_app, skip_endpoints=["/health"], enabled=True, environment="test"
    )


@pytest.mark.asyncio
async def test_middleware_non_http_request(internal_event_middleware):
    scope = {"type": "websocket"}
    receive = AsyncMock()
    send = AsyncMock()

    with (
        request_cycle_context({}),
        patch(
            "ai_gateway.api.middleware.internal_event.current_event_context"
        ) as mock_event_context,
    ):
        await internal_event_middleware(scope, receive, send)
        mock_event_context.set.assert_not_called()

    internal_event_middleware.app.assert_called_once_with(scope, receive, send)


@pytest.mark.asyncio
async def test_middleware_disabled(mock_app):
    internal_event_middleware = InternalEventMiddleware(
        mock_app, skip_endpoints=[], enabled=False, environment="test"
    )
    request = Request({"type": "http"})
    scope = request.scope
    receive = AsyncMock()
    send = AsyncMock()

    with (
        request_cycle_context({}),
        patch(
            "ai_gateway.api.middleware.internal_event.current_event_context"
        ) as mock_event_context,
    ):
        await internal_event_middleware(scope, receive, send)
        mock_event_context.set.assert_not_called()

    internal_event_middleware.app.assert_called_once_with(scope, receive, send)


@pytest.mark.asyncio
async def test_middleware_skip_path(internal_event_middleware):
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

    with (
        request_cycle_context({}),
        patch(
            "ai_gateway.api.middleware.internal_event.current_event_context"
        ) as mock_event_context,
    ):
        await internal_event_middleware(scope, receive, send)
        mock_event_context.set.assert_not_called()

    internal_event_middleware.app.assert_called_once_with(scope, receive, send)


@pytest.mark.asyncio
async def test_middleware_set_context(internal_event_middleware, user):
    request = Request(
        {
            "type": "http",
            "path": "/api/endpoint",
            "headers": [
                (b"user-agent", b"TestAgent"),
                (X_GITLAB_REALM_HEADER.lower().encode(), b"test-realm"),
                (X_GITLAB_INSTANCE_ID_HEADER.lower().encode(), b"test-instance"),
                (X_GITLAB_HOST_NAME_HEADER.lower().encode(), b"test-host"),
                (X_GITLAB_VERSION_HEADER.lower().encode(), b"test-version"),
                (X_GITLAB_GLOBAL_USER_ID_HEADER.lower().encode(), b"test-user"),
                (X_GITLAB_TEAM_MEMBER_HEADER.lower().encode(), b"true"),
                (X_GITLAB_FEATURE_ENABLEMENT_TYPE_HEADER.lower().encode(), b"duo_pro"),
                (X_GITLAB_CLIENT_NAME.lower().encode(), b"vscode"),
                (X_GITLAB_CLIENT_TYPE.lower().encode(), b"ide"),
                (X_GITLAB_CLIENT_VERSION.lower().encode(), b"1.97.0"),
                (X_GITLAB_INTERFACE.lower().encode(), b"duo_chat"),
                (X_GITLAB_NAMESPACE_ID.lower().encode(), b""),
                (X_GITLAB_PROJECT_ID.lower().encode(), b"456"),
                (X_GITLAB_ROOT_NAMESPACE_ID.lower().encode(), b""),
            ],
            "user": user,
        }
    )
    scope = request.scope
    receive = AsyncMock()
    send = AsyncMock()

    with (
        request_cycle_context({}),
        patch(
            "ai_gateway.api.middleware.internal_event.current_event_context"
        ) as mock_event_context,
        patch(
            "ai_gateway.api.middleware.internal_event.tracked_internal_events"
        ) as mock_tracked_internal_events,
    ):
        await internal_event_middleware(scope, receive, send)

        expected_context = EventContext(
            environment="test",
            source="ai-gateway-python",
            realm="test-realm",
            instance_id="test-instance",
            unique_instance_id="unique-instance-uid",
            host_name="test-host",
            instance_version="test-version",
            global_user_id="test-user",
            is_gitlab_team_member="true",
            feature_enablement_type="duo_pro",
            client_name="vscode",
            client_version="1.97.0",
            client_type="ide",
            interface="duo_chat",
            project_id=456,
            feature_enabled_by_namespace_ids=[],
            context_generated_at=mock_event_context.set.call_args[0][
                0
            ].context_generated_at,
        )
        mock_event_context.set.assert_called_once_with(expected_context)
        mock_tracked_internal_events.set.assert_called_once_with(set())
        assert context["tracked_internal_events"] == []

    internal_event_middleware.app.assert_called_once_with(scope, receive, send)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers, expected",
    [
        (
            [
                (
                    X_GITLAB_FEATURE_ENABLED_BY_NAMESPACE_IDS_HEADER.lower().encode(),
                    b"1,2,3",
                )
            ],
            [1, 2, 3],
        ),
        (
            [
                (
                    X_GITLAB_SAAS_DUO_PRO_NAMESPACE_IDS_HEADER.lower().encode(),
                    b"4,5,6",
                )
            ],
            [4, 5, 6],
        ),
        (
            [
                (
                    X_GITLAB_FEATURE_ENABLED_BY_NAMESPACE_IDS_HEADER.lower().encode(),
                    b"",
                )
            ],
            [],
        ),
        (
            [
                (
                    X_GITLAB_FEATURE_ENABLED_BY_NAMESPACE_IDS_HEADER.lower().encode(),
                    b"1,2,a",
                )
            ],
            [1, 2],
        ),
    ],
)
async def test_middleware_set_context_feature_enabled_by_namespace_ids(
    internal_event_middleware, headers, expected
):
    request = Request(
        {
            "type": "http",
            "path": "/api/endpoint",
            "headers": headers,
            "user": None,
        }
    )
    scope = request.scope
    receive = AsyncMock()
    send = AsyncMock()

    with (
        request_cycle_context({}),
        patch(
            "ai_gateway.api.middleware.internal_event.current_event_context"
        ) as mock_event_context,
    ):
        await internal_event_middleware(scope, receive, send)

        expected_context = EventContext(
            environment="test",
            source="ai-gateway-python",
            realm=None,
            instance_id=None,
            unique_instance_id=None,
            host_name=None,
            instance_version=None,
            global_user_id=None,
            is_gitlab_team_member=None,
            feature_enablement_type=None,
            feature_enabled_by_namespace_ids=expected,
            context_generated_at=mock_event_context.set.call_args[0][
                0
            ].context_generated_at,
        )
        mock_event_context.set.assert_called_once_with(expected_context)

    internal_event_middleware.app.assert_called_once_with(scope, receive, send)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers, expected",
    [
        (
            [
                (
                    X_GITLAB_NAMESPACE_ID.lower().encode(),
                    b"",
                )
            ],
            None,
        ),
        (
            [
                (
                    X_GITLAB_NAMESPACE_ID.lower().encode(),
                    b"123",
                )
            ],
            123,
        ),
        (
            [
                (
                    X_GITLAB_NAMESPACE_ID.lower().encode(),
                    b"null",
                )
            ],
            None,
        ),
    ],
)
async def test_middleware_set_context_namespace_id(
    internal_event_middleware, headers, expected
):
    request = Request(
        {
            "type": "http",
            "path": "/api/endpoint",
            "headers": headers,
            "user": None,
        }
    )
    scope = request.scope
    receive = AsyncMock()
    send = AsyncMock()

    with (
        request_cycle_context({}),
        patch(
            "ai_gateway.api.middleware.internal_event.current_event_context"
        ) as mock_event_context,
    ):
        await internal_event_middleware(scope, receive, send)

        expected_context = EventContext(
            environment="test",
            source="ai-gateway-python",
            realm=None,
            instance_id=None,
            namespace_id=int(expected) if expected else None,
            unique_instance_id=None,
            host_name=None,
            instance_version=None,
            global_user_id=None,
            is_gitlab_team_member=None,
            feature_enablement_type=None,
            feature_enabled_by_namespace_ids=[],
            context_generated_at=mock_event_context.set.call_args[0][
                0
            ].context_generated_at,
        )
        mock_event_context.set.assert_called_once_with(expected_context)

    internal_event_middleware.app.assert_called_once_with(scope, receive, send)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers, expected",
    [
        (
            [
                (
                    X_GITLAB_PROJECT_ID.lower().encode(),
                    b"",
                )
            ],
            None,
        ),
        (
            [
                (
                    X_GITLAB_PROJECT_ID.lower().encode(),
                    b"789",
                )
            ],
            789,
        ),
        (
            [
                (
                    X_GITLAB_PROJECT_ID.lower().encode(),
                    b"null",
                )
            ],
            None,
        ),
    ],
)
async def test_middleware_set_context_project_id(
    internal_event_middleware, headers, expected
):
    request = Request(
        {
            "type": "http",
            "path": "/api/endpoint",
            "headers": headers,
            "user": None,
        }
    )
    scope = request.scope
    receive = AsyncMock()
    send = AsyncMock()

    with (
        request_cycle_context({}),
        patch(
            "ai_gateway.api.middleware.internal_event.current_event_context"
        ) as mock_event_context,
    ):
        await internal_event_middleware(scope, receive, send)

        expected_context = EventContext(
            environment="test",
            source="ai-gateway-python",
            realm=None,
            instance_id=None,
            project_id=int(expected) if expected else None,
            unique_instance_id=None,
            host_name=None,
            instance_version=None,
            global_user_id=None,
            is_gitlab_team_member=None,
            feature_enablement_type=None,
            feature_enabled_by_namespace_ids=[],
            context_generated_at=mock_event_context.set.call_args[0][
                0
            ].context_generated_at,
        )
        mock_event_context.set.assert_called_once_with(expected_context)

    internal_event_middleware.app.assert_called_once_with(scope, receive, send)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers, expected",
    [
        (
            [
                (
                    X_GITLAB_ROOT_NAMESPACE_ID.lower().encode(),
                    b"",
                )
            ],
            None,
        ),
        (
            [
                (
                    X_GITLAB_ROOT_NAMESPACE_ID.lower().encode(),
                    b"999",
                )
            ],
            999,
        ),
        (
            [
                (
                    X_GITLAB_ROOT_NAMESPACE_ID.lower().encode(),
                    b"null",
                )
            ],
            None,
        ),
    ],
)
async def test_middleware_set_context_root_namespace_id(
    internal_event_middleware, headers, expected
):
    request = Request(
        {
            "type": "http",
            "path": "/api/endpoint",
            "headers": headers,
            "user": None,
        }
    )
    scope = request.scope
    receive = AsyncMock()
    send = AsyncMock()

    with (
        request_cycle_context({}),
        patch(
            "ai_gateway.api.middleware.internal_event.current_event_context"
        ) as mock_event_context,
    ):
        await internal_event_middleware(scope, receive, send)

        expected_context = EventContext(
            environment="test",
            source="ai-gateway-python",
            realm=None,
            instance_id=None,
            ultimate_parent_namespace_id=int(expected) if expected else None,
            unique_instance_id=None,
            host_name=None,
            instance_version=None,
            global_user_id=None,
            is_gitlab_team_member=None,
            feature_enablement_type=None,
            feature_enabled_by_namespace_ids=[],
            context_generated_at=mock_event_context.set.call_args[0][
                0
            ].context_generated_at,
        )
        mock_event_context.set.assert_called_once_with(expected_context)

    internal_event_middleware.app.assert_called_once_with(scope, receive, send)


@pytest.mark.asyncio
async def test_middleware_missing_headers(internal_event_middleware):
    request = Request(
        {
            "type": "http",
            "path": "/api/endpoint",
            "headers": [
                (b"user-agent", b"TestAgent"),
            ],
            "user": None,
        }
    )
    scope = request.scope
    receive = AsyncMock()
    send = AsyncMock()

    with (
        request_cycle_context({}),
        patch(
            "ai_gateway.api.middleware.internal_event.current_event_context"
        ) as mock_event_context,
    ):
        await internal_event_middleware(scope, receive, send)

        expected_context = EventContext(
            environment="test",
            source="ai-gateway-python",
            realm=None,
            instance_id=None,
            unique_instance_id=None,
            host_name=None,
            instance_version=None,
            global_user_id=None,
            feature_enabled_by_namespace_ids=[],
            context_generated_at=mock_event_context.set.call_args[0][
                0
            ].context_generated_at,
        )
        mock_event_context.set.assert_called_once_with(expected_context)

    internal_event_middleware.app.assert_called_once_with(scope, receive, send)
