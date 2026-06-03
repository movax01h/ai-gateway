from unittest.mock import AsyncMock, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims
from starlette.requests import Request
from starlette_context import context, request_cycle_context

from ai_gateway.api.middleware.headers import (
    X_GITLAB_CLIENT_NAME,
    X_GITLAB_CLIENT_TYPE,
    X_GITLAB_CLIENT_VERSION,
    X_GITLAB_DEPLOYMENT_TYPE,
    X_GITLAB_FEATURE_ENABLED_BY_NAMESPACE_IDS_HEADER,
    X_GITLAB_FEATURE_ENABLEMENT_TYPE_HEADER,
    X_GITLAB_GLOBAL_USER_ID_HEADER,
    X_GITLAB_HOST_NAME_HEADER,
    X_GITLAB_INSTANCE_ID_HEADER,
    X_GITLAB_INTERFACE,
    X_GITLAB_NAMESPACE_ID,
    X_GITLAB_ORGANIZATION_ID,
    X_GITLAB_PROJECT_ID,
    X_GITLAB_REALM_HEADER,
    X_GITLAB_ROOT_NAMESPACE_ID,
    X_GITLAB_SAAS_DUO_PRO_NAMESPACE_IDS_HEADER,
    X_GITLAB_SUBJECT_TYPE,
    X_GITLAB_TEAM_MEMBER_HEADER,
    X_GITLAB_VERSION_HEADER,
)
from ai_gateway.api.middleware.internal_event import InternalEventMiddleware
from lib.context.auth import StarletteUser
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
                (X_GITLAB_SUBJECT_TYPE.lower().encode(), b"service_account"),
                (X_GITLAB_ORGANIZATION_ID.lower().encode(), b"1337"),
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
            organization_id=1337,
            project_id=456,
            feature_enabled_by_namespace_ids=[],
            user_type="service_account",
            context_generated_at=mock_event_context.set.call_args[0][
                0
            ].context_generated_at,
        )
        mock_event_context.set.assert_called_once_with(expected_context)
        mock_tracked_internal_events.set.assert_called_once_with(set())
        assert context["tracked_internal_events"] == []

    internal_event_middleware.app.assert_called_once_with(scope, receive, send)


def test_x_gitlab_subject_type_constant_value():
    """Lock the canonical header name to catch typos in the constant."""
    assert X_GITLAB_SUBJECT_TYPE == "X-Gitlab-Subject-Type"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "header_value,expected",
    [
        (b"human", "human"),
        (b"service_account", "service_account"),
        (b"bot", "bot"),
        (b"", ""),
        (None, None),
    ],
)
async def test_middleware_propagates_subject_type_header(
    internal_event_middleware, user, header_value, expected
):
    """The x-gitlab-subject-type header is propagated to EventContext.user_type."""
    headers: list[tuple[bytes, bytes]] = [
        (X_GITLAB_REALM_HEADER.lower().encode(), b"test-realm"),
        (X_GITLAB_INSTANCE_ID_HEADER.lower().encode(), b"test-instance"),
        (X_GITLAB_HOST_NAME_HEADER.lower().encode(), b"test-host"),
        (X_GITLAB_GLOBAL_USER_ID_HEADER.lower().encode(), b"test-user"),
    ]
    if header_value is not None:
        headers.append((X_GITLAB_SUBJECT_TYPE.lower().encode(), header_value))

    request = Request(
        {
            "type": "http",
            "path": "/api/endpoint",
            "headers": headers,
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
    ):
        await internal_event_middleware(scope, receive, send)

        set_context = mock_event_context.set.call_args[0][0]
        assert set_context.user_type == expected


@pytest.mark.asyncio
async def test_middleware_logs_per_field_warning_when_required_fields_missing(
    internal_event_middleware,
):
    request = Request(
        {
            "type": "http",
            "path": "/api/endpoint",
            "headers": [],
            "user": None,
        }
    )
    scope = request.scope
    receive = AsyncMock()
    send = AsyncMock()

    with (
        request_cycle_context({}),
        patch("lib.internal_events.context_validator.log") as mock_log,
    ):
        await internal_event_middleware(scope, receive, send)

        # One warning per missing required field
        warning_fields = {
            c[1]["missing_field"] for c in mock_log.warning.call_args_list
        }
        assert {
            "realm",
            "instance_id",
            "global_user_id",
            "host_name",
            "unique_instance_id",
            "deployment_type",
        }.issubset(warning_fields)

        # Each call has the right structure
        for c in mock_log.warning.call_args_list:
            assert c[0][0] == "Internal event context missing required field"
            assert c[1]["field_type"] == "required"
            assert c[1]["endpoint"] == "/api/endpoint"
            assert "correlation_id" in c[1]


@pytest.mark.asyncio
async def test_middleware_logs_per_field_info_when_contextual_fields_missing(
    internal_event_middleware, user
):
    request = Request(
        {
            "type": "http",
            "path": "/api/endpoint",
            "headers": [
                (X_GITLAB_REALM_HEADER.lower().encode(), b"saas"),
                (X_GITLAB_INSTANCE_ID_HEADER.lower().encode(), b"inst-123"),
                (X_GITLAB_HOST_NAME_HEADER.lower().encode(), b"gitlab.com"),
                (X_GITLAB_GLOBAL_USER_ID_HEADER.lower().encode(), b"user-456"),
                (X_GITLAB_DEPLOYMENT_TYPE.lower().encode(), b".com"),
                (X_GITLAB_PROJECT_ID.lower().encode(), b"42"),
                (X_GITLAB_NAMESPACE_ID.lower().encode(), b"7"),
            ],
            "user": user,
        }
    )
    scope = request.scope
    receive = AsyncMock()
    send = AsyncMock()

    with (
        request_cycle_context({}),
        patch("lib.internal_events.context_validator.log") as mock_log,
    ):
        await internal_event_middleware(scope, receive, send)

        # No warnings — all required fields present
        mock_log.warning.assert_not_called()

        # One info per missing contextual field
        info_fields = {c[1]["missing_field"] for c in mock_log.info.call_args_list}
        # feature_enabled_by_namespace_ids defaults to [] (not None) so it's not missing
        assert "is_gitlab_team_member" in info_fields
        assert "ultimate_parent_namespace_id" in info_fields

        for c in mock_log.info.call_args_list:
            assert c[0][0] == "Internal event context missing contextual field"
            assert c[1]["field_type"] == "contextual"


@pytest.mark.asyncio
async def test_middleware_no_logs_when_all_fields_present(
    internal_event_middleware, user
):
    request = Request(
        {
            "type": "http",
            "path": "/api/endpoint",
            "headers": [
                (X_GITLAB_REALM_HEADER.lower().encode(), b"saas"),
                (X_GITLAB_INSTANCE_ID_HEADER.lower().encode(), b"inst-123"),
                (X_GITLAB_HOST_NAME_HEADER.lower().encode(), b"gitlab.com"),
                (X_GITLAB_GLOBAL_USER_ID_HEADER.lower().encode(), b"user-456"),
                (X_GITLAB_DEPLOYMENT_TYPE.lower().encode(), b".com"),
                (
                    X_GITLAB_FEATURE_ENABLED_BY_NAMESPACE_IDS_HEADER.lower().encode(),
                    b"1,2",
                ),
                (X_GITLAB_TEAM_MEMBER_HEADER.lower().encode(), b"true"),
                (X_GITLAB_ROOT_NAMESPACE_ID.lower().encode(), b"99"),
                (X_GITLAB_PROJECT_ID.lower().encode(), b"42"),
                (X_GITLAB_NAMESPACE_ID.lower().encode(), b"7"),
            ],
            "user": user,
        }
    )
    scope = request.scope
    receive = AsyncMock()
    send = AsyncMock()

    with (
        request_cycle_context({}),
        patch("lib.internal_events.context_validator.log") as mock_log,
    ):
        await internal_event_middleware(scope, receive, send)

        mock_log.warning.assert_not_called()
        mock_log.info.assert_not_called()


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
@pytest.mark.parametrize(
    "jwt_extra, header_value, expected",
    [
        ({"gitlab_root_namespace_id": 42}, b"999", 999),
        ({"gitlab_root_namespace_id": 77}, None, 77),
        ({"gitlab_root_namespace_id": 42.0}, None, 42),
        ({"gitlab_root_namespace_id": "not-a-number"}, None, None),
        ({}, None, None),
        ({"gitlab_root_namespace_id": None}, None, None),
        ({"gitlab_root_namespace_id": 42}, b"", None),
        ({"gitlab_root_namespace_id": 42}, b"null", None),
        (None, b"888", 888),
        (None, None, None),
    ],
    ids=[
        "header_wins_over_jwt",
        "jwt_fallback_no_header",
        "jwt_float_normalised",
        "jwt_invalid_string_no_crash",
        "jwt_missing_key_no_header",
        "jwt_null_no_header",
        "empty_header_no_jwt_fallback",
        "null_sentinel_header_no_jwt_fallback",
        "no_claims_header",
        "no_claims_no_header",
    ],
)
async def test_middleware_root_namespace_id_header_first_jwt_fallback(
    internal_event_middleware, jwt_extra, header_value, expected
):
    claims = (
        UserClaims(
            gitlab_realm="saas",
            scopes=[],
            gitlab_instance_uid="test-uid",
            issuer="",
            extra=jwt_extra,
        )
        if jwt_extra is not None
        else None
    )
    wrapped = StarletteUser(CloudConnectorUser(authenticated=True, claims=claims))

    headers: list[tuple[bytes, bytes]] = []
    if header_value is not None:
        headers.append((X_GITLAB_ROOT_NAMESPACE_ID.lower().encode(), header_value))

    request = Request(
        {
            "type": "http",
            "path": "/api/endpoint",
            "headers": headers,
            "user": wrapped,
        }
    )

    with (
        request_cycle_context({}),
        patch(
            "ai_gateway.api.middleware.internal_event.current_event_context"
        ) as mock_event_context,
    ):
        await internal_event_middleware(request.scope, AsyncMock(), AsyncMock())

        set_context = mock_event_context.set.call_args[0][0]
        assert set_context.ultimate_parent_namespace_id == expected


@pytest.mark.asyncio
async def test_middleware_root_namespace_id_jwt_fallback_ignored_for_non_saas(
    internal_event_middleware,
):
    claims = UserClaims(
        gitlab_realm="self-managed",
        scopes=[],
        gitlab_instance_uid="test-uid",
        issuer="",
        extra={"gitlab_root_namespace_id": 42},
    )
    wrapped = StarletteUser(CloudConnectorUser(authenticated=True, claims=claims))

    request = Request(
        {
            "type": "http",
            "path": "/api/endpoint",
            "headers": [],
            "user": wrapped,
        }
    )

    with (
        request_cycle_context({}),
        patch(
            "ai_gateway.api.middleware.internal_event.current_event_context"
        ) as mock_event_context,
    ):
        await internal_event_middleware(request.scope, AsyncMock(), AsyncMock())

        set_context = mock_event_context.set.call_args[0][0]
        assert set_context.ultimate_parent_namespace_id is None


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
