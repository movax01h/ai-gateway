from unittest.mock import Mock

import pytest
from fastapi import HTTPException, Response
from gitlab_cloud_connector import CloudConnectorUser, UserClaims

from ai_gateway.api.v1.proxy.request import (
    _check_proxy_endpoints_enabled,
    verify_project_namespace_metadata,
)
from ai_gateway.config import ConfigProxyEndpoints
from lib.context import StarletteUser
from lib.internal_events.context import EventContext, current_event_context

# ---------------------------------------------------------------------------
# verify_project_namespace_metadata tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra_claims, event_context, expected_detail",
    [
        (
            {
                "gitlab_project_id": "999",
                "gitlab_namespace_id": "456",
                "gitlab_root_namespace_id": "789",
            },
            EventContext(
                project_id=123, namespace_id=456, ultimate_parent_namespace_id=789
            ),
            "project id mismatch",
        ),
        (
            {
                "gitlab_project_id": "123",
                "gitlab_namespace_id": "999",
                "gitlab_root_namespace_id": "789",
            },
            EventContext(
                project_id=123, namespace_id=456, ultimate_parent_namespace_id=789
            ),
            "namespace id mismatch",
        ),
        (
            {
                "gitlab_project_id": "123",
                "gitlab_namespace_id": "456",
                "gitlab_root_namespace_id": "999",
            },
            EventContext(
                project_id=123, namespace_id=456, ultimate_parent_namespace_id=789
            ),
            "root namespace id mismatch",
        ),
    ],
)
async def test_verify_project_namespace_metadata_saas_mismatch(
    mock_request, extra_claims, event_context, expected_detail
):
    """SaaS verification fails when project, namespace, or root namespace ID doesn't match."""
    mock_request.user = StarletteUser(
        CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(gitlab_realm="saas", extra=extra_claims),
        )
    )
    current_event_context.set(event_context)

    @verify_project_namespace_metadata()
    async def dummy_func(_request, *_args, **_kwargs):
        return Response(content=b'{"message": "success"}', status_code=200)

    with pytest.raises(HTTPException) as exc_info:
        await dummy_func(mock_request)

    assert exc_info.value.status_code == 403
    assert expected_detail in exc_info.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "instance_uid, event_context, expect_success",
    [
        (
            "instance-uid-123",
            EventContext(instance_id="instance-uid-123"),
            True,
        ),
        (
            "wrong-instance-uid",
            EventContext(instance_id="instance-uid-123"),
            False,
        ),
        (
            # Matching UID; mismatched project/namespace IDs in extra claims should be ignored.
            "instance-uid-123",
            EventContext(
                instance_id="instance-uid-123",
                project_id=123,
                namespace_id=456,
                ultimate_parent_namespace_id=789,
            ),
            True,
        ),
    ],
)
async def test_verify_project_namespace_metadata_self_managed(
    mock_request, instance_uid, event_context, expect_success
):
    """Self-managed verification checks only the instance UID."""
    mock_request.user = StarletteUser(
        CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(
                gitlab_realm="self-managed",
                gitlab_instance_uid=instance_uid,
                extra={
                    "gitlab_instance_uid": instance_uid,
                    "gitlab_project_id": "999",
                    "gitlab_namespace_id": "999",
                    "gitlab_root_namespace_id": "999",
                },
            ),
        )
    )
    current_event_context.set(event_context)

    @verify_project_namespace_metadata()
    async def dummy_func(_request, *_args, **_kwargs):
        return Response(content=b'{"message": "success"}', status_code=200)

    if expect_success:
        response = await dummy_func(mock_request)
        assert response.status_code == 200
    else:
        with pytest.raises(HTTPException) as exc_info:
            await dummy_func(mock_request)
        assert exc_info.value.status_code == 403
        assert "instance uid mismatch" in exc_info.value.detail


# ---------------------------------------------------------------------------
# check_proxy_endpoints_enabled tests
# ---------------------------------------------------------------------------
# Tests call _check_proxy_endpoints_enabled directly, passing proxy_cfg as an
# explicit keyword argument to bypass @inject and avoid needing a wired DI
# container in unit tests.


@pytest.fixture(name="proxy_kwargs")
def proxy_kwargs_fixture():
    return {}


@pytest.fixture(name="jwt_realm")
def jwt_realm_fixture():
    return ""


@pytest.fixture(name="instance_uid")
def instance_uid_fixture():
    return ""


@pytest.fixture(name="root_namespace_id")
def root_namespace_id_fixture():
    return ""


@pytest.fixture(name="proxy_cfg")
def proxy_cfg_fixture(proxy_kwargs) -> ConfigProxyEndpoints:
    return ConfigProxyEndpoints(**proxy_kwargs)


@pytest.fixture(name="proxy_mock_request")
def proxy_mock_request_fixture(jwt_realm, instance_uid, root_namespace_id) -> Mock:
    extra = {}
    if root_namespace_id:
        extra["gitlab_root_namespace_id"] = root_namespace_id
    mock = Mock()
    mock.user = StarletteUser(
        CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(
                gitlab_realm=jwt_realm or None,
                gitlab_instance_uid=instance_uid or None,
                extra=extra,
            ),
        )
    )
    return mock


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "proxy_kwargs, jwt_realm, instance_uid, root_namespace_id",
    [
        # Default config (empty allowlist) — all instances allowed
        ({}, "self-managed", "uid-1", ""),
        ({}, "saas", "", "42"),
        # saas_enabled set — does not restrict self-managed
        ({"saas_enabled": "42"}, "self-managed", "uid-1", ""),
        # self_managed_enabled set — does not restrict SaaS
        ({"self_managed_enabled": "uid-1"}, "saas", "", "42"),
        # Instance UID is in the self-managed allowlist
        ({"self_managed_enabled": "uid-1,uid-2"}, "self-managed", "uid-1", ""),
        # Namespace ID is in the SaaS allowlist
        ({"saas_enabled": "12345,67890"}, "saas", "", "12345"),
        # Whitespace around IDs is trimmed
        ({"self_managed_enabled": " uid-a , uid-b "}, "self-managed", "uid-a", ""),
        # Empty JWT realm with empty allowlist — allowed
        ({}, "", "", ""),
        # Empty JWT realm with only saas allowlist set — allowed (saas check not applied)
        ({"saas_enabled": "42"}, "", "", ""),
        # Unknown realm with empty self-managed allowlist — allowed
        ({}, "unknown-realm", "uid-1", ""),
    ],
)
async def test_check_proxy_endpoints_enabled_allows(proxy_cfg, proxy_mock_request):
    """Requests that should pass through the allowlist check."""
    await _check_proxy_endpoints_enabled(
        proxy_mock_request,
        proxy_cfg=proxy_cfg,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "proxy_kwargs, jwt_realm, instance_uid, root_namespace_id",
    [
        # Allowlist set but instance UID is not in it
        ({"self_managed_enabled": "uid-1,uid-2"}, "self-managed", "uid-other", ""),
        # Allowlist set but namespace ID is not in it
        ({"saas_enabled": "12345,67890"}, "saas", "", "99999"),
        # Allowlist set for both realms — self-managed UID not present
        (
            {"self_managed_enabled": "uid-1", "saas_enabled": "42"},
            "self-managed",
            "uid-other",
            "",
        ),
        # Allowlist set for both realms — SaaS namespace not present
        (
            {"self_managed_enabled": "uid-1", "saas_enabled": "42"},
            "saas",
            "",
            "99999",
        ),
        # Single-item allowlist, instance not matching
        ({"self_managed_enabled": "only-uid"}, "self-managed", "wrong-uid", ""),
        # Whitespace trimming still blocks non-listed IDs
        ({"self_managed_enabled": " uid-a , uid-b "}, "self-managed", "uid-c", ""),
        # Unknown realm falls through to self-managed check and is blocked
        ({"self_managed_enabled": "uid-1"}, "unknown-realm", "uid-other", ""),
        # Empty realm falls through to self-managed check and is blocked
        ({"self_managed_enabled": "uid-1"}, "", "", ""),
    ],
)
async def test_check_proxy_endpoints_enabled_blocks(proxy_cfg, proxy_mock_request):
    """Requests that should be rejected with HTTP 402."""
    with pytest.raises(HTTPException) as exc_info:
        await _check_proxy_endpoints_enabled(
            proxy_mock_request,
            proxy_cfg=proxy_cfg,
        )

    assert exc_info.value.status_code == 402
