import pytest
from fastapi import HTTPException, Response
from gitlab_cloud_connector import CloudConnectorUser, UserClaims

from ai_gateway.api.v1.proxy.request import verify_project_namespace_metadata
from lib.context import StarletteUser
from lib.internal_events.context import EventContext, current_event_context


@pytest.mark.asyncio
async def test_verify_project_namespace_metadata_saas_project_mismatch(mock_request):
    """Test SaaS verification fails when project ID doesn't match."""
    user = StarletteUser(
        CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(
                gitlab_realm="saas",
                extra={
                    "gitlab_project_id": "999",
                    "gitlab_namespace_id": "456",
                    "gitlab_root_namespace_id": "789",
                },
            ),
        )
    )
    mock_request.user = user

    event_context = EventContext(
        project_id=123,
        namespace_id=456,
        ultimate_parent_namespace_id=789,
    )
    current_event_context.set(event_context)

    @verify_project_namespace_metadata()
    async def dummy_func(request, *args, **kwargs):  # pylint: disable=unused-argument
        return Response(content=b'{"message": "success"}', status_code=200)

    with pytest.raises(HTTPException) as exc_info:
        await dummy_func(mock_request)

    assert exc_info.value.status_code == 403
    assert "project id mismatch" in exc_info.value.detail


@pytest.mark.asyncio
async def test_verify_project_namespace_metadata_saas_namespace_mismatch(mock_request):
    """Test SaaS verification fails when namespace ID doesn't match."""
    user = StarletteUser(
        CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(
                gitlab_realm="saas",
                extra={
                    "gitlab_project_id": "123",
                    "gitlab_namespace_id": "999",
                    "gitlab_root_namespace_id": "789",
                },
            ),
        )
    )
    mock_request.user = user

    event_context = EventContext(
        project_id=123,
        namespace_id=456,
        ultimate_parent_namespace_id=789,
    )
    current_event_context.set(event_context)

    @verify_project_namespace_metadata()
    async def dummy_func(request, *args, **kwargs):  # pylint: disable=unused-argument
        return Response(content=b'{"message": "success"}', status_code=200)

    with pytest.raises(HTTPException) as exc_info:
        await dummy_func(mock_request)

    assert exc_info.value.status_code == 403
    assert "namespace id mismatch" in exc_info.value.detail


@pytest.mark.asyncio
async def test_verify_project_namespace_metadata_saas_root_namespace_mismatch(
    mock_request,
):
    """Test SaaS verification fails when root namespace ID doesn't match."""
    user = StarletteUser(
        CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(
                gitlab_realm="saas",
                extra={
                    "gitlab_project_id": "123",
                    "gitlab_namespace_id": "456",
                    "gitlab_root_namespace_id": "999",
                },
            ),
        )
    )
    mock_request.user = user

    event_context = EventContext(
        project_id=123,
        namespace_id=456,
        ultimate_parent_namespace_id=789,
    )
    current_event_context.set(event_context)

    @verify_project_namespace_metadata()
    async def dummy_func(request, *args, **kwargs):  # pylint: disable=unused-argument
        return Response(content=b'{"message": "success"}', status_code=200)

    with pytest.raises(HTTPException) as exc_info:
        await dummy_func(mock_request)

    assert exc_info.value.status_code == 403
    assert "root namespace id mismatch" in exc_info.value.detail


@pytest.mark.asyncio
async def test_verify_project_namespace_metadata_self_managed_success(mock_request):
    """Test successful verification for self-managed with matching instance UID."""
    user = StarletteUser(
        CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(
                gitlab_realm="self-managed",
                gitlab_instance_uid="instance-uid-123",
                extra={"gitlab_instance_uid": "instance-uid-123"},
            ),
        )
    )
    mock_request.user = user

    event_context = EventContext(instance_id="instance-uid-123")
    current_event_context.set(event_context)

    @verify_project_namespace_metadata()
    async def dummy_func(request, *args, **kwargs):  # pylint: disable=unused-argument
        return Response(content=b'{"message": "success"}', status_code=200)

    response = await dummy_func(mock_request)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_verify_project_namespace_metadata_self_managed_instance_mismatch(
    mock_request,
):
    """Test self-managed verification fails when instance UID doesn't match."""
    user = StarletteUser(
        CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(
                gitlab_realm="self-managed",
                gitlab_instance_uid="wrong-instance-uid",
                extra={"gitlab_instance_uid": "wrong-instance-uid"},
            ),
        )
    )
    mock_request.user = user

    event_context = EventContext(instance_id="instance-uid-123")
    current_event_context.set(event_context)

    @verify_project_namespace_metadata()
    async def dummy_func(request, *args, **kwargs):  # pylint: disable=unused-argument
        return Response(content=b'{"message": "success"}', status_code=200)

    with pytest.raises(HTTPException) as exc_info:
        await dummy_func(mock_request)

    assert exc_info.value.status_code == 403
    assert "instance uid mismatch" in exc_info.value.detail


@pytest.mark.asyncio
async def test_verify_project_namespace_metadata_self_managed_ignores_project_ids(
    mock_request,
):
    """Test self-managed verification ignores project/namespace IDs in extra claims."""
    user = StarletteUser(
        CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(
                gitlab_realm="self-managed",
                gitlab_instance_uid="instance-uid-123",
                extra={
                    "gitlab_instance_uid": "instance-uid-123",
                    "gitlab_project_id": "999",  # Should be ignored
                    "gitlab_namespace_id": "999",  # Should be ignored
                    "gitlab_root_namespace_id": "999",  # Should be ignored
                },
            ),
        )
    )
    mock_request.user = user

    event_context = EventContext(
        instance_id="instance-uid-123",
        project_id=123,  # Different from extra claims
        namespace_id=456,  # Different from extra claims
        ultimate_parent_namespace_id=789,  # Different from extra claims
    )
    current_event_context.set(event_context)

    @verify_project_namespace_metadata()
    async def dummy_func(request, *args, **kwargs):  # pylint: disable=unused-argument
        return Response(content=b'{"message": "success"}', status_code=200)

    # Should succeed because self-managed only checks instance_uid
    response = await dummy_func(mock_request)
    assert response.status_code == 200
