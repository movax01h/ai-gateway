from unittest.mock import AsyncMock, MagicMock

import pytest

from duo_workflow_service.interceptors.internal_events_interceptor import (
    InternalEventsInterceptor,
)
from duo_workflow_service.internal_events import current_event_context


@pytest.fixture
def mock_continuation():
    return AsyncMock()


@pytest.fixture
def interceptor():
    return InternalEventsInterceptor()


@pytest.fixture
def handler_call_details():
    mock_details = MagicMock()
    mock_details.invocation_metadata = (
        ("x-gitlab-realm", "test-realm"),
        ("x-gitlab-instance-id", "test-instance-id"),
        ("x-gitlab-global-user-id", "test-global-user-id"),
        ("x-gitlab-host-name", "test-gitlab-host"),
        ("x-gitlab-feature-enabled-by-namespace-ids", "1,2,3"),
        ("x-gitlab-project-id", "1"),
        ("x-gitlab-namespace-id", "2"),
        ("x-gitlab-is-a-gitlab-member", "true"),
    )
    return mock_details


@pytest.fixture
def handler_call_details_with_empty_feature():
    mock_details = MagicMock()
    mock_details.invocation_metadata = (
        ("x-gitlab-realm", "test-realm"),
        ("x-gitlab-instance-id", "test-instance-id"),
        ("x-gitlab-global-user-id", "test-global-user-id"),
        ("x-gitlab-host-name", "test-gitlab-host"),
        ("x-gitlab-feature-enabled-by-namespace-ids", ""),
        ("x-gitlab-project-id", "1"),
        ("x-gitlab-namespace-id", "2"),
        ("x-gitlab-is-a-gitlab-member", "true"),
    )
    return mock_details


@pytest.fixture
def handler_call_details_with_empty_project_and_namespace_id():
    mock_details = MagicMock()
    mock_details.invocation_metadata = (
        ("x-gitlab-realm", "test-realm"),
        ("x-gitlab-instance-id", "test-instance-id"),
        ("x-gitlab-global-user-id", "test-global-user-id"),
        ("x-gitlab-host-name", "test-gitlab-host"),
        ("x-gitlab-feature-enabled-by-namespace-ids", ""),
        ("x-gitlab-is-a-gitlab-member", "false"),
    )
    return mock_details


@pytest.mark.asyncio
async def test_interceptor_with_internal_events_disabled(
    interceptor, mock_continuation, handler_call_details
):
    await interceptor.intercept_service(mock_continuation, handler_call_details)
    event_context = current_event_context.get()
    assert event_context.realm == "test-realm"
    assert event_context.instance_id == "test-instance-id"
    assert event_context.global_user_id == "test-global-user-id"
    assert event_context.host_name == "test-gitlab-host"
    assert event_context.feature_enabled_by_namespace_ids == [1, 2, 3]
    assert event_context.project_id == 1
    assert event_context.namespace_id == 2
    assert event_context.is_gitlab_team_member is True


@pytest.mark.asyncio
async def test_interceptor_with_empty_feature_enabled_attribute(
    interceptor, mock_continuation, handler_call_details_with_empty_feature
):
    await interceptor.intercept_service(
        mock_continuation, handler_call_details_with_empty_feature
    )
    event_context = current_event_context.get()
    assert event_context.realm == "test-realm"
    assert event_context.instance_id == "test-instance-id"
    assert event_context.global_user_id == "test-global-user-id"
    assert event_context.host_name == "test-gitlab-host"
    assert event_context.feature_enabled_by_namespace_ids is None
    assert event_context.project_id == 1
    assert event_context.namespace_id == 2
    assert event_context.is_gitlab_team_member is True


@pytest.mark.asyncio
async def test_interceptor_with_empty_project_and_namespace_ids(
    interceptor,
    mock_continuation,
    handler_call_details_with_empty_project_and_namespace_id,
):
    await interceptor.intercept_service(
        mock_continuation, handler_call_details_with_empty_project_and_namespace_id
    )
    event_context = current_event_context.get()
    assert event_context.realm == "test-realm"
    assert event_context.instance_id == "test-instance-id"
    assert event_context.global_user_id == "test-global-user-id"
    assert event_context.host_name == "test-gitlab-host"
    assert event_context.feature_enabled_by_namespace_ids is None
    assert event_context.project_id is None
    assert event_context.namespace_id is None


@pytest.mark.asyncio
async def test_interceptor_with_gitlab_member_false(
    interceptor,
    mock_continuation,
    handler_call_details_with_empty_project_and_namespace_id,
):
    await interceptor.intercept_service(
        mock_continuation, handler_call_details_with_empty_project_and_namespace_id
    )
    event_context = current_event_context.get()
    assert event_context.is_gitlab_team_member is False
