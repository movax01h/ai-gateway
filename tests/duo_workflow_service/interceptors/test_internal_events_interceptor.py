from unittest.mock import AsyncMock, MagicMock

import pytest

from duo_workflow_service.interceptors.internal_events_interceptor import (
    InternalEventsInterceptor,
)
from lib.internal_events import current_event_context


@pytest.fixture(name="mock_continuation")
def mock_continuation_fixture():
    return AsyncMock()


@pytest.fixture(name="interceptor")
def interceptor_fixture():
    return InternalEventsInterceptor()


def create_handler_call_details(metadata_dict):
    """Helper function to create handler call details with given metadata."""
    mock_details = MagicMock()
    mock_details.invocation_metadata = tuple(metadata_dict.items())
    return mock_details


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata,expected",
    [
        pytest.param(
            {
                "x-gitlab-realm": "test-realm",
                "x-gitlab-instance-id": "test-instance-id",
                "x-gitlab-global-user-id": "test-global-user-id",
                "x-gitlab-host-name": "test-gitlab-host",
                "x-gitlab-feature-enabled-by-namespace-ids": "1,2,3",
                "x-gitlab-feature-enablement-type": "duo_pro",
                "x-gitlab-project-id": "1",
                "x-gitlab-namespace-id": "2",
                "x-gitlab-root-namespace-id": "3",
                "x-gitlab-is-a-gitlab-member": "true",
            },
            {
                "realm": "test-realm",
                "instance_id": "test-instance-id",
                "global_user_id": "test-global-user-id",
                "host_name": "test-gitlab-host",
                "feature_enabled_by_namespace_ids": [1, 2, 3],
                "feature_enablement_type": "duo_pro",
                "project_id": 1,
                "namespace_id": 2,
                "ultimate_parent_namespace_id": 3,
                "is_gitlab_team_member": True,
            },
            id="standard_metadata",
        ),
        pytest.param(
            {
                "x-gitlab-realm": "test-realm",
                "x-gitlab-instance-id": "test-instance-id",
                "x-gitlab-global-user-id": "test-global-user-id",
                "x-gitlab-host-name": "test-gitlab-host",
                "x-gitlab-feature-enabled-by-namespace-ids": "1,2,3",
                "x-gitlab-feature-enablement-type": "duo_pro",
                "x-gitlab-project-id": "1",
                "x-gitlab-namespace-id": "2",
                "x-gitlab-root-namespace-id": "",
                "x-gitlab-is-a-gitlab-member": "true",
            },
            {
                "realm": "test-realm",
                "instance_id": "test-instance-id",
                "global_user_id": "test-global-user-id",
                "host_name": "test-gitlab-host",
                "feature_enabled_by_namespace_ids": [1, 2, 3],
                "feature_enablement_type": "duo_pro",
                "project_id": 1,
                "namespace_id": 2,
                "ultimate_parent_namespace_id": None,
                "is_gitlab_team_member": True,
            },
            id="missing_ultimate_parent_namespace_id",
        ),
        pytest.param(
            {
                "x-gitlab-realm": "test-realm",
                "x-gitlab-instance-id": "test-instance-id",
                "x-gitlab-global-user-id": "test-global-user-id",
                "x-gitlab-host-name": "test-gitlab-host",
                "x-gitlab-feature-enabled-by-namespace-ids": "1,2,3",
                "x-gitlab-feature-enablement-type": "duo_pro",
                "x-gitlab-project-id": "1",
                "x-gitlab-namespace-id": "2",
                "x-gitlab-root-namespace-id": None,
                "x-gitlab-is-a-gitlab-member": "true",
            },
            {
                "realm": "test-realm",
                "instance_id": "test-instance-id",
                "global_user_id": "test-global-user-id",
                "host_name": "test-gitlab-host",
                "feature_enabled_by_namespace_ids": [1, 2, 3],
                "feature_enablement_type": "duo_pro",
                "project_id": 1,
                "namespace_id": 2,
                "ultimate_parent_namespace_id": None,
                "is_gitlab_team_member": True,
            },
            id="none_ultimate_parent_namespace_id",
        ),
        pytest.param(
            {
                "x-gitlab-realm": "test-realm",
                "x-gitlab-instance-id": "test-instance-id",
                "x-gitlab-global-user-id": "test-global-user-id",
                "x-gitlab-host-name": "test-gitlab-host",
                "x-gitlab-feature-enabled-by-namespace-ids": "",
                "x-gitlab-feature-enablement-type": "duo_pro",
                "x-gitlab-project-id": "1",
                "x-gitlab-namespace-id": "2",
                "x-gitlab-root-namespace-id": "3",
                "x-gitlab-is-a-gitlab-member": "true",
            },
            {
                "realm": "test-realm",
                "instance_id": "test-instance-id",
                "global_user_id": "test-global-user-id",
                "host_name": "test-gitlab-host",
                "feature_enabled_by_namespace_ids": None,
                "feature_enablement_type": "duo_pro",
                "project_id": 1,
                "namespace_id": 2,
                "ultimate_parent_namespace_id": 3,
                "is_gitlab_team_member": True,
            },
            id="empty_feature_enabled",
        ),
        pytest.param(
            {
                "x-gitlab-realm": "test-realm",
                "x-gitlab-instance-id": "test-instance-id",
                "x-gitlab-global-user-id": "test-global-user-id",
                "x-gitlab-host-name": "test-gitlab-host",
                "x-gitlab-feature-enabled-by-namespace-ids": "1,2,2,3,1,4,3,5",
                "x-gitlab-feature-enablement-type": "duo_pro",
                "x-gitlab-project-id": "1",
                "x-gitlab-namespace-id": "2",
                "x-gitlab-root-namespace-id": "3",
                "x-gitlab-is-a-gitlab-member": "true",
            },
            {
                "realm": "test-realm",
                "instance_id": "test-instance-id",
                "global_user_id": "test-global-user-id",
                "host_name": "test-gitlab-host",
                "feature_enabled_by_namespace_ids": [1, 2, 3, 4, 5],
                "feature_enablement_type": "duo_pro",
                "project_id": 1,
                "namespace_id": 2,
                "ultimate_parent_namespace_id": 3,
                "is_gitlab_team_member": True,
            },
            id="duplicate_namespace_ids_removed",
        ),
        pytest.param(
            {
                "x-gitlab-realm": "test-realm",
                "x-gitlab-instance-id": "test-instance-id",
                "x-gitlab-global-user-id": "test-global-user-id",
                "x-gitlab-host-name": "test-gitlab-host",
                "x-gitlab-feature-enabled-by-namespace-ids": "",
                "x-gitlab-feature-enablement-type": "duo_pro",
                "x-gitlab-is-a-gitlab-member": "false",
                "x-gitlab-root-namespace-id": "3",
            },
            {
                "realm": "test-realm",
                "instance_id": "test-instance-id",
                "global_user_id": "test-global-user-id",
                "host_name": "test-gitlab-host",
                "feature_enabled_by_namespace_ids": None,
                "feature_enablement_type": "duo_pro",
                "project_id": None,
                "namespace_id": None,
                "ultimate_parent_namespace_id": 3,
                "is_gitlab_team_member": False,
            },
            id="empty_project_and_namespace_ids",
        ),
    ],
)
async def test_interceptor_metadata_handling(
    interceptor, mock_continuation, metadata, expected
):
    """Test that the interceptor correctly processes various metadata configurations."""
    handler_call_details = create_handler_call_details(metadata)

    await interceptor.intercept_service(mock_continuation, handler_call_details)

    event_context = current_event_context.get()

    assert event_context.realm == expected["realm"]
    assert event_context.instance_id == expected["instance_id"]
    assert event_context.global_user_id == expected["global_user_id"]
    assert event_context.host_name == expected["host_name"]
    assert (
        event_context.feature_enabled_by_namespace_ids
        == expected["feature_enabled_by_namespace_ids"]
    )
    assert event_context.feature_enablement_type == expected["feature_enablement_type"]
    assert event_context.project_id == expected["project_id"]
    assert event_context.namespace_id == expected["namespace_id"]
    assert (
        event_context.ultimate_parent_namespace_id
        == expected["ultimate_parent_namespace_id"]
    )
    assert event_context.is_gitlab_team_member == expected["is_gitlab_team_member"]
