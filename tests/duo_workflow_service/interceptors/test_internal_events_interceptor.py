from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims

from duo_workflow_service.interceptors.authentication_interceptor import current_user
from duo_workflow_service.interceptors.internal_events_interceptor import (
    InternalEventsInterceptor,
)
from lib.context import gitlab_version, language_server_version
from lib.internal_events import current_event_context
from lib.language_server import LanguageServerVersion


@pytest.fixture(name="mock_continuation")
def mock_continuation_fixture():
    return AsyncMock()


@pytest.fixture(name="interceptor")
def interceptor_fixture():
    return InternalEventsInterceptor()


@pytest.fixture(name="mock_user")
def mock_user():
    return CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(
            scopes=[],
            subject="1234",
            gitlab_realm="self-managed",
            gitlab_instance_uid="00000000-1111-2222-3333-000000000000",
        ),
    )


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
                "unique_instance_id": "00000000-1111-2222-3333-000000000000",
                "global_user_id": "test-global-user-id",
                "host_name": "test-gitlab-host",
                "feature_enabled_by_namespace_ids": [1, 2, 3],
                "feature_enablement_type": "duo_pro",
                "project_id": 1,
                "namespace_id": 2,
                "ultimate_parent_namespace_id": 3,
                "is_gitlab_team_member": True,
                "client_name": None,
                "client_type": None,
                "client_version": None,
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
                "x-gitlab-root-namespace-id": "3",
                "x-gitlab-is-team-member": "true",
            },
            {
                "realm": "test-realm",
                "instance_id": "test-instance-id",
                "unique_instance_id": "00000000-1111-2222-3333-000000000000",
                "global_user_id": "test-global-user-id",
                "host_name": "test-gitlab-host",
                "feature_enabled_by_namespace_ids": [1, 2, 3],
                "feature_enablement_type": "duo_pro",
                "project_id": 1,
                "namespace_id": 2,
                "ultimate_parent_namespace_id": 3,
                "is_gitlab_team_member": True,
                "client_name": None,
                "client_type": None,
                "client_version": None,
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
                "unique_instance_id": "00000000-1111-2222-3333-000000000000",
                "global_user_id": "test-global-user-id",
                "host_name": "test-gitlab-host",
                "feature_enabled_by_namespace_ids": [1, 2, 3],
                "feature_enablement_type": "duo_pro",
                "project_id": 1,
                "namespace_id": 2,
                "ultimate_parent_namespace_id": None,
                "is_gitlab_team_member": True,
                "client_name": None,
                "client_type": None,
                "client_version": None,
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
                "unique_instance_id": "00000000-1111-2222-3333-000000000000",
                "global_user_id": "test-global-user-id",
                "host_name": "test-gitlab-host",
                "feature_enabled_by_namespace_ids": [1, 2, 3],
                "feature_enablement_type": "duo_pro",
                "project_id": 1,
                "namespace_id": 2,
                "ultimate_parent_namespace_id": None,
                "is_gitlab_team_member": True,
                "client_name": None,
                "client_type": None,
                "client_version": None,
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
                "unique_instance_id": "00000000-1111-2222-3333-000000000000",
                "global_user_id": "test-global-user-id",
                "host_name": "test-gitlab-host",
                "feature_enabled_by_namespace_ids": None,
                "feature_enablement_type": "duo_pro",
                "project_id": 1,
                "namespace_id": 2,
                "ultimate_parent_namespace_id": 3,
                "is_gitlab_team_member": True,
                "client_name": None,
                "client_type": None,
                "client_version": None,
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
                "unique_instance_id": "00000000-1111-2222-3333-000000000000",
                "global_user_id": "test-global-user-id",
                "host_name": "test-gitlab-host",
                "feature_enabled_by_namespace_ids": [1, 2, 3, 4, 5],
                "feature_enablement_type": "duo_pro",
                "project_id": 1,
                "namespace_id": 2,
                "ultimate_parent_namespace_id": 3,
                "is_gitlab_team_member": True,
                "client_name": None,
                "client_type": None,
                "client_version": None,
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
                "unique_instance_id": "00000000-1111-2222-3333-000000000000",
                "global_user_id": "test-global-user-id",
                "host_name": "test-gitlab-host",
                "feature_enabled_by_namespace_ids": None,
                "feature_enablement_type": "duo_pro",
                "project_id": None,
                "namespace_id": None,
                "ultimate_parent_namespace_id": 3,
                "is_gitlab_team_member": False,
                "client_name": None,
                "client_type": None,
                "client_version": None,
            },
            id="empty_project_and_namespace_ids",
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
                "x-gitlab-root-namespace-id": "3",
                "x-gitlab-is-a-gitlab-member": "true",
                "x-gitlab-client-name": "gitlab-duo-workflow",
                "x-gitlab-client-type": "node-websocket",
                "x-gitlab-client-version": "1.0.0",
            },
            {
                "realm": "test-realm",
                "instance_id": "test-instance-id",
                "unique_instance_id": "00000000-1111-2222-3333-000000000000",
                "global_user_id": "test-global-user-id",
                "host_name": "test-gitlab-host",
                "feature_enabled_by_namespace_ids": [1, 2, 3],
                "feature_enablement_type": "duo_pro",
                "project_id": 1,
                "namespace_id": 2,
                "ultimate_parent_namespace_id": 3,
                "is_gitlab_team_member": True,
                "client_name": "gitlab-duo-workflow",
                "client_type": "node-websocket",
                "client_version": "1.0.0",
            },
            id="with_client_headers",
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
                "x-gitlab-root-namespace-id": "3",
                "x-gitlab-is-a-gitlab-member": "true",
            },
            {
                "realm": "test-realm",
                "instance_id": "test-instance-id",
                "unique_instance_id": "00000000-1111-2222-3333-000000000000",
                "global_user_id": "test-global-user-id",
                "host_name": "test-gitlab-host",
                "feature_enabled_by_namespace_ids": [1, 2, 3],
                "feature_enablement_type": "duo_pro",
                "project_id": 1,
                "namespace_id": 2,
                "ultimate_parent_namespace_id": 3,
                "is_gitlab_team_member": True,
                "client_name": None,
                "client_type": None,
                "client_version": None,
            },
            id="missing_client_headers",
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
                "x-gitlab-root-namespace-id": "3",
                "x-gitlab-is-a-gitlab-member": "true",
                "x-gitlab-client-name": "",
                "x-gitlab-client-type": None,
                "x-gitlab-client-version": "",
            },
            {
                "realm": "test-realm",
                "instance_id": "test-instance-id",
                "unique_instance_id": "00000000-1111-2222-3333-000000000000",
                "global_user_id": "test-global-user-id",
                "host_name": "test-gitlab-host",
                "feature_enabled_by_namespace_ids": [1, 2, 3],
                "feature_enablement_type": "duo_pro",
                "project_id": 1,
                "namespace_id": 2,
                "ultimate_parent_namespace_id": 3,
                "is_gitlab_team_member": True,
                "client_name": None,
                "client_type": None,
                "client_version": None,
            },
            id="empty_and_none_client_headers",
        ),
    ],
)
async def test_interceptor_metadata_handling(
    interceptor, mock_continuation, mock_user, metadata, expected
):
    """Test that the interceptor correctly processes various metadata configurations."""
    handler_call_details = create_handler_call_details(metadata)
    current_user.set(mock_user)

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
    assert event_context.unique_instance_id == expected["unique_instance_id"]
    assert event_context.client_name == expected["client_name"]
    assert event_context.client_type == expected["client_type"]
    assert event_context.client_version == expected["client_version"]
    # By default, no lsp_version should be in extra
    assert event_context.extra == {}
    # By default, instance_version should be None
    assert event_context.instance_version is None


@pytest.mark.asyncio
async def test_interceptor_with_lsp_version(interceptor, mock_continuation, mock_user):
    """Test that lsp_version is included in extra when language_server_version is set."""
    metadata = {
        "x-gitlab-realm": "test-realm",
        "x-gitlab-instance-id": "test-instance-id",
        "x-gitlab-global-user-id": "test-global-user-id",
        "x-gitlab-host-name": "test-gitlab-host",
    }
    handler_call_details = create_handler_call_details(metadata)
    current_user.set(mock_user)

    # Set the language server version in the context
    test_version = LanguageServerVersion.from_string("7.43.0")
    language_server_version.set(test_version)

    await interceptor.intercept_service(mock_continuation, handler_call_details)

    event_context = current_event_context.get()

    # Verify lsp_version is in extra
    assert "lsp_version" in event_context.extra
    assert event_context.extra["lsp_version"] == "7.43.0"


@pytest.mark.asyncio
async def test_interceptor_without_lsp_version(
    interceptor, mock_continuation, mock_user
):
    """Test that extra is empty when language_server_version is not set."""
    metadata = {
        "x-gitlab-realm": "test-realm",
        "x-gitlab-instance-id": "test-instance-id",
        "x-gitlab-global-user-id": "test-global-user-id",
        "x-gitlab-host-name": "test-gitlab-host",
    }
    handler_call_details = create_handler_call_details(metadata)
    current_user.set(mock_user)

    # Reset the language server version context
    language_server_version.set(None)

    await interceptor.intercept_service(mock_continuation, handler_call_details)

    event_context = current_event_context.get()

    # Verify extra is empty
    assert event_context.extra == {}


@pytest.mark.asyncio
async def test_interceptor_with_invalid_lsp_version_object(
    interceptor, mock_continuation, mock_user
):
    """Test that extra is empty when language_server_version has invalid object without version attribute."""
    metadata = {
        "x-gitlab-realm": "test-realm",
        "x-gitlab-instance-id": "test-instance-id",
        "x-gitlab-global-user-id": "test-global-user-id",
        "x-gitlab-host-name": "test-gitlab-host",
    }
    handler_call_details = create_handler_call_details(metadata)
    current_user.set(mock_user)

    # Set an invalid object without version attribute
    invalid_object = MagicMock(spec=[])  # Object with no attributes
    language_server_version.set(invalid_object)

    await interceptor.intercept_service(mock_continuation, handler_call_details)

    event_context = current_event_context.get()

    # Verify extra is empty (no crash, gracefully handled)
    assert event_context.extra == {}


@pytest.mark.asyncio
async def test_interceptor_with_instance_version(
    interceptor, mock_continuation, mock_user
):
    """Test that instance_version is included when gitlab_version is set."""
    metadata = {
        "x-gitlab-realm": "test-realm",
        "x-gitlab-instance-id": "test-instance-id",
        "x-gitlab-global-user-id": "test-global-user-id",
        "x-gitlab-host-name": "test-gitlab-host",
    }
    handler_call_details = create_handler_call_details(metadata)
    current_user.set(mock_user)

    # Set the GitLab instance version in the context
    gitlab_version.set("16.11.0")

    await interceptor.intercept_service(mock_continuation, handler_call_details)

    event_context = current_event_context.get()

    # Verify instance_version is set
    assert event_context.instance_version == "16.11.0"


@pytest.mark.asyncio
async def test_interceptor_without_instance_version(
    interceptor, mock_continuation, mock_user
):
    """Test that instance_version is None when gitlab_version is not set."""
    metadata = {
        "x-gitlab-realm": "test-realm",
        "x-gitlab-instance-id": "test-instance-id",
        "x-gitlab-global-user-id": "test-global-user-id",
        "x-gitlab-host-name": "test-gitlab-host",
    }
    handler_call_details = create_handler_call_details(metadata)
    current_user.set(mock_user)

    # Reset the GitLab version context
    gitlab_version.set(None)

    await interceptor.intercept_service(mock_continuation, handler_call_details)

    event_context = current_event_context.get()

    # Verify instance_version is None
    assert event_context.instance_version is None


@pytest.mark.asyncio
async def test_interceptor_with_both_versions(
    interceptor, mock_continuation, mock_user
):
    """Test that both instance_version and lsp_version are included when both are set."""
    metadata = {
        "x-gitlab-realm": "test-realm",
        "x-gitlab-instance-id": "test-instance-id",
        "x-gitlab-global-user-id": "test-global-user-id",
        "x-gitlab-host-name": "test-gitlab-host",
    }
    handler_call_details = create_handler_call_details(metadata)
    current_user.set(mock_user)

    # Set both versions
    gitlab_version.set("16.11.0")
    test_lsp_version = LanguageServerVersion.from_string("7.43.0")
    language_server_version.set(test_lsp_version)

    await interceptor.intercept_service(mock_continuation, handler_call_details)

    event_context = current_event_context.get()

    # Verify both versions are set
    assert event_context.instance_version == "16.11.0"
    assert "lsp_version" in event_context.extra
    assert event_context.extra["lsp_version"] == "7.43.0"


@pytest.mark.asyncio
async def test_interceptor_logs_per_field_warning_when_required_fields_missing(
    interceptor, mock_continuation, mock_user
):
    """Test that one warning is logged per missing required field."""
    metadata = {
        "x-gitlab-realm": "test-realm",
        "x-gitlab-instance-id": "test-instance-id",
        # missing global_user_id, host_name, deployment_type
    }
    handler_call_details = create_handler_call_details(metadata)
    handler_call_details.method = "/duo_workflow.v1.DuoWorkflow/Execute"
    current_user.set(mock_user)

    with patch("lib.internal_events.context_validator.log") as mock_log:
        await interceptor.intercept_service(mock_continuation, handler_call_details)

        warning_fields = {
            c[1]["missing_field"] for c in mock_log.warning.call_args_list
        }
        assert {
            "global_user_id",
            "host_name",
            "deployment_type",
        }.issubset(warning_fields)

        for c in mock_log.warning.call_args_list:
            assert c[0][0] == "Internal event context missing required field"
            assert c[1]["field_type"] == "required"
            assert c[1]["grpc_method"] == "/duo_workflow.v1.DuoWorkflow/Execute"
            assert "correlation_id" in c[1]


@pytest.mark.asyncio
async def test_interceptor_logs_per_field_info_when_contextual_fields_missing(
    interceptor, mock_continuation, mock_user
):
    """Test that one info log is emitted per missing contextual field."""
    metadata = {
        "x-gitlab-realm": "test-realm",
        "x-gitlab-instance-id": "test-instance-id",
        "x-gitlab-global-user-id": "test-global-user-id",
        "x-gitlab-host-name": "test-host",
        "x-gitlab-deployment-type": ".com",
        "x-gitlab-project-id": "42",
        "x-gitlab-namespace-id": "7",
        # missing contextual: feature_enabled_by_namespace_ids, is_gitlab_team_member, etc.
    }
    handler_call_details = create_handler_call_details(metadata)
    handler_call_details.method = "/duo_workflow.v1.DuoWorkflow/Execute"
    current_user.set(mock_user)

    with patch("lib.internal_events.context_validator.log") as mock_log:
        await interceptor.intercept_service(mock_continuation, handler_call_details)

        mock_log.warning.assert_not_called()

        info_fields = {c[1]["missing_field"] for c in mock_log.info.call_args_list}
        assert "is_gitlab_team_member" in info_fields
        assert "ultimate_parent_namespace_id" in info_fields

        for c in mock_log.info.call_args_list:
            assert c[0][0] == "Internal event context missing contextual field"
            assert c[1]["field_type"] == "contextual"


@pytest.mark.asyncio
async def test_interceptor_no_logs_when_all_fields_present(
    interceptor, mock_continuation, mock_user
):
    """Test that no logs are emitted when all fields are present."""
    metadata = {
        "x-gitlab-realm": "test-realm",
        "x-gitlab-instance-id": "test-instance-id",
        "x-gitlab-global-user-id": "test-global-user-id",
        "x-gitlab-host-name": "test-host",
        "x-gitlab-deployment-type": ".com",
        "x-gitlab-feature-enabled-by-namespace-ids": "1,2,3",
        "x-gitlab-is-a-gitlab-member": "true",
        "x-gitlab-root-namespace-id": "99",
        "x-gitlab-project-id": "42",
        "x-gitlab-namespace-id": "7",
    }
    handler_call_details = create_handler_call_details(metadata)
    current_user.set(mock_user)

    with patch("lib.internal_events.context_validator.log") as mock_log:
        await interceptor.intercept_service(mock_continuation, handler_call_details)

        mock_log.warning.assert_not_called()
        mock_log.info.assert_not_called()
