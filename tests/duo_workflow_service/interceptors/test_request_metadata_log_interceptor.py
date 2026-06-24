from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog

from duo_workflow_service.interceptors.request_metadata_log_interceptor import (
    RequestMetadataLogInterceptor,
)


@pytest.fixture(name="mock_continuation")
def mock_continuation_fixture():
    return AsyncMock()


@pytest.fixture(name="interceptor")
def interceptor_fixture():
    return RequestMetadataLogInterceptor()


@pytest.mark.asyncio
async def test_gitlab_metadata_bound_to_structlog_contextvars(
    interceptor, mock_continuation
):
    """Verify that GitLab request metadata is bound to structlog context vars."""
    mock_details = MagicMock()
    mock_details.invocation_metadata = (
        ("x-gitlab-instance-id", "instance-abc"),
        ("x-gitlab-host-name", "my-gitlab.example.com"),
        ("x-gitlab-realm", "saas"),
        ("x-gitlab-version", "17.0.0"),
        ("x-gitlab-language-server-version", "1.2.3"),
        ("x-gitlab-root-namespace-id", "99"),
        ("x-gitlab-feature-enabled-by-namespace-ids", "1,2,3"),
        ("x-gitlab-feature-enablement-type", "duo_pro"),
        ("x-gitlab-organization-id", "42"),
        ("x-gitlab-is-team-member", "true"),
    )

    structlog.contextvars.clear_contextvars()
    await interceptor.intercept_service(mock_continuation, mock_details)

    bound_vars = structlog.contextvars.get_contextvars()
    assert bound_vars["gitlab_instance_id"] == "instance-abc"
    assert bound_vars["gitlab_host_name"] == "my-gitlab.example.com"
    assert bound_vars["gitlab_realm"] == "saas"
    assert bound_vars["gitlab_version"] == "17.0.0"
    assert bound_vars["gitlab_language_server_version"] == "1.2.3"
    assert bound_vars["gitlab_root_namespace_id"] == "99"
    assert bound_vars["gitlab_feature_enabled_by_namespace_ids"] == "1,2,3"
    assert bound_vars["gitlab_feature_enablement_type"] == "duo_pro"
    assert bound_vars["gitlab_organization_id"] == "42"
    assert bound_vars["is_gitlab_team_member"] == "true"


@pytest.mark.asyncio
async def test_missing_metadata_bound_as_none(interceptor, mock_continuation):
    """Verify that missing metadata headers are bound as None in structlog context vars."""
    mock_details = MagicMock()
    mock_details.invocation_metadata = ()

    structlog.contextvars.clear_contextvars()
    await interceptor.intercept_service(mock_continuation, mock_details)

    bound_vars = structlog.contextvars.get_contextvars()
    assert bound_vars["gitlab_instance_id"] is None
    assert bound_vars["gitlab_host_name"] is None
    assert bound_vars["gitlab_realm"] is None
    assert bound_vars["gitlab_version"] is None
    assert bound_vars["gitlab_language_server_version"] is None
    assert bound_vars["gitlab_root_namespace_id"] is None
    assert bound_vars["gitlab_feature_enabled_by_namespace_ids"] is None
    assert bound_vars["gitlab_feature_enablement_type"] is None
    assert bound_vars["gitlab_organization_id"] is None
    assert bound_vars["is_gitlab_team_member"] is None


@pytest.mark.asyncio
async def test_metadata_propagated_to_all_log_entries(interceptor, mock_continuation):
    """Verify that bound metadata appears in all log entries emitted during a request."""
    mock_details = MagicMock()
    mock_details.invocation_metadata = (
        ("x-gitlab-instance-id", "inst-001"),
        ("x-gitlab-host-name", "gitlab.example.com"),
        ("x-gitlab-realm", "self-managed"),
    )

    structlog.contextvars.clear_contextvars()

    log_entries = []

    async def capturing_continuation(_details):
        """Simulate a handler that emits log entries during request processing."""
        # Capture context vars directly since capture_logs doesn't use merge_contextvars
        log_entries.append(structlog.contextvars.get_contextvars())
        return "response"

    mock_continuation.side_effect = capturing_continuation

    await interceptor.intercept_service(mock_continuation, mock_details)

    assert len(log_entries) == 1
    ctx = log_entries[0]
    assert ctx["gitlab_instance_id"] == "inst-001"
    assert ctx["gitlab_host_name"] == "gitlab.example.com"
    assert ctx["gitlab_realm"] == "self-managed"


@pytest.mark.asyncio
async def test_is_gitlab_team_member_fallback(interceptor, mock_continuation):
    """Verify that x-gitlab-is-a-gitlab-member is used when x-gitlab-is-team-member is absent."""
    mock_details = MagicMock()
    mock_details.invocation_metadata = (("x-gitlab-is-a-gitlab-member", "true"),)

    structlog.contextvars.clear_contextvars()
    await interceptor.intercept_service(mock_continuation, mock_details)

    bound_vars = structlog.contextvars.get_contextvars()
    assert bound_vars["is_gitlab_team_member"] == "true"
