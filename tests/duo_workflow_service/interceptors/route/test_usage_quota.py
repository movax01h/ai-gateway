from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims
from grpc.aio import ServicerContext

from contract import contract_pb2
from duo_workflow_service.interceptors.authentication_interceptor import current_user
from duo_workflow_service.server import DuoWorkflowService
from lib.context import gitlab_version
from lib.internal_events.context import EventContext, current_event_context
from lib.usage_quota import UsageQuotaEvent
from lib.usage_quota.client import SKIP_USAGE_CUTOFF_CLAIM


@pytest.fixture(name="mock_user_with_skip_usage_cutoff")
def mock_user_with_skip_usage_cutoff_fixture():
    return CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(extra={SKIP_USAGE_CUTOFF_CLAIM: True}),
    )


@pytest.fixture(autouse=True)
def mock_usage_quota_service(mock_duo_workflow_service_container):
    """Auto-use fixture to properly wire DI container and mock UsageQuotaService.

    This ensures the @has_sufficient_usage_quota decorator works correctly.
    """
    from duo_workflow_service import server as server_module

    service_instance = MagicMock()
    service_instance.execute = AsyncMock()
    service_instance.aclose = AsyncMock()

    mock_duo_workflow_service_container.usage_quota.service.override(service_instance)
    mock_duo_workflow_service_container.wire(modules=[server_module])

    yield service_instance

    mock_duo_workflow_service_container.usage_quota.service.reset_override()


@pytest.fixture(name="mock_context")
def mock_context_fixture():
    """Mock gRPC context."""
    mock_context = MagicMock(spec=ServicerContext)
    mock_context.abort = AsyncMock()
    return mock_context


@pytest.fixture(name="duo_service")
def duo_service_fixture():
    """Create a DuoWorkflowService instance for testing."""
    return DuoWorkflowService()


async def mock_request_generator() -> AsyncIterator[contract_pb2.ClientEvent]:
    """Helper to generate mock request stream."""
    yield contract_pb2.ClientEvent(
        startRequest=contract_pb2.StartWorkflowRequest(
            workflowDefinition="test_workflow"
        )
    )


async def mock_track_self_hosted_request_generator() -> (
    AsyncIterator[contract_pb2.TrackSelfHostedClientEvent]
):
    """Helper to generate mock TrackSelfHostedClientEvent stream."""
    yield contract_pb2.TrackSelfHostedClientEvent(
        requestID="test-request-id",
        workflowID="test-workflow-id",
        featureQualifiedName="test_feature",
        featureAiCatalogItem=True,
    )


@pytest.mark.asyncio
async def test_execute_workflow_returns_early_when_stream_closed_before_first_message(
    mock_usage_quota_service,
    duo_service,
    mock_context,
) -> None:
    """Test ExecuteWorkflow decorator handles stream closed before first message gracefully."""

    async def empty_request_generator() -> AsyncIterator[contract_pb2.ClientEvent]:
        return
        yield  # make it an async generator

    from duo_workflow_service.interceptors.route.usage_quota import (
        has_sufficient_usage_quota,
    )

    inner = MagicMock()
    inner.__name__ = "ExecuteWorkflow"
    inner.__qualname__ = "DuoWorkflowServicer.ExecuteWorkflow"

    decorated = has_sufficient_usage_quota(event=UsageQuotaEvent.DAP_FLOW_ON_EXECUTE)(
        inner
    )

    results = [
        item
        async for item in decorated(
            duo_service,
            empty_request_generator(),
            mock_context,
            service=mock_usage_quota_service,
        )
    ]

    assert results == []
    mock_usage_quota_service.execute.assert_not_called()
    inner.assert_not_called()
    mock_context.set_code.assert_called_once_with(grpc.StatusCode.OK)
    mock_context.set_details.assert_called_once_with("workflow execution never started")


@pytest.mark.asyncio
async def test_execute_workflow_for_user_with_skip_usage_cutoff_extra_claim(
    mock_usage_quota_service,
    mock_user_with_skip_usage_cutoff,
    duo_service,
    mock_context,
) -> None:
    """Test ExecuteWorkflow decorator skips quota check for users with skip claim."""
    current_user.set(mock_user_with_skip_usage_cutoff)

    async def mock_impl():
        async for item in mock_request_generator():
            yield item

    assert not mock_usage_quota_service.execute.called


@pytest.mark.asyncio
async def test_execute_workflow_calls_usage_quota_service(
    mock_usage_quota_service, duo_service, mock_context
) -> None:
    """Test ExecuteWorkflow decorator calls usage quota service."""
    regular_user = CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(extra={}),
    )
    current_user.set(regular_user)

    async def mock_execute_request():
        yield contract_pb2.ClientEvent(
            startRequest=contract_pb2.StartWorkflowRequest(
                workflowDefinition="test_workflow", goal="test goal"
            )
        )

    assert mock_usage_quota_service.execute is not None
    assert isinstance(mock_usage_quota_service.execute, AsyncMock)


@pytest.mark.asyncio
async def test_generate_token_calls_usage_quota_service(
    mock_usage_quota_service, duo_service, mock_context
) -> None:
    """Test GenerateToken decorator calls usage quota service for regular users."""
    regular_user = CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(extra={}),
    )
    current_user.set(regular_user)

    assert mock_usage_quota_service.execute is not None
    assert isinstance(mock_usage_quota_service.execute, AsyncMock)


@pytest.mark.asyncio
async def test_generate_token_skips_quota_for_skip_users(
    mock_usage_quota_service, mock_user_with_skip_usage_cutoff, duo_service
) -> None:
    """Test GenerateToken decorator skips quota check for users with skip claim."""
    current_user.set(mock_user_with_skip_usage_cutoff)

    assert not mock_usage_quota_service.execute.called


@pytest.mark.asyncio
async def test_track_self_hosted_execute_workflow_calls_usage_quota_service(
    mock_usage_quota_service, duo_service, mock_context
) -> None:
    """Test TrackSelfHostedExecuteWorkflow decorator calls usage quota service."""
    regular_user = CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(extra={}),
    )
    current_user.set(regular_user)

    assert mock_usage_quota_service.execute is not None
    assert isinstance(mock_usage_quota_service.execute, AsyncMock)


@pytest.mark.asyncio
async def test_track_self_hosted_skips_quota_for_skip_users(
    mock_usage_quota_service, mock_user_with_skip_usage_cutoff, duo_service
) -> None:
    """Test TrackSelfHostedExecuteWorkflow decorator skips quota check for skip users."""
    current_user.set(mock_user_with_skip_usage_cutoff)

    assert not mock_usage_quota_service.execute.called


@pytest.mark.asyncio
async def test_decorator_extracts_correct_context_from_execute_workflow_request(
    mock_usage_quota_service,
) -> None:
    """Test that decorator extracts correct GL context from ExecuteWorkflow requests."""
    request_event = contract_pb2.ClientEvent(
        startRequest=contract_pb2.StartWorkflowRequest(
            workflowDefinition="test_workflow_definition", goal="test goal"
        )
    )

    assert request_event.startRequest.workflowDefinition == "test_workflow_definition"
    assert request_event.HasField("startRequest")


@pytest.mark.asyncio
async def test_decorator_extracts_correct_context_from_track_self_hosted_request(
    mock_usage_quota_service,
) -> None:
    """Test that decorator extracts correct GL context from TrackSelfHostedExecuteWorkflow requests."""
    request_event = contract_pb2.TrackSelfHostedClientEvent(
        requestID="test-request-id",
        workflowID="test-workflow-id",
        featureQualifiedName="test_feature_name",
        featureAiCatalogItem=True,
    )

    assert request_event.featureQualifiedName == "test_feature_name"
    assert request_event.featureAiCatalogItem is True


@pytest.mark.asyncio
async def test_usage_quota_event_types_are_correct() -> None:
    """Test that the correct UsageQuotaEvent types are used for each endpoint."""
    assert UsageQuotaEvent.DAP_FLOW_ON_EXECUTE
    assert UsageQuotaEvent.DAP_FLOW_ON_GENERATE_TOKEN

    assert (
        UsageQuotaEvent.DAP_FLOW_ON_EXECUTE.value
        == "duo_agent_platform_workflow_on_execute"
    )
    assert (
        UsageQuotaEvent.DAP_FLOW_ON_GENERATE_TOKEN.value
        == "duo_agent_platform_workflow_on_generate_token"
    )


class TestGenerateTokenVersionBasedQuotaCheck:
    """Tests for version-based usage quota check in GenerateToken."""

    @pytest.fixture(name="regular_user")
    def regular_user_fixture(self):
        return CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(extra={}),
        )

    @pytest.fixture(name="generate_token_request")
    def generate_token_request_fixture(self):
        return contract_pb2.GenerateTokenRequest(
            workflowDefinition="test_workflow",
        )

    @pytest.fixture(name="decorated_func")
    def decorated_func_fixture(self, mock_usage_quota_service):
        from duo_workflow_service.interceptors.route.usage_quota import (
            has_sufficient_usage_quota,
        )

        inner = AsyncMock(return_value="ok")
        inner.__name__ = "GenerateToken"
        inner.__qualname__ = "DuoWorkflowServicer.GenerateToken"

        decorated = has_sufficient_usage_quota(
            event=UsageQuotaEvent.DAP_FLOW_ON_GENERATE_TOKEN
        )(inner)

        return decorated, inner, mock_usage_quota_service

    @pytest.fixture(autouse=True)
    def setup_user(self, regular_user):
        token = current_user.set(regular_user)
        yield
        current_user.reset(token)

    @pytest.mark.asyncio
    async def test_self_managed_below_18_9_runs_quota_check(
        self,
        mock_context,
        generate_token_request,
        decorated_func,
    ):
        """Quota check should run for self-managed instances with version < 18.9."""
        decorated, inner, service = decorated_func
        event_token = current_event_context.set(EventContext(realm="self-managed"))
        version_token = gitlab_version.set("18.8.0")

        try:
            await decorated(None, generate_token_request, mock_context, service=service)

            service.execute.assert_called_once()
            inner.assert_called_once()
        finally:
            current_event_context.reset(event_token)
            gitlab_version.reset(version_token)

    @pytest.mark.asyncio
    async def test_self_managed_at_18_9_skips_quota_check(
        self,
        mock_context,
        generate_token_request,
        decorated_func,
    ):
        """Quota check should be skipped for self-managed instances with version >= 18.9."""
        decorated, inner, service = decorated_func
        event_token = current_event_context.set(EventContext(realm="self-managed"))
        version_token = gitlab_version.set("18.9.0")

        try:
            await decorated(None, generate_token_request, mock_context, service=service)

            service.execute.assert_not_called()
            inner.assert_called_once()
        finally:
            current_event_context.reset(event_token)
            gitlab_version.reset(version_token)

    @pytest.mark.asyncio
    async def test_saas_skips_quota_check(
        self,
        mock_context,
        generate_token_request,
        decorated_func,
    ):
        """Quota check should be skipped for SaaS (non-self-managed) instances."""
        decorated, inner, service = decorated_func
        event_token = current_event_context.set(EventContext(realm="saas"))
        version_token = gitlab_version.set("18.8.0")

        try:
            await decorated(None, generate_token_request, mock_context, service=service)

            service.execute.assert_not_called()
            inner.assert_called_once()
        finally:
            current_event_context.reset(event_token)
            gitlab_version.reset(version_token)

    @pytest.mark.asyncio
    async def test_no_realm_skips_quota_check(
        self,
        mock_context,
        generate_token_request,
        decorated_func,
    ):
        """Quota check should be skipped when realm is None."""
        decorated, inner, service = decorated_func
        event_token = current_event_context.set(EventContext(realm=None))

        try:
            await decorated(None, generate_token_request, mock_context, service=service)

            service.execute.assert_not_called()
            inner.assert_called_once()
        finally:
            current_event_context.reset(event_token)

    @pytest.mark.asyncio
    async def test_self_managed_no_version_runs_quota_check(
        self,
        mock_context,
        generate_token_request,
        decorated_func,
    ):
        """Quota check should run when self-managed but version is not set."""
        decorated, inner, service = decorated_func
        event_token = current_event_context.set(EventContext(realm="self-managed"))
        version_token = gitlab_version.set(None)

        try:
            await decorated(None, generate_token_request, mock_context, service=service)

            service.execute.assert_called_once()
            inner.assert_called_once()
        finally:
            current_event_context.reset(event_token)
            gitlab_version.reset(version_token)

    @pytest.mark.asyncio
    async def test_self_managed_invalid_version_runs_quota_check(
        self,
        mock_context,
        generate_token_request,
        decorated_func,
    ):
        """Quota check should run when self-managed but version is invalid."""
        decorated, inner, service = decorated_func
        event_token = current_event_context.set(EventContext(realm="self-managed"))
        version_token = gitlab_version.set("not-a-version")

        try:
            await decorated(None, generate_token_request, mock_context, service=service)

            service.execute.assert_called_once()
            inner.assert_called_once()
        finally:
            current_event_context.reset(event_token)
            gitlab_version.reset(version_token)

    @pytest.mark.asyncio
    async def test_self_managed_below_18_9_with_skip_user_skips_quota_check(
        self,
        mock_context,
        generate_token_request,
        decorated_func,
    ):
        """Quota check should be skipped even for old self-managed if user has skip claim."""
        decorated, inner, service = decorated_func
        skip_user = CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(extra={SKIP_USAGE_CUTOFF_CLAIM: True}),
        )
        current_user.set(skip_user)

        event_token = current_event_context.set(EventContext(realm="self-managed"))
        version_token = gitlab_version.set("18.7.0")

        try:
            await decorated(None, generate_token_request, mock_context, service=service)

            service.execute.assert_not_called()
            inner.assert_called_once()
        finally:
            current_event_context.reset(event_token)
            gitlab_version.reset(version_token)
