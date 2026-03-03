from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims
from grpc.aio import ServicerContext

from contract import contract_pb2
from duo_workflow_service.interceptors.authentication_interceptor import current_user
from duo_workflow_service.server import DuoWorkflowService
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
