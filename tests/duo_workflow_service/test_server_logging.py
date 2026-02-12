# pylint: disable=file-naming-for-tests
"""Tests for enhanced logging functionality in the Duo Workflow Service server module."""

from typing import AsyncIterable
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest
from gitlab_cloud_connector import CloudConnectorUser

from contract import contract_pb2
from duo_workflow_service import server as server_module
from duo_workflow_service.executor.outbox import OutboxSignal
from duo_workflow_service.interceptors.authentication_interceptor import current_user
from duo_workflow_service.server import DuoWorkflowService


def create_mock_internal_event_client():
    """Helper function to create a mock internal event client for tests."""
    mock_client = MagicMock()
    mock_client.track_event = MagicMock()
    return mock_client


@pytest.fixture(autouse=True)
def mock_usage_quota_service(mock_duo_workflow_service_container):
    """Auto-use fixture to properly wire DI container and mock UsageQuotaService.

    This ensures the @has_sufficient_usage_quota decorator works correctly.
    """

    service_instance = MagicMock()
    service_instance.execute = AsyncMock()
    service_instance.aclose = AsyncMock()

    mock_duo_workflow_service_container.wire(modules=[server_module])
    mock_duo_workflow_service_container.usage_quota.service.override(service_instance)

    yield service_instance

    mock_duo_workflow_service_container.usage_quota.service.reset_override()


@pytest.mark.asyncio
@patch("duo_workflow_service.server.log")
@patch("duo_workflow_service.server.current_event_context")
@patch("duo_workflow_service.server.AbstractWorkflow")
@patch("duo_workflow_service.server.resolve_workflow_class")
async def test_execute_workflow_enhanced_logging_with_context(
    mock_resolve_workflow,
    mock_abstract_workflow_class,
    mock_current_event_context,
    mock_log,
):
    """Test that the enhanced logging includes additional context fields from event context."""
    # Import here to avoid import-outside-toplevel warning
    from lib.internal_events.context import (  # pylint: disable=import-outside-toplevel
        EventContext,
    )

    # Setup mocks
    mock_workflow = mock_abstract_workflow_class.return_value
    mock_workflow.is_done = True
    mock_workflow.run = AsyncMock()
    mock_workflow.cleanup = AsyncMock()
    mock_workflow.get_from_outbox = AsyncMock(
        return_value=OutboxSignal.NO_MORE_OUTBOUND_REQUESTS
    )
    mock_resolve_workflow.return_value = mock_abstract_workflow_class

    # Setup event context with test data
    test_event_context = EventContext(
        instance_id="test-instance-123",
        host_name="gitlab.example.com",
        realm="saas",
        is_gitlab_team_member=True,
        global_user_id="user-456",
        correlation_id="corr-789",
    )
    mock_current_event_context.get.return_value = test_event_context

    async def mock_request_iterator() -> AsyncIterable[contract_pb2.ClientEvent]:
        yield contract_pb2.ClientEvent(
            startRequest=contract_pb2.StartWorkflowRequest(
                workflowID="test-workflow-123",
                workflowDefinition="software_development",
            )
        )

    # Setup user and context
    user = CloudConnectorUser(authenticated=True, is_debug=True)
    current_user.set(user)
    mock_context = MagicMock(spec=grpc.ServicerContext)
    mock_context.invocation_metadata.return_value = []

    # Setup servicer
    servicer = DuoWorkflowService()

    # Execute the test
    mock_internal_event_client = create_mock_internal_event_client()
    result = servicer.ExecuteWorkflow(
        mock_request_iterator(),
        mock_context,
        internal_event_client=mock_internal_event_client,
    )

    # Consume the async iterator to trigger the logging
    with pytest.raises(StopAsyncIteration):
        await anext(result)

    # Verify the enhanced logging was called with expected context
    mock_log.info.assert_called()
    log_calls = mock_log.info.call_args_list

    # Find the "Starting workflow" log call
    starting_workflow_call = None
    for log_call in log_calls:
        if len(log_call[0]) > 0 and "Starting workflow" in log_call[0][0]:
            starting_workflow_call = log_call
            break

    assert starting_workflow_call is not None, "Starting workflow log call not found"

    # Verify the extra context fields were included
    call_kwargs = starting_workflow_call[1]
    assert "extra" in call_kwargs
    extra_fields = call_kwargs["extra"]

    assert extra_fields["workflow_id"] == "test-workflow-123"
    assert extra_fields["workflow_definition"] == "software_development"
    assert extra_fields["instance_id"] == "test-instance-123"
    assert extra_fields["host_name"] == "gitlab.example.com"
    assert extra_fields["realm"] == "saas"
    assert extra_fields["is_gitlab_team_member"] == "True"
    assert extra_fields["global_user_id"] == "user-456"
    assert extra_fields["correlation_id"] == "corr-789"


@pytest.mark.asyncio
@patch("duo_workflow_service.server.log")
@patch("duo_workflow_service.server.current_event_context")
@patch("duo_workflow_service.server.AbstractWorkflow")
@patch("duo_workflow_service.server.resolve_workflow_class")
async def test_execute_workflow_enhanced_logging_without_context(
    mock_resolve_workflow,
    mock_abstract_workflow_class,
    mock_current_event_context,
    mock_log,
):
    """Test that the enhanced logging handles missing event context gracefully."""
    # Setup mocks
    mock_workflow = mock_abstract_workflow_class.return_value
    mock_workflow.is_done = True
    mock_workflow.run = AsyncMock()
    mock_workflow.cleanup = AsyncMock()
    mock_workflow.get_from_outbox = AsyncMock(
        return_value=OutboxSignal.NO_MORE_OUTBOUND_REQUESTS
    )
    mock_resolve_workflow.return_value = mock_abstract_workflow_class

    # Setup event context to return None
    mock_current_event_context.get.return_value = None

    async def mock_request_iterator() -> AsyncIterable[contract_pb2.ClientEvent]:
        yield contract_pb2.ClientEvent(
            startRequest=contract_pb2.StartWorkflowRequest(
                workflowID="test-workflow-123",
                workflowDefinition="software_development",
            )
        )

    # Setup user and context
    user = CloudConnectorUser(authenticated=True, is_debug=True)
    current_user.set(user)
    mock_context = MagicMock(spec=grpc.ServicerContext)
    mock_context.invocation_metadata.return_value = []

    # Setup servicer
    servicer = DuoWorkflowService()

    # Execute the test
    mock_internal_event_client = create_mock_internal_event_client()
    result = servicer.ExecuteWorkflow(
        mock_request_iterator(),
        mock_context,
        internal_event_client=mock_internal_event_client,
    )

    # Consume the async iterator to trigger the logging
    with pytest.raises(StopAsyncIteration):
        await anext(result)

    # Verify the debug log was called for missing context
    debug_calls = [
        debug_call
        for debug_call in mock_log.debug.call_args_list
        if len(debug_call[0]) > 0 and "Event context not available" in debug_call[0][0]
    ]
    assert len(debug_calls) == 1, "Debug log for missing event context not found"

    # Verify the enhanced logging was called with basic context only
    mock_log.info.assert_called()
    log_calls = mock_log.info.call_args_list

    # Find the "Starting workflow" log call
    starting_workflow_call = None
    for log_call in log_calls:
        if len(log_call[0]) > 0 and "Starting workflow" in log_call[0][0]:
            starting_workflow_call = log_call
            break

    assert starting_workflow_call is not None, "Starting workflow log call not found"

    # Verify only basic context fields were included (no event context fields)
    call_kwargs = starting_workflow_call[1]
    assert "extra" in call_kwargs
    extra_fields = call_kwargs["extra"]

    assert extra_fields["workflow_id"] == "test-workflow-123"
    assert extra_fields["workflow_definition"] == "software_development"

    # Verify event context fields are not present
    assert "instance_id" not in extra_fields
    assert "host_name" not in extra_fields
    assert "realm" not in extra_fields
    assert "is_gitlab_team_member" not in extra_fields
    assert "global_user_id" not in extra_fields
    assert "correlation_id" not in extra_fields
