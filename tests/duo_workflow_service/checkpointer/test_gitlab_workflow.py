# pylint: disable=comparison-with-callable,import-outside-toplevel,line-too-long,no-else-raise,no-else-return,too-many-lines
import asyncio
import json
import zlib
from asyncio import CancelledError
from typing import Any, Optional, Sequence, TypedDict
from unittest.mock import ANY, AsyncMock, Mock, call, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import MemorySaver

from duo_workflow_service.checkpointer.gitlab_workflow import (
    GitLabWorkflow,
    WorkflowStatusEventEnum,
    _dict_of_list_delta,
    _get_orbit_tool_calls,
    _serialize_channel_blobs,
)
from duo_workflow_service.checkpointer.gitlab_workflow_utils import compress_checkpoint
from duo_workflow_service.entities.state import WorkflowStatusEnum
from duo_workflow_service.errors.typing import NotifiableException
from duo_workflow_service.gitlab.http_client import (
    GitLabHttpResponse,
    checkpoint_decoder,
)
from duo_workflow_service.interceptors.authentication_interceptor import current_user
from duo_workflow_service.json_encoder.encoder import CustomEncoder
from duo_workflow_service.status_updater.gitlab_status_updater import (
    UnsupportedStatusEvent,
)
from lib.billing_events import BillingEvent, ExecutionEnvironment
from lib.billing_events.service import LLMOperation
from lib.context import current_model_metadata_context, llm_operations
from lib.context.tool_executions import init_tool_executions, tool_executions
from lib.internal_events import InternalEventAdditionalProperties
from lib.internal_events.event_enum import EventEnum, EventLabelEnum, EventPropertyEnum


class CustomRunnableConfig(TypedDict):
    configurable: Optional[dict]


@pytest.fixture(name="http_client_for_retry")
def http_client_for_retry_fixture(http_client, workflow_id):
    async def mock_aget(path, **_kwargs):
        if (
            path
            == f"/api/v4/ai/duo_workflows/workflows/{workflow_id}/checkpoints?per_page=1"
        ):
            # Return a checkpoint to make aget_tuple return something (not None)
            return [
                {
                    "thread_ts": "some-id",
                    "parent_ts": None,
                    "checkpoint": {"id": "some-id"},
                    "metadata": {},
                }
            ]
        elif path == f"/api/v4/ai/duo_workflows/workflows/{workflow_id}":
            # Return status for workflow completion
            return GitLabHttpResponse(status_code=200, body={"status": "finished"})
        else:
            raise ValueError(f"Unexpected path: {path}")

    http_client.aget = mock_aget
    http_client.apatch.return_value = GitLabHttpResponse(status_code=200, body={})
    http_client.apost.return_value = {"status": 200}

    return http_client


@pytest.fixture(name="config")
def config_fixture() -> CustomRunnableConfig:
    return {"configurable": {"thread_id": "1234", "checkpoint_id": "5678"}}


@pytest.fixture(name="http_client")
def http_client_fixture():
    return AsyncMock()


@pytest.fixture(name="mock_user")
def mock_user_fixture():
    return CloudConnectorUser(
        authenticated=True, claims=UserClaims(gitlab_instance_uid="test-instance-123")
    )


@pytest.fixture(name="mock_llm_operations")
def mock_llm_operations_fixture():
    """Set up LLM operations context for billing tests."""
    operations = [
        {
            "token_count": 100,
            "model_id": "claude-3-sonnet",
            "model_engine": "anthropic",
            "model_provider": "anthropic",
            "prompt_tokens": 80,
            "completion_tokens": 20,
            "agent_name": "foo",
            "cache_read_tokens": 50,
            "cache_write_tokens": 10,
        },
        {
            "token_count": 150,
            "model_id": "gpt-4",
            "model_engine": "openai",
            "model_provider": "openai",
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "agent_name": "bar",
            "cache_read_tokens": 10,
            "cache_write_tokens": 5,
        },
    ]
    llm_operations.set(operations)
    # Return the expected serialized output after LLMOperation.model_validate().model_dump()
    # which includes default values for operation_type
    return [LLMOperation.model_validate(op).model_dump() for op in operations]


@pytest.fixture(autouse=True)
def prepare_container(  # pylint: disable=unused-argument  # fixture-on-fixture ordering dep
    mock_duo_workflow_service_container,
):
    pass


@pytest.fixture(name="workflow_config")
def workflow_config_fixture():
    return {
        "first_checkpoint": None,
        "latest_checkpoint": None,
        "workflow_status": "created",
        "agent_privileges_names": ["read_repository"],
        "pre_approved_agent_privileges_names": [],
        "mcp_enabled": True,
        "allow_agent_to_request_user": True,
        "archived": False,
        "stalled": False,
    }


@pytest.fixture(name="gitlab_workflow")
def gitlab_workflow_fixture(
    http_client,
    workflow_id,
    workflow_type,
    workflow_config,
):
    return GitLabWorkflow(
        http_client,
        workflow_id,
        workflow_type,
        workflow_config,
    )


@pytest.fixture(name="checkpoint_body")
def checkpoint_body_fixture():
    return """[
        {
            "thread_ts": "5678",
            "parent_ts": null,
            "checkpoint": {
                "id": "5678",
                "channel_values": {
                    "conversation_history": {
                        "planner": [
                            {
                                "type": "SystemMessage",
                                "content": "You are a planner. Could you create a detailed plan that the execution agent"
                            },
                            {
                                "type": "HumanMessage",
                                "content": "Work"
                            },
                            {
                                "type": "AIMessage",
                                "content": "I did"
                            }
                        ]
                    }
                }
            },
            "metadata": {"timestamp": "2024-01-01"}
        }
    ]"""


@pytest.fixture(name="checkpoint_data")
def checkpoint_data_fixture(checkpoint_body):
    return json.loads(checkpoint_body, object_hook=checkpoint_decoder)


@pytest.fixture(name="compressed_checkpoint_data")
def compressed_checkpoint_data_fixture(checkpoint_data):
    compressed_checkpoint = compress_checkpoint(checkpoint_data[0]["checkpoint"])
    compressed_data = [
        {
            **checkpoint_data[0],
            "compressed_checkpoint": compressed_checkpoint,
        }
    ]
    del compressed_data[0]["checkpoint"]
    return compressed_data


@pytest.fixture(name="checkpoint_metadata")
def checkpoint_metadata_fixture():
    metadata = CheckpointMetadata()
    metadata["writes"] = {"some_node": {"status": "Created", "last_human_input": {}}}
    return metadata


@pytest.mark.asyncio
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_workflow_event_tracking_for_cancelled_workflow(
    mock_duo_workflow_metrics,
    gitlab_workflow,
    http_client,
    workflow_id,
    workflow_type,
    internal_event_client: Mock,
):
    # Create a workflow config with no checkpoint to trigger START
    workflow_config = {
        "first_checkpoint": None,
        "workflow_status": "created",
        "agent_privileges_names": ["read_repository"],
        "pre_approved_agent_privileges_names": [],
        "mcp_enabled": True,
        "allow_agent_to_request_user": True,
        "archived": False,
        "stalled": False,
    }

    gitlab_workflow = GitLabWorkflow(
        http_client,
        workflow_id,
        workflow_type,
        workflow_config,  # type: ignore[arg-type]
    )
    gitlab_workflow._internal_event_client = internal_event_client

    async def mock_aget(path, **_kwargs):
        if (
            path
            == f"/api/v4/ai/duo_workflows/workflows/{workflow_id}/checkpoints?per_page=1"
        ):
            return GitLabHttpResponse(
                status_code=200, body=[]
            )  # No checkpoints for a new workflow
        elif path == f"/api/v4/ai/duo_workflows/workflows/{workflow_id}":
            return GitLabHttpResponse(
                status_code=200, body={"status": "stopped"}
            )  # Workflow was cancelled
        raise ValueError(f"Unexpected path: {path}")

    async def mock_apatch(  # pylint: disable=unused-argument  # http_client.apatch callback signature
        path, **kwargs
    ):
        return GitLabHttpResponse(status_code=200, body={})

    http_client.aget.side_effect = mock_aget
    http_client.apatch.side_effect = mock_apatch

    async with gitlab_workflow as workflow:
        assert isinstance(workflow, GitLabWorkflow)

    http_client.apatch.assert_called_once_with(
        path=f"/api/v4/ai/duo_workflows/workflows/{workflow_id}",
        body=json.dumps({"status_event": WorkflowStatusEventEnum.START.value}),
        parse_json=True,
    )

    mock_duo_workflow_metrics.count_agent_platform_session_start.assert_called_once_with(
        flow_type=workflow_type.value,
    )

    assert internal_event_client.track_event.call_count == 2
    internal_event_client.track_event.assert_has_calls(
        [
            call(
                event_name=EventEnum.WORKFLOW_START.value,
                additional_properties=InternalEventAdditionalProperties(
                    label=EventLabelEnum.WORKFLOW_START_LABEL.value,
                    property=EventPropertyEnum.WORKFLOW_ID.value,
                    value="1234",
                ),
                category=workflow_type.value,
            ),
            call(
                event_name=EventEnum.WORKFLOW_FINISH_SUCCESS.value,
                additional_properties=InternalEventAdditionalProperties(
                    label=EventLabelEnum.WORKFLOW_FINISH_LABEL.value,
                    property=EventPropertyEnum.CANCELLED_BY_USER.value,
                    value="1234",
                    duration_seconds=ANY,
                ),
                category=workflow_type.value,
            ),
        ]
    )

    mock_duo_workflow_metrics.count_agent_platform_session_success.assert_called_once_with(
        flow_type=workflow_type.value,
    )


@pytest.mark.asyncio
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_workflow_context_manager_success(
    mock_duo_workflow_metrics,
    gitlab_workflow,
    http_client,
    workflow_id,
    workflow_type,
    internal_event_client: Mock,
):
    # Create a workflow config with no checkpoint to trigger START
    workflow_config = {
        "first_checkpoint": None,
        "workflow_status": "created",
        "agent_privileges_names": ["read_repository"],
        "pre_approved_agent_privileges_names": [],
        "mcp_enabled": True,
        "allow_agent_to_request_user": True,
        "archived": False,
        "stalled": False,
    }

    gitlab_workflow = GitLabWorkflow(
        http_client,
        workflow_id,
        workflow_type,
        workflow_config,  # type: ignore[arg-type]
    )
    gitlab_workflow._internal_event_client = internal_event_client

    async def mock_aget(path, **_kwargs):
        if (
            path
            == f"/api/v4/ai/duo_workflows/workflows/{workflow_id}/checkpoints?per_page=1"
        ):
            return GitLabHttpResponse(
                status_code=200, body=[]
            )  # No checkpoints for a new workflow
        elif path == f"/api/v4/ai/duo_workflows/workflows/{workflow_id}":
            return GitLabHttpResponse(status_code=200, body={"status": "finished"})
        raise ValueError(f"Unexpected path: {path}")

    async def mock_apatch(  # pylint: disable=unused-argument  # http_client.apatch callback signature
        path, **kwargs
    ):
        return GitLabHttpResponse(status_code=200, body={})

    http_client.aget.side_effect = mock_aget
    http_client.apatch.side_effect = mock_apatch

    async with gitlab_workflow as workflow:
        assert isinstance(workflow, GitLabWorkflow)

    http_client.apatch.assert_called_once_with(
        path=f"/api/v4/ai/duo_workflows/workflows/{workflow_id}",
        body=json.dumps({"status_event": WorkflowStatusEventEnum.START.value}),
        parse_json=True,
    )

    mock_duo_workflow_metrics.count_agent_platform_session_start.assert_called_once_with(
        flow_type=workflow_type.value,
    )

    assert internal_event_client.track_event.call_count == 2

    internal_event_client.track_event.assert_has_calls(
        [
            call(
                event_name=EventEnum.WORKFLOW_START.value,
                additional_properties=InternalEventAdditionalProperties(
                    label=EventLabelEnum.WORKFLOW_START_LABEL.value,
                    property=EventPropertyEnum.WORKFLOW_ID.value,
                    value="1234",
                ),
                category=workflow_type,
            ),
            call(
                event_name=EventEnum.WORKFLOW_FINISH_SUCCESS.value,
                additional_properties=InternalEventAdditionalProperties(
                    label=EventLabelEnum.WORKFLOW_FINISH_LABEL.value,
                    property=EventPropertyEnum.WORKFLOW_COMPLETED.value,
                    value="1234",
                    duration_seconds=ANY,
                ),
                category=workflow_type,
            ),
        ]
    )

    mock_duo_workflow_metrics.count_agent_platform_session_success.assert_called_once_with(
        flow_type=workflow_type.value,
    )


@pytest.mark.asyncio
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
@patch("duo_workflow_service.checkpointer.gitlab_workflow.log_exception")
async def test_workflow_context_manager_startup_error(
    mock_log_exception,
    mock_duo_workflow_metrics,
    http_client,
    workflow_id,
    workflow_type,
    workflow_config,
    internal_event_client: Mock,
):
    gitlab_workflow = GitLabWorkflow(
        http_client,
        workflow_id,
        workflow_type,
        workflow_config,
    )
    gitlab_workflow._internal_event_client = internal_event_client

    # Set up the mock to raise an exception during the first status update call
    # but succeed on the second call (DROP status)
    call_count = 0

    async def mock_apatch(  # pylint: disable=unused-argument  # http_client.apatch callback signature
        path, **kwargs
    ):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # First call (START status) returns 400
            return GitLabHttpResponse(
                status_code=400,
                body={"message": "Can not start workflow that has status failed"},
            )
        else:
            # Second call (DROP status) succeeds
            return GitLabHttpResponse(status_code=200, body={})

    http_client.apatch.side_effect = mock_apatch

    with pytest.raises(UnsupportedStatusEvent) as exc_info:
        async with gitlab_workflow:
            pytest.fail("Context manager body should not execute")

    assert (
        str(exc_info.value)
        == "Session status cannot be updated due to bad status event: start, error: {'message': 'Can not start workflow that has status failed'}"
    )

    # Verify that apatch was called once - with START (which failed)
    assert http_client.apatch.call_count == 1

    # Check the first call was START
    first_call = http_client.apatch.call_args_list[0]
    assert first_call[1]["path"] == f"/api/v4/ai/duo_workflows/workflows/{workflow_id}"
    assert (
        json.loads(first_call[1]["body"])["status_event"]
        == WorkflowStatusEventEnum.START.value
    )

    internal_event_client.track_event.assert_called_once_with(
        event_name=EventEnum.WORKFLOW_REJECT.value,
        additional_properties=InternalEventAdditionalProperties(
            label=EventLabelEnum.WORKFLOW_REJECT_LABEL.value,
            property="UnsupportedStatusEvent(\"Session status cannot be updated due to bad status event: start, error: {'message': 'Can not start workflow that has status failed'}\")",
            value=workflow_id,
        ),
        category=workflow_type,
    )

    assert mock_log_exception.call_count == 1
    assert isinstance(mock_log_exception.call_args[0][0], UnsupportedStatusEvent)

    # Verify the failure metric was called
    mock_duo_workflow_metrics.count_agent_platform_session_failure.assert_not_called()


@pytest.mark.asyncio
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
@patch("duo_workflow_service.checkpointer.gitlab_workflow.log_exception")
async def test_workflow_context_manager_startup_error_with_status_update_failure(
    mock_log_exception,
    mock_duo_workflow_metrics,
    http_client,
    workflow_id,
    workflow_type,
    workflow_config,
    internal_event_client: Mock,
):
    gitlab_workflow = GitLabWorkflow(
        http_client,
        workflow_id,
        workflow_type,
        workflow_config,
    )
    gitlab_workflow._internal_event_client = internal_event_client

    # Set up the mock to raise an exception during the first status update call
    # and fail on the second call (DROP status)
    call_count = 0

    # Create a specific instance of the error to use in both the mock and assertion
    status_error = ConnectionError("Status update failed")

    async def mock_apatch(  # pylint: disable=unused-argument  # http_client.apatch callback signature
        path, **kwargs
    ):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # First call (START status) raises an exception
            raise ValueError("Startup error simulated")
        else:
            # Second call (DROP status) fails with a connection error
            raise status_error

    http_client.apatch.side_effect = mock_apatch

    # Set up aget to return empty checkpoints (for a new workflow)
    async def mock_aget(path, **_kwargs):
        if "checkpoints" in path:
            return GitLabHttpResponse(status_code=200, body=[])
        else:
            return GitLabHttpResponse(status_code=200, body={"status": "created"})

    http_client.aget.side_effect = mock_aget

    with pytest.raises(ValueError) as exc_info:
        async with gitlab_workflow:
            pytest.fail("Context manager body should not execute")

    assert str(exc_info.value) == "Startup error simulated"

    # Verify that apatch was called twice - first with START (which failed) and then with DROP (which also failed)
    assert http_client.apatch.call_count == 2

    # Check the first call was START
    first_call = http_client.apatch.call_args_list[0]
    assert first_call[1]["path"] == f"/api/v4/ai/duo_workflows/workflows/{workflow_id}"
    assert (
        json.loads(first_call[1]["body"])["status_event"]
        == WorkflowStatusEventEnum.START.value
    )

    # Check the second call was DROP
    second_call = http_client.apatch.call_args_list[1]
    assert second_call[1]["path"] == f"/api/v4/ai/duo_workflows/workflows/{workflow_id}"
    assert (
        json.loads(second_call[1]["body"])["status_event"]
        == WorkflowStatusEventEnum.DROP.value
    )

    internal_event_client.track_event.assert_called_once_with(
        event_name=EventEnum.WORKFLOW_FINISH_FAILURE.value,
        additional_properties=InternalEventAdditionalProperties(
            label=EventLabelEnum.WORKFLOW_FINISH_LABEL.value,
            property="ValueError('Startup error simulated')",
            value=workflow_id,
            error_type="ValueError",
        ),
        category=workflow_type,
    )

    # Verify the status update error was logged
    assert mock_log_exception.call_count == 2
    assert isinstance(mock_log_exception.call_args_list[0][0][0], ValueError)
    assert str(mock_log_exception.call_args_list[0][0][0]) == "Startup error simulated"
    assert mock_log_exception.call_args_list[1][0][0] == status_error
    assert (
        mock_log_exception.call_args_list[1][1]["extra"]["workflow_id"] == workflow_id
    )
    assert (
        mock_log_exception.call_args_list[1][1]["extra"]["context"]
        == "Failed to update workflow status during startup error"
    )

    # Verify the failure metric was called
    mock_duo_workflow_metrics.count_agent_platform_session_failure.assert_called_once_with(
        flow_type=workflow_type.value,
        failure_reason="ValueError",
    )


@pytest.mark.asyncio
async def test_workflow_context_manager_resume_interrupted(
    gitlab_workflow,
    http_client,
    workflow_id,
    workflow_type,
    internal_event_client: Mock,
):
    # Create a workflow config with a checkpoint and INPUT_REQUIRED status
    workflow_config = {
        "first_checkpoint": {"checkpoint": "{}"},
        "workflow_status": WorkflowStatusEnum.INPUT_REQUIRED,
        "agent_privileges_names": ["read_repository"],
        "pre_approved_agent_privileges_names": [],
        "mcp_enabled": True,
        "allow_agent_to_request_user": True,
        "archived": False,
        "stalled": False,
    }

    gitlab_workflow = GitLabWorkflow(
        http_client,
        workflow_id,
        workflow_type,
        workflow_config,  # type: ignore[arg-type]
    )

    gitlab_workflow._internal_event_client = internal_event_client
    gitlab_workflow._status_handler = AsyncMock()

    async with gitlab_workflow as workflow:
        assert isinstance(workflow, GitLabWorkflow)

    gitlab_workflow._status_handler.update_workflow_status.assert_called_once_with(
        workflow_id, WorkflowStatusEventEnum.RESUME
    )

    internal_event_client.track_event.assert_has_calls(
        [
            call(
                event_name=EventEnum.WORKFLOW_RESUME.value,
                additional_properties=InternalEventAdditionalProperties(
                    label=EventLabelEnum.WORKFLOW_RESUME_LABEL.value,
                    property=EventPropertyEnum.WORKFLOW_RESUME_BY_PLAN_AFTER_INPUT.value,
                    value=workflow_id,
                ),
                category=workflow_type,
            ),
        ]
    )


@pytest.mark.asyncio
async def test_workflow_context_manager_resume_interrupted_approval(
    gitlab_workflow,
    http_client,
    workflow_id,
    workflow_type,
    internal_event_client: Mock,
):
    # Create a workflow config with a checkpoint and PLAN_APPROVAL_REQUIRED status
    workflow_config = {
        "first_checkpoint": {"checkpoint": "{}"},
        "workflow_status": WorkflowStatusEnum.PLAN_APPROVAL_REQUIRED,
        "agent_privileges_names": ["read_repository"],
        "pre_approved_agent_privileges_names": [],
        "mcp_enabled": True,
        "allow_agent_to_request_user": True,
        "archived": False,
        "stalled": False,
    }

    gitlab_workflow = GitLabWorkflow(
        http_client,
        workflow_id,
        workflow_type,
        workflow_config,  # type: ignore[arg-type]
    )

    gitlab_workflow._internal_event_client = internal_event_client
    gitlab_workflow._status_handler = AsyncMock()

    async with gitlab_workflow as workflow:
        assert isinstance(workflow, GitLabWorkflow)

    gitlab_workflow._status_handler.update_workflow_status.assert_called_once_with(
        workflow_id, WorkflowStatusEventEnum.RESUME
    )

    internal_event_client.track_event.assert_has_calls(
        [
            call(
                event_name=EventEnum.WORKFLOW_RESUME.value,
                additional_properties=InternalEventAdditionalProperties(
                    label=EventLabelEnum.WORKFLOW_RESUME_LABEL.value,
                    property=EventPropertyEnum.WORKFLOW_RESUME_BY_PLAN_AFTER_APPROVAL.value,
                    value=workflow_id,
                ),
                category=workflow_type,
            ),
        ]
    )


@pytest.mark.asyncio
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_workflow_context_manager_retry_success(
    mock_duo_workflow_metrics,
    gitlab_workflow,
    http_client_for_retry,
    workflow_id,
    workflow_type,
    internal_event_client: Mock,
):
    # Create a workflow config with a checkpoint and a status that will trigger RETRY
    workflow_config = {
        "first_checkpoint": {"checkpoint": "{}"},
        "workflow_status": "failed",  # Only statuses that can be retried should be used
        "agent_privileges_names": ["read_repository"],
        "pre_approved_agent_privileges_names": [],
        "mcp_enabled": True,
        "allow_agent_to_request_user": True,
        "archived": False,
        "stalled": False,
    }

    gitlab_workflow = GitLabWorkflow(
        http_client_for_retry,
        workflow_id,
        workflow_type,
        workflow_config,  # type: ignore[arg-type]
    )

    gitlab_workflow._internal_event_client = internal_event_client

    async with gitlab_workflow as workflow:
        assert isinstance(workflow, GitLabWorkflow)

    http_client_for_retry.apatch.assert_called_once_with(
        path=f"/api/v4/ai/duo_workflows/workflows/{workflow_id}",
        body=json.dumps({"status_event": WorkflowStatusEventEnum.RETRY.value}),
        parse_json=True,
    )

    assert internal_event_client.track_event.call_count == 2
    internal_event_client.track_event.assert_has_calls(
        [
            call(
                event_name=EventEnum.WORKFLOW_RETRY.value,
                additional_properties=InternalEventAdditionalProperties(
                    label=EventLabelEnum.WORKFLOW_RESUME_LABEL.value,
                    property=EventPropertyEnum.WORKFLOW_RESUME_BY_USER.value,
                    value=workflow_id,
                ),
                category=workflow_type,
            ),
            call(
                event_name=EventEnum.WORKFLOW_FINISH_SUCCESS.value,
                additional_properties=InternalEventAdditionalProperties(
                    label=EventLabelEnum.WORKFLOW_FINISH_LABEL.value,
                    property=EventPropertyEnum.WORKFLOW_COMPLETED.value,
                    value=workflow_id,
                    duration_seconds=ANY,
                ),
                category=workflow_type,
            ),
        ]
    )

    mock_duo_workflow_metrics.count_agent_platform_session_success.assert_called_once_with(
        flow_type=workflow_type.value,
    )
    mock_duo_workflow_metrics.count_agent_platform_session_retry.assert_called_once_with(
        flow_type=workflow_type.value,
    )


@pytest.mark.asyncio
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
@pytest.mark.parametrize(
    "exception,expected_event_name,expected_additional_properties",
    [
        (
            ValueError("Test error"),
            EventEnum.WORKFLOW_FINISH_FAILURE.value,
            InternalEventAdditionalProperties(
                label=EventLabelEnum.WORKFLOW_FINISH_LABEL.value,
                property="ValueError('Test error')",
                value="1234",
                error_type="ValueError",
            ),
        ),
        (
            asyncio.exceptions.CancelledError("Task cancelled"),
            EventEnum.WORKFLOW_ABORTED.value,
            InternalEventAdditionalProperties(
                label=EventLabelEnum.WORKFLOW_FINISH_LABEL.value,
                property="CancelledError('Task cancelled')",
                value="1234",
                error_type="CancelledError",
            ),
        ),
    ],
)
async def test_workflow_context_manager_error(
    mock_duo_workflow_metrics,
    gitlab_workflow,
    http_client,
    workflow_id,
    workflow_type,
    internal_event_client: Mock,
    exception,
    expected_event_name,
    expected_additional_properties,
):
    # Create a workflow config with no checkpoint to trigger START
    workflow_config = {
        "first_checkpoint": None,
        "workflow_status": "created",
        "agent_privileges_names": ["read_repository"],
        "pre_approved_agent_privileges_names": [],
        "mcp_enabled": True,
        "allow_agent_to_request_user": True,
        "archived": False,
        "stalled": False,
    }

    gitlab_workflow = GitLabWorkflow(
        http_client,
        workflow_id,
        workflow_type,
        workflow_config,  # type: ignore[arg-type]
    )
    gitlab_workflow._internal_event_client = internal_event_client

    # Mock different responses for different API calls
    def mock_aget(path, **_kwargs):
        if "checkpoints" in path:
            return GitLabHttpResponse(status_code=200, body=[])
        else:
            return GitLabHttpResponse(status_code=200, body={"status": "running"})

    http_client.aget.side_effect = mock_aget
    http_client.apatch.return_value = GitLabHttpResponse(status_code=200, body={})

    with pytest.raises(type(exception)):
        async with gitlab_workflow:
            raise exception

    http_client.apatch.assert_called_with(
        path=f"/api/v4/ai/duo_workflows/workflows/{workflow_id}",
        body=json.dumps({"status_event": WorkflowStatusEventEnum.DROP.value}),
        parse_json=True,
    )

    mock_duo_workflow_metrics.count_agent_platform_session_start.assert_called_once_with(
        flow_type=workflow_type.value,
    )

    assert internal_event_client.track_event.call_count == 2

    internal_event_client.track_event.assert_has_calls(
        [
            call(
                event_name=EventEnum.WORKFLOW_START.value,
                additional_properties=InternalEventAdditionalProperties(
                    label=EventLabelEnum.WORKFLOW_START_LABEL.value,
                    property=EventPropertyEnum.WORKFLOW_ID.value,
                    value="1234",
                ),
                category=workflow_type,
            ),
            call(
                event_name=expected_event_name,
                additional_properties=expected_additional_properties,
                category=workflow_type,
            ),
        ]
    )

    # Verify the failure metric was called
    if isinstance(exception, ValueError):
        mock_duo_workflow_metrics.count_agent_platform_session_failure.assert_called_once_with(
            flow_type=workflow_type.value,
            failure_reason="ValueError",
        )

    if isinstance(exception, CancelledError):
        mock_duo_workflow_metrics.count_agent_platform_session_abort.assert_called_once_with(
            flow_type=workflow_type.value,
        )


@pytest.mark.asyncio
async def test_aget_tuple(
    gitlab_workflow,
    http_client,
    config,
    workflow_id,
    checkpoint_data,
    compressed_checkpoint_data,
):
    mock_response = GitLabHttpResponse(
        status_code=200,
        body=compressed_checkpoint_data,
    )
    http_client.aget.return_value = mock_response

    result = await gitlab_workflow.aget_tuple(config)

    assert result is not None
    assert isinstance(result, CheckpointTuple)

    assert result.config is not None
    configurable = result.config.get("configurable")
    assert isinstance(configurable, dict)
    assert configurable.get("thread_id") == "1234"
    assert configurable.get("checkpoint_id") == "5678"

    assert result.checkpoint == checkpoint_data[0]["checkpoint"]
    assert result.metadata == checkpoint_data[0]["metadata"]

    http_client.aget.assert_called_once()
    call_kwargs = http_client.aget.call_args[1]
    assert (
        f"/api/v4/ai/duo_workflows/workflows/{workflow_id}/checkpoints"
        in call_kwargs["path"]
    )
    assert "accept_compressed=true" in call_kwargs["path"]
    assert call_kwargs.get("object_hook") == checkpoint_decoder


@pytest.mark.asyncio
async def test_aget_tuple_when_config_has_no_checkpoint_id_and_checkpoints_present(
    http_client, workflow_id, checkpoint_data, compressed_checkpoint_data, workflow_type
):
    # Set first_checkpoint to non-None to allow API call
    workflow_config = {
        "first_checkpoint": {},
        "latest_checkpoint": None,
        "workflow_status": "created",
        "agent_privileges_names": ["read_repository"],
        "pre_approved_agent_privileges_names": [],
        "mcp_enabled": True,
        "allow_agent_to_request_user": True,
        "archived": False,
        "stalled": False,
    }

    gitlab_workflow = GitLabWorkflow(
        http_client,
        workflow_id,
        workflow_type,
        workflow_config,
    )

    config: CustomRunnableConfig = {"configurable": {"thread_id": workflow_id}}

    mock_response = GitLabHttpResponse(
        status_code=200,
        body=compressed_checkpoint_data,
    )
    http_client.aget.return_value = mock_response

    result = await gitlab_workflow.aget_tuple(config)
    assert result is not None
    assert isinstance(result, CheckpointTuple)
    assert result.checkpoint == checkpoint_data[0]["checkpoint"]
    assert result.metadata == checkpoint_data[0]["metadata"]

    http_client.aget.assert_called_once()
    call_kwargs = http_client.aget.call_args[1]
    assert (
        f"/api/v4/ai/duo_workflows/workflows/{workflow_id}/checkpoints"
        in call_kwargs["path"]
    )
    assert "per_page=1" in call_kwargs["path"]
    assert "accept_compressed=true" in call_kwargs["path"]
    assert call_kwargs.get("object_hook") == checkpoint_decoder


@pytest.mark.asyncio
async def test_aget_tuple_no_checkpoints(gitlab_workflow, http_client, config):
    mock_response = GitLabHttpResponse(
        status_code=200,
        body=[],
    )
    http_client.aget.return_value = mock_response
    result = await gitlab_workflow.aget_tuple(config)
    assert result is None


@pytest.mark.asyncio
async def test_aget_tuple_when_server_returns_non_success_response(
    http_client, workflow_id, workflow_type
):
    # Set first_checkpoint to non-None to allow API call
    workflow_config = {
        "first_checkpoint": {},
        "latest_checkpoint": None,
        "workflow_status": "created",
        "agent_privileges_names": ["read_repository"],
        "pre_approved_agent_privileges_names": [],
        "mcp_enabled": True,
        "allow_agent_to_request_user": True,
        "archived": False,
        "stalled": False,
    }

    gitlab_workflow = GitLabWorkflow(
        http_client,
        workflow_id,
        workflow_type,
        workflow_config,
    )

    config: CustomRunnableConfig = {"configurable": {"thread_id": workflow_id}}

    mock_response = GitLabHttpResponse(
        status_code=400,
        body={"status": 400, "reason": "Bad request"},
    )
    gitlab_workflow._client.aget = AsyncMock(return_value=mock_response)

    with pytest.raises(Exception) as exc_info:
        await gitlab_workflow.aget_tuple(config)

    assert (
        str(exc_info.value)
        == "Failed to fetch checkpoints: {'status': 400, 'reason': 'Bad request'}"
    )

    gitlab_workflow._client.aget.assert_called_once()
    call_kwargs = gitlab_workflow._client.aget.call_args[1]
    assert (
        f"/api/v4/ai/duo_workflows/workflows/{workflow_id}/checkpoints"
        in call_kwargs["path"]
    )
    assert "per_page=1" in call_kwargs["path"]
    assert "accept_compressed=true" in call_kwargs["path"]
    assert call_kwargs.get("object_hook") == checkpoint_decoder


@pytest.mark.asyncio
async def test_alist(gitlab_workflow, http_client, workflow_id):
    checkpoints = [
        {
            "thread_ts": "checkpoint-1",
            "parent_ts": None,
            "checkpoint": {"id": "checkpoint-1", "data": "test1"},
            "metadata": {"timestamp": "2024-01-01"},
        },
        {
            "thread_ts": "checkpoint-2",
            "parent_ts": "checkpoint-1",
            "checkpoint": {"id": "checkpoint-2", "data": "test2"},
            "metadata": {"timestamp": "2024-01-02"},
        },
    ]

    compressed_checkpoints = [
        {
            **{k: v for k, v in cp.items() if k != "checkpoint"},
            "compressed_checkpoint": compress_checkpoint(cp["checkpoint"]),
        }
        for cp in checkpoints
    ]

    mock_response = GitLabHttpResponse(
        status_code=200,
        body=compressed_checkpoints,
    )
    http_client.aget.return_value = mock_response

    results: list[CheckpointTuple] = [
        checkpoint async for checkpoint in gitlab_workflow.alist(None)
    ]

    assert len(results) == 2
    assert results[0].checkpoint == checkpoints[0]["checkpoint"]
    assert results[1].checkpoint == checkpoints[1]["checkpoint"]

    call_path = http_client.aget.call_args[1]["path"]
    assert f"/api/v4/ai/duo_workflows/workflows/{workflow_id}/checkpoints" in call_path
    assert "accept_compressed=true" in call_path


@pytest.mark.asyncio
async def test_aput(
    gitlab_workflow, http_client, checkpoint_data, checkpoint_metadata, workflow_id
):
    config = {"configurable": {"checkpoint_id": "parent-checkpoint"}}
    checkpoint = checkpoint_data[0]["checkpoint"]
    checkpoint["channel_values"]["status"] = WorkflowStatusEnum.COMPLETED

    http_client.apost.return_value = GitLabHttpResponse(status_code=200, body={})

    result = await gitlab_workflow.aput(
        config, checkpoint, checkpoint_metadata, ChannelVersions()
    )

    http_client.apost.assert_called_once()
    post_call_body = json.loads(http_client.apost.call_args[1]["body"])

    assert "compressed_checkpoint" in post_call_body
    assert "checkpoint" not in post_call_body
    assert post_call_body["compressed_checkpoint"] == compress_checkpoint(checkpoint)
    assert post_call_body["thread_ts"] == checkpoint["id"]
    assert post_call_body["parent_ts"] == "parent-checkpoint"
    assert "channel_blobs" not in post_call_body

    assert result == {
        "configurable": {"thread_id": workflow_id, "checkpoint_id": checkpoint["id"]}
    }


@pytest.mark.asyncio
@pytest.mark.usefixtures("workflow_id")
async def test_aput_includes_model_metadata_json_when_context_is_set(
    gitlab_workflow,
    http_client,
    checkpoint_data,
    checkpoint_metadata,
    model_metadata,
):
    """Test that aput() includes model_metadata_json in the HTTP POST payload when the context is set."""
    current_model_metadata_context.set(model_metadata)

    config = {"configurable": {"checkpoint_id": "parent-checkpoint"}}
    checkpoint = checkpoint_data[0]["checkpoint"]
    checkpoint["channel_values"]["status"] = WorkflowStatusEnum.COMPLETED

    http_client.apost.return_value = GitLabHttpResponse(status_code=200, body={})

    await gitlab_workflow.aput(
        config, checkpoint, checkpoint_metadata, ChannelVersions()
    )

    http_client.apost.assert_called_once()
    post_call_body = json.loads(http_client.apost.call_args[1]["body"])

    assert "model_metadata_json" in post_call_body
    assert post_call_body["model_metadata_json"] == model_metadata.model_dump_json(
        exclude={"llm_definition", "friendly_name"}
    )


@pytest.mark.asyncio
async def test_aput_omits_model_metadata_json_when_context_is_not_set(
    gitlab_workflow,
    http_client,
    checkpoint_data,
    checkpoint_metadata,
):
    """Test that aput() does not include model_metadata_json in the HTTP POST payload when the context is not set."""
    # Ensure context is not set (default)
    current_model_metadata_context.set(None)

    config = {"configurable": {"checkpoint_id": "parent-checkpoint"}}
    checkpoint = checkpoint_data[0]["checkpoint"]
    checkpoint["channel_values"]["status"] = WorkflowStatusEnum.COMPLETED

    http_client.apost.return_value = GitLabHttpResponse(status_code=200, body={})

    await gitlab_workflow.aput(
        config, checkpoint, checkpoint_metadata, ChannelVersions()
    )

    http_client.apost.assert_called_once()
    post_call_body = json.loads(http_client.apost.call_args[1]["body"])

    assert "model_metadata_json" not in post_call_body


@pytest.mark.usefixtures("checkpoint_data", "checkpoint_metadata")
def test_aput_with_no_status_update(
    http_client,
    workflow_id,
    workflow_type,
    workflow_config,
):
    workflow = GitLabWorkflow(
        http_client,
        workflow_id,
        workflow_type=workflow_type,
        workflow_config=workflow_config,
    )

    # no status update in writes
    writes: Sequence[tuple[str, Any]] = []
    status_event = workflow._get_workflow_status_event(writes)
    assert status_event is None

    # status update in writes but not "status" attribute
    writes = [("execution_handover", {"handover": []})]
    status_event = workflow._get_workflow_status_event(writes)
    assert status_event is None


def test_aput_with_noop_status_update(
    http_client,
    workflow_id,
    workflow_type,
    workflow_config,
):
    workflow = GitLabWorkflow(
        http_client,
        workflow_id,
        workflow_type=workflow_type,
        workflow_config=workflow_config,
    )

    # status update with noop status (approval_error)
    writes: Sequence[tuple[str, Any]] = [("status", WorkflowStatusEnum.APPROVAL_ERROR)]
    status_event = workflow._get_workflow_status_event(writes)
    assert status_event is None


def test_aput_with_no_status_update_and_human_input(
    http_client,
    workflow_id,
    workflow_type,
    workflow_config,
):
    workflow = GitLabWorkflow(
        client=http_client,
        workflow_id=workflow_id,
        workflow_type=workflow_type,
        workflow_config=workflow_config,
    )

    # no status update in writes
    writes: Sequence[tuple[str, Any]] = []
    status_event = workflow._get_workflow_status_event(writes)
    assert status_event is None

    # status update with non-matching attribute
    writes = [
        ("planning_check_human_input", {"last_human_input": {"event_type": "resume"}})
    ]
    status_event = workflow._get_workflow_status_event(writes)
    assert status_event is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,expected_event",
    [
        (WorkflowStatusEnum.COMPLETED, WorkflowStatusEventEnum.FINISH),
        (WorkflowStatusEnum.ERROR, WorkflowStatusEventEnum.DROP),
        (WorkflowStatusEnum.CANCELLED, WorkflowStatusEventEnum.STOP),
        (WorkflowStatusEnum.PAUSED, WorkflowStatusEventEnum.PAUSE),
        (
            WorkflowStatusEnum.PLAN_APPROVAL_REQUIRED,
            WorkflowStatusEventEnum.REQUIRE_PLAN_APPROVAL,
        ),
        (WorkflowStatusEnum.INPUT_REQUIRED, WorkflowStatusEventEnum.REQUIRE_INPUT),
        (
            WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED,
            WorkflowStatusEventEnum.REQUIRE_TOOL_CALL_APPROVAL,
        ),
    ],
)
async def test_workflow_status_events(
    gitlab_workflow,
    http_client,
    workflow_id,
    status,
    expected_event,
):
    config: RunnableConfig = {
        "configurable": {"checkpoint_id": "test-id", "thread_id": workflow_id}
    }
    writes: Sequence[tuple[str, Any]] = [("status", status)]

    http_client.apatch.return_value = GitLabHttpResponse(status_code=200, body={})

    await gitlab_workflow.aput_writes(config, writes, "task_id")

    http_client.apatch.assert_called_once_with(
        path=f"/api/v4/ai/duo_workflows/workflows/{workflow_id}",
        body=json.dumps({"status_event": expected_event.value}),
        parse_json=True,
    )
    http_client.apatch.reset_mock()


@pytest.mark.asyncio
async def test_track_workflow_completion_early_return(
    gitlab_workflow, internal_event_client: Mock
):
    gitlab_workflow._internal_event_client = internal_event_client

    # Test with a status that doesn't match any of the specific cases
    await gitlab_workflow._track_workflow_completion("some_other_status")

    # Verify no event was tracked
    internal_event_client.track_event.assert_not_called()


@pytest.mark.asyncio
@patch.dict("os.environ", {"USE_MEMSAVER": "true"}, clear=True)
async def test_offline_mode(http_client, workflow_id, workflow_type, workflow_config):
    gitlab_workflow = GitLabWorkflow(
        client=http_client,
        workflow_id=workflow_id,
        workflow_type=workflow_type,
        workflow_config=workflow_config,
    )

    async with gitlab_workflow as workflow:
        assert isinstance(workflow, MemorySaver)

    http_client.apost.assert_not_called()
    http_client.aget.assert_not_called()


@pytest.mark.asyncio
async def test_aput_writes_without_interrupt(gitlab_workflow, http_client):
    config = {"configurable": {"checkpoint_id": "id", "thread_id": "thread_id"}}
    writes: Sequence[tuple[str, Any]] = []

    await gitlab_workflow.aput_writes(config, writes, "task_id")

    http_client.apost.assert_not_called()


@pytest.mark.asyncio
async def test_aput_writes_with_interrupt(gitlab_workflow, http_client):
    config = {"configurable": {"checkpoint_id": "id", "thread_id": "123"}}
    writes = [("__interrupt__", "some value")]

    await gitlab_workflow.aput_writes(config, writes, "task_id")

    http_client.apost.assert_called_once_with(
        path="/api/v4/ai/duo_workflows/workflows/123/checkpoint_writes_batch",
        body=json.dumps(
            {
                "thread_ts": "id",
                "checkpoint_writes": [
                    {
                        "task": "task_id",
                        "channel": "__interrupt__",
                        "data": "qnNvbWUgdmFsdWU=",
                        "write_type": "msgpack",
                        "idx": 0,
                    }
                ],
            }
        ),
    )


@pytest.mark.asyncio
async def test_created_status_with_no_checkpoint_succeeds(
    http_client,
    workflow_id,
    workflow_type,
):
    workflow_config = {
        "first_checkpoint": None,
        "workflow_status": WorkflowStatusEnum.CREATED,
        "agent_privileges_names": ["read_repository"],
        "pre_approved_agent_privileges_names": [],
        "mcp_enabled": True,
        "allow_agent_to_request_user": True,
        "archived": False,
        "stalled": False,
    }

    gitlab_workflow = GitLabWorkflow(
        http_client,
        workflow_id,
        workflow_type,
        workflow_config,
    )

    config: RunnableConfig = {"configurable": {}}

    # This should succeed and return START event
    status_event, event_property = await gitlab_workflow._get_initial_status_event(
        config
    )

    assert status_event == WorkflowStatusEventEnum.START
    assert event_property == EventPropertyEnum.WORKFLOW_ID


@pytest.mark.asyncio
async def test_created_status_with_existing_checkpoint_raises_error(
    http_client,
    workflow_id,
    workflow_type,
):
    mock_checkpoint = {"checkpoint": "test_checkpoint_data"}
    workflow_config = {
        "first_checkpoint": mock_checkpoint,
        "workflow_status": WorkflowStatusEnum.CREATED,
        "agent_privileges_names": ["read_repository"],
        "pre_approved_agent_privileges_names": [],
        "mcp_enabled": True,
        "allow_agent_to_request_user": True,
        "archived": False,
        "stalled": False,
    }

    gitlab_workflow = GitLabWorkflow(
        http_client,
        workflow_id,
        workflow_type,
        workflow_config,
    )

    config: RunnableConfig = {"configurable": {}}

    with pytest.raises(UnsupportedStatusEvent) as exc_info:
        await gitlab_workflow._get_initial_status_event(config)

    assert "Workflow with status 'created' should not have existing checkpoints" in str(
        exc_info.value
    )
    assert f"Found checkpoint: {mock_checkpoint}" in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        WorkflowStatusEnum.FINISHED,
        WorkflowStatusEnum.STOPPED,
        WorkflowStatusEnum.INPUT_REQUIRED,
        WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED,
        WorkflowStatusEnum.PLAN_APPROVAL_REQUIRED,
    ],
)
async def test_track_workflow_completion_with_billing_event(
    gitlab_workflow,
    workflow_id,
    workflow_type,
    billing_event_service,
    billing_event_client,
    mock_user,
    mock_llm_operations,
    status,
):
    """Test that workflow completion triggers billing event for trackable statuses."""
    current_user.set(mock_user)
    gitlab_workflow._billing_event_service = billing_event_service

    await gitlab_workflow._track_workflow_completion(status)

    billing_event_client.track_billing_event.assert_called_once_with(
        mock_user,
        BillingEvent.DAP_FLOW_ON_COMPLETION,
        "GitLabWorkflow",
        unit_of_measure="request",
        quantity=1,
        metadata={
            "workflow_id": workflow_id,
            "feature_qualified_name": workflow_type.feature_qualified_name,
            "feature_ai_catalog_item": workflow_type.feature_ai_catalog_item,
            "execution_environment": ExecutionEnvironment.DAP.value,
            "llm_operations": mock_llm_operations,
            "tool_names": [],
            "orbit_called": False,
        },
    )


@pytest.mark.asyncio
async def test_track_workflow_completion_with_non_billable_status(
    gitlab_workflow,
    billing_event_service,
    billing_event_client,
):
    """Test that workflow completion doesn't trigger billing event for non-trackable statuses."""
    gitlab_workflow._billing_event_service = billing_event_service

    await gitlab_workflow._track_workflow_completion("some_other_status")

    billing_event_client.track_billing_event.assert_not_called()


@pytest.mark.asyncio
async def test_track_workflow_completion_with_orbit_called(
    gitlab_workflow,
    workflow_id,
    workflow_type,
    billing_event_service,
    billing_event_client,
    mock_user,
):
    """Test that orbit_called=True is passed to billing when orbit tools were used."""
    current_user.set(mock_user)
    gitlab_workflow._billing_event_service = billing_event_service
    gitlab_workflow._orbit_called = True

    operations = [
        {
            "token_count": 100,
            "model_id": "claude-3-sonnet",
            "model_engine": "anthropic",
            "model_provider": "anthropic",
            "prompt_tokens": 80,
            "completion_tokens": 20,
        },
    ]
    llm_operations.set(operations)

    await gitlab_workflow._track_workflow_completion(WorkflowStatusEnum.FINISHED)

    mock_llm_operations = [
        LLMOperation.model_validate(op).model_dump() for op in operations
    ]
    billing_event_client.track_billing_event.assert_called_once_with(
        mock_user,
        BillingEvent.DAP_FLOW_ON_COMPLETION,
        "GitLabWorkflow",
        unit_of_measure="request",
        quantity=1,
        metadata={
            "workflow_id": workflow_id,
            "feature_qualified_name": workflow_type.feature_qualified_name,
            "feature_ai_catalog_item": workflow_type.feature_ai_catalog_item,
            "execution_environment": ExecutionEnvironment.DAP.value,
            "llm_operations": mock_llm_operations,
            "tool_names": [],
            "orbit_called": True,
        },
    )


class TestGetOrbitToolCalls:
    def test_no_tool_calls(self):
        checkpoint = {"channel_values": {"ui_chat_log": []}}
        assert _get_orbit_tool_calls(checkpoint) is False

    def test_no_orbit_tools(self):
        checkpoint = {
            "channel_values": {
                "ui_chat_log": [
                    {"tool_info": {"name": "read_file", "args": {}}},
                    {"tool_info": {"name": "run_command", "args": {}}},
                ]
            }
        }
        assert _get_orbit_tool_calls(checkpoint) is False

    def test_with_orbit_tool(self):
        checkpoint = {
            "channel_values": {
                "ui_chat_log": [
                    {"tool_info": {"name": "read_file", "args": {}}},
                    {"tool_info": {"name": "orbit_search", "args": {}}},
                ]
            }
        }
        assert _get_orbit_tool_calls(checkpoint) is True

    def test_missing_channel_values(self):
        checkpoint = {}
        assert _get_orbit_tool_calls(checkpoint) is False

    def test_entries_without_tool_info(self):
        checkpoint = {
            "channel_values": {
                "ui_chat_log": [
                    {"tool_info": None},
                    {"message_type": "agent"},
                ]
            }
        }
        assert _get_orbit_tool_calls(checkpoint) is False


@pytest.mark.asyncio
@pytest.mark.usefixtures("workflow_id")
async def test_aput_sets_orbit_called_when_orbit_tool_present(
    gitlab_workflow, http_client, checkpoint_metadata
):
    """Test that aput sets _orbit_called when checkpoint contains orbit tool calls."""
    config = {"configurable": {"checkpoint_id": "parent-checkpoint"}}
    checkpoint = {
        "id": "new-id",
        "channel_values": {
            "ui_chat_log": [{"tool_info": {"name": "orbit_search", "args": {}}}]
        },
    }
    http_client.apost.return_value = GitLabHttpResponse(status_code=200, body={})

    assert gitlab_workflow._orbit_called is False
    await gitlab_workflow.aput(
        config, checkpoint, checkpoint_metadata, ChannelVersions()
    )
    assert gitlab_workflow._orbit_called is True


@pytest.mark.asyncio
@pytest.mark.usefixtures("workflow_id")
async def test_aput_does_not_reset_orbit_called_once_set(
    gitlab_workflow, http_client, checkpoint_metadata
):
    """Test that _orbit_called stays True once set, even if subsequent checkpoints have no orbit tools."""
    config = {"configurable": {"checkpoint_id": "parent-checkpoint"}}
    http_client.apost.return_value = GitLabHttpResponse(status_code=200, body={})

    gitlab_workflow._orbit_called = True
    checkpoint = {
        "id": "new-id",
        "channel_values": {
            "ui_chat_log": [{"tool_info": {"name": "read_file", "args": {}}}]
        },
    }
    await gitlab_workflow.aput(
        config, checkpoint, checkpoint_metadata, ChannelVersions()
    )
    assert gitlab_workflow._orbit_called is True


@pytest.mark.asyncio
@pytest.mark.usefixtures("workflow_id", "workflow_type")
async def test_track_workflow_completion_includes_duration_seconds(
    gitlab_workflow,
    internal_event_client: Mock,
):
    """Test that completion events include duration_seconds when _flow_start_time is set."""
    gitlab_workflow._internal_event_client = internal_event_client
    gitlab_workflow._billing_event_service = Mock()
    gitlab_workflow._flow_start_time = 1000.0

    with patch("duo_workflow_service.checkpointer.gitlab_workflow.time") as mock_time:
        mock_time.time.return_value = 1005.5
        await gitlab_workflow._track_workflow_completion("finished")

    internal_event_client.track_event.assert_called_once()
    call_kwargs = internal_event_client.track_event.call_args[1]
    additional_props = call_kwargs["additional_properties"]

    assert additional_props.extra["duration_seconds"] == 5.5
    assert additional_props.label == EventLabelEnum.WORKFLOW_FINISH_LABEL.value
    assert call_kwargs["event_name"] == EventEnum.WORKFLOW_FINISH_SUCCESS.value


@pytest.mark.asyncio
async def test_track_workflow_completion_without_start_time(
    gitlab_workflow,
    internal_event_client: Mock,
):
    """Test that completion events omit duration_seconds when _flow_start_time is not set."""
    gitlab_workflow._internal_event_client = internal_event_client
    gitlab_workflow._billing_event_service = Mock()

    await gitlab_workflow._track_workflow_completion("finished")

    internal_event_client.track_event.assert_called_once()
    call_kwargs = internal_event_client.track_event.call_args[1]
    additional_props = call_kwargs["additional_properties"]

    assert "duration_seconds" not in additional_props.extra


@pytest.mark.asyncio
async def test_aget_tuple_with_latest_checkpoint(
    http_client,
    workflow_id,
    workflow_type,
    checkpoint_data,
):
    """Test aget_tuple uses latestCheckpoint when available."""
    latest_checkpoint = {
        "threadTs": "latest-id",
        "parentTs": None,
        "checkpoint": json.dumps(checkpoint_data[0]["checkpoint"], cls=CustomEncoder),
        "metadata": json.dumps(checkpoint_data[0]["metadata"], cls=CustomEncoder),
    }

    workflow_config = {
        "first_checkpoint": None,
        "latest_checkpoint": latest_checkpoint,
        "workflow_status": "created",
        "agent_privileges_names": ["read_repository"],
        "pre_approved_agent_privileges_names": [],
        "mcp_enabled": True,
        "allow_agent_to_request_user": True,
        "archived": False,
        "stalled": False,
    }

    gitlab_workflow = GitLabWorkflow(
        http_client,
        workflow_id,
        workflow_type,
        workflow_config,
    )

    config: CustomRunnableConfig = {"configurable": {"thread_id": workflow_id}}

    result = await gitlab_workflow.aget_tuple(config)

    assert result is not None
    assert isinstance(result, CheckpointTuple)
    assert result.checkpoint == checkpoint_data[0]["checkpoint"]
    assert result.metadata == checkpoint_data[0]["metadata"]

    # Should not call the API when latestCheckpoint is available
    http_client.aget.assert_not_called()


@pytest.mark.asyncio
async def test_aget_tuple_with_compressed_latest_checkpoint(
    http_client,
    workflow_id,
    workflow_type,
    checkpoint_data,
):
    """Test aget_tuple decompresses compressedCheckpoint from GraphQL latestCheckpoint (19.0+)."""
    compressed = compress_checkpoint(checkpoint_data[0]["checkpoint"])
    latest_checkpoint = {
        "threadTs": "latest-id",
        "parentTs": None,
        "compressedCheckpoint": compressed,
        "metadata": json.dumps(checkpoint_data[0]["metadata"], cls=CustomEncoder),
    }

    workflow_config = {
        "first_checkpoint": None,
        "latest_checkpoint": latest_checkpoint,
        "workflow_status": "created",
        "agent_privileges_names": ["read_repository"],
        "pre_approved_agent_privileges_names": [],
        "mcp_enabled": True,
        "allow_agent_to_request_user": True,
        "archived": False,
        "stalled": False,
    }

    gitlab_workflow = GitLabWorkflow(
        http_client,
        workflow_id,
        workflow_type,
        workflow_config,
    )

    config: CustomRunnableConfig = {"configurable": {"thread_id": workflow_id}}

    result = await gitlab_workflow.aget_tuple(config)

    assert result is not None
    assert isinstance(result, CheckpointTuple)
    assert result.checkpoint == checkpoint_data[0]["checkpoint"]
    assert result.metadata == checkpoint_data[0]["metadata"]

    # Should not call the API when latestCheckpoint is available
    http_client.aget.assert_not_called()


@pytest.mark.asyncio
async def test_aget_tuple_returns_none_when_no_first_checkpoint_and_no_latest_checkpoint(
    http_client,
    workflow_id,
    workflow_type,
):
    """Test aget_tuple returns None when first_checkpoint is None and latest_checkpoint is not set."""
    workflow_config = {
        "first_checkpoint": None,
        "latest_checkpoint": None,
        "workflow_status": "created",
        "agent_privileges_names": ["read_repository"],
        "pre_approved_agent_privileges_names": [],
        "mcp_enabled": True,
        "allow_agent_to_request_user": True,
        "archived": False,
        "stalled": False,
    }

    gitlab_workflow = GitLabWorkflow(
        http_client,
        workflow_id,
        workflow_type,
        workflow_config,
    )

    config: CustomRunnableConfig = {"configurable": {"thread_id": workflow_id}}

    result = await gitlab_workflow.aget_tuple(config)

    assert result is None


@pytest.mark.asyncio
async def test_archived_workflow_raises_error(
    http_client,
    workflow_id,
    workflow_type,
):
    """Test that archived workflows raise NotifiableException."""
    workflow_config = {
        "first_checkpoint": None,
        "latest_checkpoint": None,
        "workflow_status": "created",
        "agent_privileges_names": ["read_repository"],
        "pre_approved_agent_privileges_names": [],
        "mcp_enabled": True,
        "allow_agent_to_request_user": True,
        "archived": True,
        "stalled": False,
    }

    gitlab_workflow = GitLabWorkflow(
        http_client,
        workflow_id,
        workflow_type,
        workflow_config,
    )

    config: CustomRunnableConfig = {"configurable": {"thread_id": workflow_id}}

    with pytest.raises(NotifiableException) as exc_info:
        await gitlab_workflow._get_initial_status_event(config)

    assert (
        "Archived workflow can not be executed. Please create a new workflow."
        in str(exc_info.value)
    )


@pytest.mark.asyncio
async def test_stalled_workflow_raises_error(
    http_client,
    workflow_id,
    workflow_type,
):
    """Test that stalled workflows raise NotifiableException."""
    workflow_config = {
        "first_checkpoint": None,
        "latest_checkpoint": None,
        "workflow_status": "created",
        "agent_privileges_names": ["read_repository"],
        "pre_approved_agent_privileges_names": [],
        "mcp_enabled": True,
        "allow_agent_to_request_user": True,
        "archived": False,
        "stalled": True,
    }

    gitlab_workflow = GitLabWorkflow(
        http_client,
        workflow_id,
        workflow_type,
        workflow_config,
    )

    config: CustomRunnableConfig = {"configurable": {"thread_id": workflow_id}}

    with pytest.raises(NotifiableException) as exc_info:
        await gitlab_workflow._get_initial_status_event(config)

    assert "Stalled workflow can not be executed. Please create a new workflow." in str(
        exc_info.value
    )


@pytest.mark.asyncio
async def test_track_workflow_completion_includes_response_schema_tracking(
    gitlab_workflow,
    internal_event_client: Mock,
):
    """Test that completion event includes response schema tracking data from ContextVar."""
    from duo_workflow_service.tracking.response_schema_tracking_context import (
        response_schema_tracking_results,
    )

    gitlab_workflow._internal_event_client = internal_event_client
    gitlab_workflow._billing_event_client = Mock()

    token = response_schema_tracking_results.set(
        {
            "fix_pipeline_decide_approach": {
                "failure_category": "test_failure",
                "suggested_fix_type": "code_fix",
            }
        }
    )
    try:
        await gitlab_workflow._track_workflow_completion("finished")
    finally:
        response_schema_tracking_results.reset(token)

    internal_event_client.track_event.assert_called_once()
    call_kwargs = internal_event_client.track_event.call_args[1]
    additional_props = call_kwargs["additional_properties"]

    assert additional_props.extra["fix_pipeline_decide_approach_output"] == json.dumps(
        {"failure_category": "test_failure", "suggested_fix_type": "code_fix"}
    )


@pytest.mark.asyncio
async def test_track_workflow_completion_omits_tracking_when_absent(
    gitlab_workflow,
    internal_event_client: Mock,
):
    """Test that completion event has no tracking data when no components tracked."""
    from duo_workflow_service.tracking.response_schema_tracking_context import (
        response_schema_tracking_results,
    )

    gitlab_workflow._internal_event_client = internal_event_client
    gitlab_workflow._billing_event_client = Mock()

    token = response_schema_tracking_results.set({})
    try:
        await gitlab_workflow._track_workflow_completion("finished")
    finally:
        response_schema_tracking_results.reset(token)

    internal_event_client.track_event.assert_called_once()
    call_kwargs = internal_event_client.track_event.call_args[1]
    additional_props = call_kwargs["additional_properties"]

    assert "fix_pipeline_decide_approach_output" not in additional_props.extra


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        WorkflowStatusEnum.FINISHED,
        WorkflowStatusEnum.STOPPED,
        WorkflowStatusEnum.INPUT_REQUIRED,
        WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED,
        WorkflowStatusEnum.PLAN_APPROVAL_REQUIRED,
    ],
)
async def test_track_workflow_completion_with_billing_event_includes_tool_names(
    gitlab_workflow,
    workflow_id,
    workflow_type,
    billing_event_service,
    billing_event_client,
    mock_user,
    mock_llm_operations,
    status,
):
    """Test that tool names accumulated during workflow are passed to billing event."""
    current_user.set(mock_user)
    gitlab_workflow._billing_event_service = billing_event_service

    # Simulate tools that were executed during the workflow
    init_tool_executions()
    tool_executions.get().append("read_file")
    tool_executions.get().append("write_file")
    tool_executions.get().append("read_file")

    await gitlab_workflow._track_workflow_completion(status)

    billing_event_client.track_billing_event.assert_called_once_with(
        mock_user,
        BillingEvent.DAP_FLOW_ON_COMPLETION,
        "GitLabWorkflow",
        unit_of_measure="request",
        quantity=1,
        metadata={
            "feature_qualified_name": workflow_type.feature_qualified_name,
            "feature_ai_catalog_item": workflow_type.feature_ai_catalog_item,
            "execution_environment": ExecutionEnvironment.DAP.value,
            "llm_operations": mock_llm_operations,
            "tool_names": ["read_file", "write_file", "read_file"],
            "orbit_called": False,
            "workflow_id": workflow_id,
        },
    )


@pytest.mark.asyncio
async def test_track_workflow_completion_fires_orbit_session_summary(
    gitlab_workflow,
    internal_event_client,
    workflow_id,
    workflow_type,
):
    """Test that orbit_dap_session_summary fires when orbit tools were used."""
    from lib.context.orbit import orbit_tool_call_count, total_tool_call_count

    gitlab_workflow._internal_event_client = internal_event_client

    orbit_tool_call_count.set(3)
    total_tool_call_count.set(7)

    await gitlab_workflow._track_workflow_completion("finished")

    assert internal_event_client.track_event.call_count == 2

    orbit_calls = [
        c
        for c in internal_event_client.track_event.call_args_list
        if c[1]["event_name"] == EventEnum.ORBIT_DAP_SESSION_SUMMARY.value
    ]
    assert len(orbit_calls) == 1
    orbit_call = orbit_calls[0]

    additional_props = orbit_call[1]["additional_properties"]
    assert additional_props.value == workflow_id
    assert additional_props.extra["workflow_type"] == workflow_type.value
    assert additional_props.extra["orbit_calls_count"] == 3
    assert additional_props.extra["non_orbit_tool_calls"] == 4
    assert additional_props.extra["total_tool_calls"] == 7


@pytest.mark.asyncio
async def test_track_workflow_completion_skips_orbit_summary_when_no_orbit_calls(
    gitlab_workflow,
    internal_event_client,
):
    """Test that orbit_dap_session_summary does NOT fire when no orbit tools were used."""
    from lib.context.orbit import orbit_tool_call_count, total_tool_call_count

    gitlab_workflow._internal_event_client = internal_event_client

    orbit_tool_call_count.set(0)
    total_tool_call_count.set(5)

    await gitlab_workflow._track_workflow_completion("finished")

    assert internal_event_client.track_event.call_count == 1
    call_args = internal_event_client.track_event.call_args
    assert call_args[1]["event_name"] != EventEnum.ORBIT_DAP_SESSION_SUMMARY.value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pause_status",
    [WorkflowStatusEnum.INPUT_REQUIRED, WorkflowStatusEnum.PLAN_APPROVAL_REQUIRED],
)
async def test_track_workflow_completion_skips_orbit_summary_on_pause(
    gitlab_workflow,
    internal_event_client,
    pause_status,
):
    """Orbit summary must not fire on pause statuses — counters reset on each resume, so firing here would produce
    partial summaries instead of one session-level summary."""
    from lib.context.orbit import orbit_tool_call_count, total_tool_call_count

    gitlab_workflow._internal_event_client = internal_event_client

    orbit_tool_call_count.set(2)
    total_tool_call_count.set(4)

    await gitlab_workflow._track_workflow_completion(pause_status)

    event_names = [
        c[1]["event_name"] for c in internal_event_client.track_event.call_args_list
    ]
    assert EventEnum.ORBIT_DAP_SESSION_SUMMARY.value not in event_names


# ---------------------------------------------------------------------------
# Incremental checkpoint tests (phase 1 shadow writes — gitlab#596714)
# ---------------------------------------------------------------------------


def test_dict_of_list_delta_appends_only():
    prev = {"a": ["x"], "b": ["y", "z"]}
    current = {"a": ["x", "x2"], "b": ["y", "z"], "c": ["new"]}
    delta = _dict_of_list_delta(prev, current)
    assert delta is not None
    assert delta.values == {"a": ["x2"], "c": ["new"]}
    assert delta.is_append is True


def test_dict_of_list_delta_shrink_stores_full():
    prev = {"a": ["x", "y"]}
    current = {"a": ["x"]}
    delta = _dict_of_list_delta(prev, current)
    assert delta is not None
    assert delta.values == {"a": ["x"]}
    assert delta.is_append is False


def test_dict_of_list_delta_non_list_always_stored():
    prev = {"a": "old"}
    current = {"a": "new", "b": 42}
    delta = _dict_of_list_delta(prev, current)
    assert delta is not None
    assert delta.values == {"a": "new", "b": 42}
    assert delta.is_append is True


def test_dict_of_list_delta_unchanged_returns_none():
    prev = {"a": ["x", "y"]}
    current = {"a": ["x", "y"]}
    assert _dict_of_list_delta(prev, current) is None


def test_serialize_channel_blobs_only_changed_channels():
    import base64

    from duo_workflow_service.checkpointer.utils.serializer import CheckpointSerializer

    serde = CheckpointSerializer()
    checkpoint = {
        "id": "ckpt1",
        "channel_values": {
            "messages": ["a", "b"],
            "status": "running",
        },
    }
    new_versions = ChannelVersions({"messages": "2.0"})

    blobs, _ = _serialize_channel_blobs(checkpoint, new_versions, serde, {})

    assert len(blobs) == 1
    assert blobs[0]["channel"] == "messages"
    assert blobs[0]["version"] == "2.0"
    assert blobs[0]["write_type"] == "msgpack"
    assert base64.b64decode(blobs[0]["data"])


def test_serialize_channel_blobs_skips_scalar_channels():
    from duo_workflow_service.checkpointer.utils.serializer import CheckpointSerializer

    serde = CheckpointSerializer()
    checkpoint = {
        "id": "ckpt1",
        "channel_values": {
            "messages": ["a", "b"],
            "status": "running",
            "goal": "fix the bug",
        },
    }
    new_versions = ChannelVersions({"messages": "2.0", "status": "1.0", "goal": "1.0"})

    blobs, _ = _serialize_channel_blobs(checkpoint, new_versions, serde, {})

    channels = [b["channel"] for b in blobs]
    assert "status" not in channels
    assert "goal" not in channels
    assert "messages" in channels


def test_serialize_channel_blobs_list_delta():
    import base64

    from duo_workflow_service.checkpointer.utils.serializer import CheckpointSerializer

    serde = CheckpointSerializer()
    checkpoint = {
        "id": "ckpt2",
        "channel_values": {"messages": ["a", "b", "c"]},
    }
    new_versions = ChannelVersions({"messages": "3.0"})
    prev_channel_values = {"messages": ["a", "b"]}

    blobs, _ = _serialize_channel_blobs(
        checkpoint, new_versions, serde, prev_channel_values
    )

    assert len(blobs) == 1
    val = serde.loads_typed(
        (blobs[0]["write_type"], zlib.decompress(base64.b64decode(blobs[0]["data"])))
    )
    assert val == ["c"]


def test_serialize_channel_blobs_skips_unknown_channels():
    from duo_workflow_service.checkpointer.utils.serializer import CheckpointSerializer

    serde = CheckpointSerializer()
    checkpoint = {"id": "ckpt3", "channel_values": {}}
    new_versions = ChannelVersions({"nonexistent": "1.0"})

    blobs, _ = _serialize_channel_blobs(checkpoint, new_versions, serde, {})

    assert not blobs


def test_serialize_channel_blobs_list_unchanged_skips():
    from duo_workflow_service.checkpointer.utils.serializer import CheckpointSerializer

    serde = CheckpointSerializer()
    checkpoint = {
        "id": "ckpt_unchanged",
        "channel_values": {"messages": ["a", "b"]},
    }
    new_versions = ChannelVersions({"messages": "2.0"})
    prev_channel_values = {"messages": ["a", "b"]}

    blobs, is_compaction = _serialize_channel_blobs(
        checkpoint, new_versions, serde, prev_channel_values
    )

    assert not blobs
    assert not is_compaction


def test_serialize_channel_blobs_list_shrink_stores_full():
    import base64

    from duo_workflow_service.checkpointer.utils.serializer import CheckpointSerializer

    serde = CheckpointSerializer()
    checkpoint = {
        "id": "ckpt5",
        "channel_values": {"messages": ["a"]},
    }
    new_versions = ChannelVersions({"messages": "4.0"})
    prev_channel_values = {"messages": ["a", "b", "c"]}

    blobs, is_compaction = _serialize_channel_blobs(
        checkpoint, new_versions, serde, prev_channel_values
    )

    assert is_compaction
    assert len(blobs) == 1
    val = serde.loads_typed(
        (blobs[0]["write_type"], zlib.decompress(base64.b64decode(blobs[0]["data"])))
    )
    assert val == ["a"]


def test_serialize_channel_blobs_dict_channel_delta():
    import base64

    from duo_workflow_service.checkpointer.utils.serializer import CheckpointSerializer

    serde = CheckpointSerializer()
    checkpoint = {
        "id": "ckpt4",
        "channel_values": {
            "conversation_history": {
                "planner": ["msg1", "msg2", "msg3"],
                "executor": ["a"],
            }
        },
    }
    new_versions = ChannelVersions({"conversation_history": "2.0"})
    prev_channel_values = {
        "conversation_history": {
            "planner": ["msg1", "msg2"],
            "executor": ["a"],
        }
    }

    blobs, _ = _serialize_channel_blobs(
        checkpoint, new_versions, serde, prev_channel_values
    )

    assert len(blobs) == 1
    assert blobs[0]["channel"] == "conversation_history"
    delta = serde.loads_typed(
        (blobs[0]["write_type"], zlib.decompress(base64.b64decode(blobs[0]["data"])))
    )
    assert delta == {"planner": ["msg3"]}


def test_serialize_channel_blobs_dict_unchanged_skips_blob():
    from duo_workflow_service.checkpointer.utils.serializer import CheckpointSerializer

    serde = CheckpointSerializer()
    values = {"conversation_history": {"planner": ["msg1"], "executor": ["a"]}}
    checkpoint = {"id": "ckpt6", "channel_values": values}
    new_versions = ChannelVersions({"conversation_history": "2.0"})

    blobs, _ = _serialize_channel_blobs(checkpoint, new_versions, serde, dict(values))

    assert not blobs


def test_serialize_channel_blobs_compaction_stores_full_value():
    import base64

    from duo_workflow_service.checkpointer.utils.serializer import CheckpointSerializer

    serde = CheckpointSerializer()
    prev_channel_values = {
        "conversation_history": {"planner": ["msg1", "msg2", "msg3", "msg4", "msg5"]}
    }
    checkpoint = {
        "id": "ckpt7",
        "channel_values": {"conversation_history": {"planner": ["summary", "msg5"]}},
    }
    new_versions = ChannelVersions({"conversation_history": "3.0"})

    blobs, is_compaction = _serialize_channel_blobs(
        checkpoint, new_versions, serde, prev_channel_values
    )

    assert is_compaction
    assert len(blobs) == 1
    assert blobs[0]["step_action"] == "compaction"
    val = serde.loads_typed(
        (blobs[0]["write_type"], zlib.decompress(base64.b64decode(blobs[0]["data"])))
    )
    assert val == {"planner": ["summary", "msg5"]}


def test_serialize_channel_blobs_dict_same_length_rewrite_is_compaction():
    from duo_workflow_service.checkpointer.utils.serializer import CheckpointSerializer

    serde = CheckpointSerializer()
    prev_channel_values = {"conversation_history": {"planner": ["msg1", "msg2"]}}
    checkpoint = {
        "id": "ckpt9",
        "channel_values": {
            "conversation_history": {"planner": ["summary_a", "summary_b"]}
        },
    }
    new_versions = ChannelVersions({"conversation_history": "3.0"})

    blobs, is_compaction = _serialize_channel_blobs(
        checkpoint, new_versions, serde, prev_channel_values
    )

    assert is_compaction
    assert len(blobs) == 1
    assert blobs[0]["step_action"] == "compaction"


def test_serialize_channel_blobs_force_rewrite_bypasses_delta():
    import base64

    from duo_workflow_service.checkpointer.utils.serializer import CheckpointSerializer

    serde = CheckpointSerializer()
    checkpoint = {
        "id": "ckpt_force",
        "channel_values": {"messages": ["a", "b", "c"]},
    }
    new_versions = ChannelVersions({"messages": "2.0"})
    prev_channel_values = {"messages": ["a", "b"]}

    blobs, is_compaction = _serialize_channel_blobs(
        checkpoint,
        new_versions,
        serde,
        prev_channel_values,
        force_rewrite=True,
    )

    assert not is_compaction
    assert len(blobs) == 1
    assert blobs[0]["step_action"] == "compaction"
    val = serde.loads_typed(
        (blobs[0]["write_type"], zlib.decompress(base64.b64decode(blobs[0]["data"])))
    )
    assert val == ["a", "b", "c"]


def test_serialize_channel_blobs_conversation_sends_delta():
    import base64

    from duo_workflow_service.checkpointer.utils.serializer import CheckpointSerializer

    serde = CheckpointSerializer()
    checkpoint = {
        "id": "ckpt8",
        "channel_values": {"messages": ["a", "b", "c"]},
    }
    new_versions = ChannelVersions({"messages": "3.0"})

    blobs, is_compaction = _serialize_channel_blobs(
        checkpoint, new_versions, serde, {"messages": ["a", "b"]}
    )

    assert not is_compaction
    assert len(blobs) == 1
    assert blobs[0]["step_action"] == "conversation"
    val = serde.loads_typed(
        (blobs[0]["write_type"], zlib.decompress(base64.b64decode(blobs[0]["data"])))
    )
    assert val == ["c"]


@pytest.mark.asyncio
@patch(
    "duo_workflow_service.checkpointer.gitlab_workflow.is_client_capable",
    return_value=True,
)
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_aput_sends_full_checkpoint_and_channel_blobs(
    _mock_duo_workflow_metrics,
    _mock_is_client_capable,
    gitlab_workflow,
    http_client,
    checkpoint_data,
    checkpoint_metadata,
):
    """Phase 1 (shadow): aput sends the full compressed checkpoint alongside channel_blobs.

    Rails stores both; reads still use the embedded channel_values.
    """
    config = {"configurable": {"checkpoint_id": "parent-checkpoint"}}
    checkpoint = checkpoint_data[0]["checkpoint"]
    checkpoint["channel_values"]["messages"] = ["msg1", "msg2"]
    checkpoint["channel_values"]["status"] = WorkflowStatusEnum.EXECUTION

    new_versions = ChannelVersions({"messages": "2.1"})
    http_client.apost.return_value = GitLabHttpResponse(status_code=200, body={})

    await gitlab_workflow.aput(config, checkpoint, checkpoint_metadata, new_versions)

    post_call_body = json.loads(http_client.apost.call_args[1]["body"])

    # Full checkpoint must still be present (Phase 1 — backward compatible reads)
    assert post_call_body["compressed_checkpoint"] == compress_checkpoint(checkpoint)

    blobs = post_call_body["channel_blobs"]
    assert len(blobs) == 1
    assert blobs[0]["channel"] == "messages"
    assert blobs[0]["version"] == "2.1"


@pytest.mark.asyncio
@patch(
    "duo_workflow_service.checkpointer.gitlab_workflow.is_client_capable",
    return_value=True,
)
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_aput_accumulates_list_deltas_across_calls(
    _mock_duo_workflow_metrics,
    _mock_is_client_capable,
    gitlab_workflow,
    http_client,
    checkpoint_data,
    checkpoint_metadata,
):
    """Aput tracks previous channel values between calls so each blob contains only the items appended since the
    previous checkpoint."""
    import base64

    from duo_workflow_service.checkpointer.utils.serializer import CheckpointSerializer

    serde = CheckpointSerializer()
    http_client.apost.return_value = GitLabHttpResponse(status_code=200, body={})

    checkpoint = checkpoint_data[0]["checkpoint"]

    checkpoint["id"] = "ckpt-1"
    checkpoint["channel_values"]["messages"] = ["a", "b"]
    await gitlab_workflow.aput(
        {"configurable": {"checkpoint_id": None}},
        checkpoint,
        checkpoint_metadata,
        ChannelVersions({"messages": "1.0"}),
    )
    body1 = json.loads(http_client.apost.call_args[1]["body"])
    val1 = serde.loads_typed(
        (
            body1["channel_blobs"][0]["write_type"],
            zlib.decompress(base64.b64decode(body1["channel_blobs"][0]["data"])),
        )
    )
    assert val1 == ["a", "b"]

    checkpoint["id"] = "ckpt-2"
    checkpoint["channel_values"]["messages"] = ["a", "b", "c", "d"]
    await gitlab_workflow.aput(
        {"configurable": {"checkpoint_id": "ckpt-1"}},
        checkpoint,
        checkpoint_metadata,
        ChannelVersions({"messages": "2.0"}),
    )
    body2 = json.loads(http_client.apost.call_args[1]["body"])
    val2 = serde.loads_typed(
        (
            body2["channel_blobs"][0]["write_type"],
            zlib.decompress(base64.b64decode(body2["channel_blobs"][0]["data"])),
        )
    )
    assert val2 == ["c", "d"]


@pytest.mark.asyncio
@patch(
    "duo_workflow_service.checkpointer.gitlab_workflow.is_client_capable",
    return_value=True,
)
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_aput_increments_thread_id_on_compaction(
    _mock_duo_workflow_metrics,
    _mock_is_client_capable,
    gitlab_workflow,
    http_client,
    checkpoint_data,
    checkpoint_metadata,
):
    """thread_id stays 0 for conversation steps and increments to 1 at a compaction."""
    checkpoint = checkpoint_data[0]["checkpoint"]
    http_client.apost.return_value = GitLabHttpResponse(status_code=200, body={})

    # Normal conversation step — thread_id stays 0
    checkpoint["id"] = "ckpt-1"
    checkpoint["channel_values"]["messages"] = ["a", "b"]
    await gitlab_workflow.aput(
        {"configurable": {"checkpoint_id": None}},
        checkpoint,
        checkpoint_metadata,
        ChannelVersions({"messages": "1.0"}),
    )
    assert json.loads(http_client.apost.call_args[1]["body"])["current_thread"] == 0

    # Compaction step: list shrank — thread_id increments to 1
    checkpoint["id"] = "ckpt-2"
    checkpoint["channel_values"]["messages"] = ["summary"]
    await gitlab_workflow.aput(
        {"configurable": {"checkpoint_id": "ckpt-1"}},
        checkpoint,
        checkpoint_metadata,
        ChannelVersions({"messages": "2.0"}),
    )
    assert json.loads(http_client.apost.call_args[1]["body"])["current_thread"] == 1

    # Subsequent conversation step — thread_id stays 1
    checkpoint["id"] = "ckpt-3"
    checkpoint["channel_values"]["messages"] = ["summary", "new_msg"]
    await gitlab_workflow.aput(
        {"configurable": {"checkpoint_id": "ckpt-2"}},
        checkpoint,
        checkpoint_metadata,
        ChannelVersions({"messages": "3.0"}),
    )
    assert json.loads(http_client.apost.call_args[1]["body"])["current_thread"] == 1


@pytest.mark.asyncio
@patch(
    "duo_workflow_service.checkpointer.gitlab_workflow.is_client_capable",
    return_value=True,
)
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_aput_resets_cache_on_stale_checkpoint_id(
    _mock_duo_workflow_metrics,
    _mock_is_client_capable,
    gitlab_workflow,
    http_client,
    checkpoint_data,
    checkpoint_metadata,
):
    """When parent checkpoint_id doesn't match the cached id, cache is reset and a warning is logged."""
    import base64

    from duo_workflow_service.checkpointer.utils.serializer import CheckpointSerializer

    serde = CheckpointSerializer()
    checkpoint = checkpoint_data[0]["checkpoint"]
    http_client.apost.return_value = GitLabHttpResponse(status_code=200, body={})

    # First call — seeds the cache with ckpt-1 as prev
    checkpoint["id"] = "ckpt-1"
    checkpoint["channel_values"]["messages"] = ["a", "b"]
    await gitlab_workflow.aput(
        {"configurable": {"checkpoint_id": None}},
        checkpoint,
        checkpoint_metadata,
        ChannelVersions({"messages": "1.0"}),
    )

    # Second call — skips ckpt-1 (simulates a missed checkpoint) by passing a wrong parent
    checkpoint["id"] = "ckpt-3"
    checkpoint["channel_values"]["messages"] = ["a", "b", "c"]
    await gitlab_workflow.aput(
        {"configurable": {"checkpoint_id": "ckpt-unknown"}},
        checkpoint,
        checkpoint_metadata,
        ChannelVersions({"messages": "2.0"}),
    )

    # Cache was reset to {} so full list is stored (no delta), not just ["c"]
    body = json.loads(http_client.apost.call_args[1]["body"])
    val = serde.loads_typed(
        (
            body["channel_blobs"][0]["write_type"],
            zlib.decompress(base64.b64decode(body["channel_blobs"][0]["data"])),
        )
    )
    assert val == ["a", "b", "c"]
    # Thread must be bumped so Rails starts reconstruction from this checkpoint
    assert body["current_thread"] == 1


@pytest.mark.asyncio
@patch(
    "duo_workflow_service.checkpointer.gitlab_workflow.is_client_capable",
    return_value=False,
)
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_aput_omits_channel_blobs_when_not_capable(
    _mock_duo_workflow_metrics,
    _mock_is_client_capable,
    gitlab_workflow,
    http_client,
    checkpoint_data,
    checkpoint_metadata,
):
    """Rails instances that don't declare incremental_checkpoints must not receive channel_blobs."""
    config = {"configurable": {"checkpoint_id": "parent-checkpoint"}}
    checkpoint = checkpoint_data[0]["checkpoint"]
    checkpoint["channel_values"]["messages"] = ["msg1", "msg2"]

    http_client.apost.return_value = GitLabHttpResponse(status_code=200, body={})

    await gitlab_workflow.aput(
        config, checkpoint, checkpoint_metadata, ChannelVersions({"messages": "1.0"})
    )

    post_call_body = json.loads(http_client.apost.call_args[1]["body"])
    assert "compressed_checkpoint" in post_call_body
    assert "channel_blobs" not in post_call_body


@pytest.mark.asyncio
@patch(
    "duo_workflow_service.checkpointer.gitlab_workflow.is_client_capable",
    return_value=True,
)
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_aget_tuple_hydrates_current_thread_from_response(
    _mock_duo_workflow_metrics,
    _mock_is_client_capable,
    gitlab_workflow,
    http_client,
    compressed_checkpoint_data,
):
    """On resume, current_thread must be restored from the server so subsequent aput emits the same thread."""
    compressed_checkpoint_data[0]["current_thread"] = 3
    http_client.aget.return_value = GitLabHttpResponse(
        status_code=200, body=compressed_checkpoint_data
    )

    config = {"configurable": {"thread_id": "1234", "checkpoint_id": "5678"}}
    result = await gitlab_workflow.aget_tuple(config)

    assert result is not None
    assert gitlab_workflow._current_thread == 3
    assert gitlab_workflow._prev_checkpoint_id == "5678"
    assert "conversation_history" in gitlab_workflow._prev_channel_values


@pytest.mark.asyncio
@patch(
    "duo_workflow_service.checkpointer.gitlab_workflow.is_client_capable",
    return_value=True,
)
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_aget_tuple_hydrates_current_thread_on_latest_fetch_path(
    _mock_duo_workflow_metrics,
    _mock_is_client_capable,
    http_client,
    workflow_id,
    workflow_type,
    compressed_checkpoint_data,
):
    """Hydration must also fire on the latest-fetch path (no checkpoint_id; first_checkpoint set)."""
    workflow_config = {
        "first_checkpoint": {},
        "latest_checkpoint": None,
        "workflow_status": "created",
        "agent_privileges_names": ["read_repository"],
        "pre_approved_agent_privileges_names": [],
        "mcp_enabled": True,
        "allow_agent_to_request_user": True,
        "archived": False,
        "stalled": False,
    }
    gitlab_workflow = GitLabWorkflow(
        http_client,
        workflow_id,
        workflow_type,
        workflow_config,
    )

    compressed_checkpoint_data[0]["current_thread"] = 7
    http_client.aget.return_value = GitLabHttpResponse(
        status_code=200, body=compressed_checkpoint_data
    )

    config = {"configurable": {"thread_id": workflow_id}}
    result = await gitlab_workflow.aget_tuple(config)

    assert result is not None
    assert "per_page=1" in http_client.aget.call_args[1]["path"]
    assert gitlab_workflow._current_thread == 7
    assert gitlab_workflow._prev_checkpoint_id == "5678"
    assert "conversation_history" in gitlab_workflow._prev_channel_values


@pytest.mark.asyncio
@patch(
    "duo_workflow_service.checkpointer.gitlab_workflow.is_client_capable",
    return_value=True,
)
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_aget_tuple_hydration_tolerates_missing_current_thread(
    _mock_duo_workflow_metrics,
    _mock_is_client_capable,
    gitlab_workflow,
    http_client,
    compressed_checkpoint_data,
):
    """Older Rails versions don't return current_thread; hydration must still seed prev_* without raising."""
    assert "current_thread" not in compressed_checkpoint_data[0]
    http_client.aget.return_value = GitLabHttpResponse(
        status_code=200, body=compressed_checkpoint_data
    )

    config = {"configurable": {"thread_id": "1234", "checkpoint_id": "5678"}}
    await gitlab_workflow.aget_tuple(config)

    assert gitlab_workflow._current_thread == 0
    assert gitlab_workflow._prev_checkpoint_id == "5678"


@pytest.mark.asyncio
@patch(
    "duo_workflow_service.checkpointer.gitlab_workflow.is_client_capable",
    return_value=True,
)
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_aget_tuple_hydration_tolerates_malformed_current_thread(
    _mock_duo_workflow_metrics,
    _mock_is_client_capable,
    gitlab_workflow,
    http_client,
    compressed_checkpoint_data,
):
    """Malformed current_thread values must not raise; default is kept and hydration of other fields continues."""
    compressed_checkpoint_data[0]["current_thread"] = "not-a-number"
    http_client.aget.return_value = GitLabHttpResponse(
        status_code=200, body=compressed_checkpoint_data
    )

    config = {"configurable": {"thread_id": "1234", "checkpoint_id": "5678"}}
    await gitlab_workflow.aget_tuple(config)

    assert gitlab_workflow._current_thread == 0
    assert gitlab_workflow._prev_checkpoint_id == "5678"


@pytest.mark.asyncio
@patch(
    "duo_workflow_service.checkpointer.gitlab_workflow.is_client_capable",
    return_value=False,
)
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_aget_tuple_skips_hydration_when_capability_disabled(
    _mock_duo_workflow_metrics,
    _mock_is_client_capable,
    gitlab_workflow,
    http_client,
    compressed_checkpoint_data,
):
    compressed_checkpoint_data[0]["current_thread"] = 5
    http_client.aget.return_value = GitLabHttpResponse(
        status_code=200, body=compressed_checkpoint_data
    )

    config = {"configurable": {"thread_id": "1234", "checkpoint_id": "5678"}}
    await gitlab_workflow.aget_tuple(config)

    assert gitlab_workflow._current_thread == 0
    assert gitlab_workflow._prev_checkpoint_id is None


@pytest.mark.asyncio
@patch(
    "duo_workflow_service.checkpointer.gitlab_workflow.is_client_capable",
    return_value=True,
)
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_aput_after_hydration_chains_delta_without_stale_cache_reset(
    _mock_duo_workflow_metrics,
    _mock_is_client_capable,
    gitlab_workflow,
    http_client,
    compressed_checkpoint_data,
    checkpoint_metadata,
):
    """End-to-end: simulate a restart by hydrating then writing — must reuse server thread, no current_thread bump."""
    import base64

    from duo_workflow_service.checkpointer.utils.serializer import CheckpointSerializer

    serde = CheckpointSerializer()
    compressed_checkpoint_data[0]["current_thread"] = 2
    http_client.aget.return_value = GitLabHttpResponse(
        status_code=200, body=compressed_checkpoint_data
    )
    http_client.apost.return_value = GitLabHttpResponse(status_code=200, body={})

    config = {"configurable": {"thread_id": "1234", "checkpoint_id": "5678"}}
    fetched = await gitlab_workflow.aget_tuple(config)
    assert fetched is not None

    next_checkpoint = {
        "id": "ckpt-next",
        "channel_values": dict(fetched.checkpoint["channel_values"]),
    }
    next_checkpoint["channel_values"]["messages"] = ["new"]

    await gitlab_workflow.aput(
        {"configurable": {"checkpoint_id": "5678"}},
        next_checkpoint,
        checkpoint_metadata,
        ChannelVersions({"messages": "1.0"}),
    )

    body = json.loads(http_client.apost.call_args[1]["body"])
    assert body["current_thread"] == 2
    assert len(body["channel_blobs"]) == 1
    blob = body["channel_blobs"][0]
    assert blob["channel"] == "messages"
    assert blob["step_action"] == "compaction"
    val = serde.loads_typed(
        (blob["write_type"], zlib.decompress(base64.b64decode(blob["data"])))
    )
    assert val == ["new"]


def test_decode_graphql_latest_checkpoint_hydrates(gitlab_workflow):
    with patch(
        "duo_workflow_service.checkpointer.gitlab_workflow.is_client_capable",
        return_value=True,
    ):
        gitlab_workflow._decode_graphql_latest_checkpoint(
            {
                "threadTs": "gql-ckpt",
                "parentTs": None,
                "checkpoint": json.dumps(
                    {"id": "gql-ckpt", "channel_values": {"x": [1]}}
                ),
                "metadata": "{}",
                "currentThread": 4,
            }
        )

    assert gitlab_workflow._current_thread == 4
    assert gitlab_workflow._prev_checkpoint_id == "gql-ckpt"
    assert gitlab_workflow._prev_channel_values == {"x": [1]}
