import asyncio
import json
from typing import Any, Optional, Sequence, TypedDict
from unittest.mock import AsyncMock, Mock, call, patch

import pytest
from langchain_core.messages import SystemMessage
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
)
from duo_workflow_service.entities.state import WorkflowStatusEnum
from duo_workflow_service.gitlab.http_client import (
    GitLabHttpResponse,
    checkpoint_decoder,
)
from duo_workflow_service.json_encoder.encoder import CustomEncoder
from lib.internal_events import InternalEventAdditionalProperties
from lib.internal_events.event_enum import (
    CategoryEnum,
    EventEnum,
    EventLabelEnum,
    EventPropertyEnum,
)


class CustomRunnableConfig(TypedDict):
    configurable: Optional[dict]


@pytest.fixture(name="http_client_for_retry")
def http_client_for_retry_fixture(http_client, workflow_id):
    async def mock_aget(path, **kwargs):
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
            return {"status": "finished"}
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


@pytest.fixture(name="workflow_id")
def workflow_id_fixture():
    return "1234"


@pytest.fixture(name="workflow_type")
def workflow_type_fixture():
    return CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT


@pytest.fixture(autouse=True)
def prepare_container(mock_duo_workflow_service_container):
    pass


@pytest.fixture(name="workflow_config")
def workflow_config_fixture():
    return {
        "first_checkpoint": None,
        "workflow_status": "created",
        "agent_privileges_names": ["read_repository"],
        "pre_approved_agent_privileges_names": [],
        "mcp_enabled": True,
        "allow_agent_to_request_user": True,
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


@pytest.fixture(name="checkpoint_data")
def checkpoint_data_fixture():
    return [
        {
            "thread_ts": "5678",
            "parent_ts": None,
            "checkpoint": {
                "id": "checkpoint_id",
                "channel_values": {
                    "conversation_history": {
                        "planner": SystemMessage(
                            content="You are a planner. Could you create a detailed plan that the execution agent"
                        )
                    }
                },
            },
            "metadata": {"timestamp": "2024-01-01"},
        }
    ]


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
    }

    gitlab_workflow = GitLabWorkflow(
        http_client,
        workflow_id,
        workflow_type,
        workflow_config,  # type: ignore[arg-type]
    )
    gitlab_workflow._internal_event_client = internal_event_client

    async def mock_aget(path, **kwargs):
        if (
            path
            == f"/api/v4/ai/duo_workflows/workflows/{workflow_id}/checkpoints?per_page=1"
        ):
            return []  # No checkpoints for a new workflow
        elif path == f"/api/v4/ai/duo_workflows/workflows/{workflow_id}":
            return {"status": "stopped"}  # Workflow was cancelled
        raise ValueError(f"Unexpected path: {path}")

    async def mock_apatch(path, **kwargs):
        return GitLabHttpResponse(status_code=200, body={})

    http_client.aget.side_effect = mock_aget
    http_client.apatch.side_effect = mock_apatch

    async with gitlab_workflow as workflow:
        assert isinstance(workflow, GitLabWorkflow)

    http_client.apatch.assert_called_once_with(
        path=f"/api/v4/ai/duo_workflows/workflows/{workflow_id}",
        body=json.dumps({"status_event": WorkflowStatusEventEnum.START.value}),
        parse_json=True,
        use_http_response=True,
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
    }

    gitlab_workflow = GitLabWorkflow(
        http_client,
        workflow_id,
        workflow_type,
        workflow_config,  # type: ignore[arg-type]
    )
    gitlab_workflow._internal_event_client = internal_event_client

    async def mock_aget(path, **kwargs):
        if (
            path
            == f"/api/v4/ai/duo_workflows/workflows/{workflow_id}/checkpoints?per_page=1"
        ):
            return []  # No checkpoints for a new workflow
        elif path == f"/api/v4/ai/duo_workflows/workflows/{workflow_id}":
            return {"status": "finished"}
        raise ValueError(f"Unexpected path: {path}")

    async def mock_apatch(path, **kwargs):
        return GitLabHttpResponse(status_code=200, body={})

    http_client.aget.side_effect = mock_aget
    http_client.apatch.side_effect = mock_apatch

    async with gitlab_workflow as workflow:
        assert isinstance(workflow, GitLabWorkflow)

    http_client.apatch.assert_called_once_with(
        path=f"/api/v4/ai/duo_workflows/workflows/{workflow_id}",
        body=json.dumps({"status_event": WorkflowStatusEventEnum.START.value}),
        parse_json=True,
        use_http_response=True,
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

    async def mock_apatch(path, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # First call (START status) raises an exception
            raise ValueError("Startup error simulated")
        else:
            # Second call (DROP status) succeeds
            return GitLabHttpResponse(status_code=200, body={})

    http_client.apatch.side_effect = mock_apatch

    with pytest.raises(ValueError) as exc_info:
        async with gitlab_workflow:
            pytest.fail("Context manager body should not execute")

    assert str(exc_info.value) == "Startup error simulated"

    # Verify that apatch was called twice - first with START (which failed) and then with DROP
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

    # The log_exception for status update shouldn't be called since the second status update succeeded
    mock_log_exception.assert_not_called()

    # Verify the failure metric was called
    mock_duo_workflow_metrics.count_agent_platform_session_failure.assert_called_once_with(
        flow_type=workflow_type.value,
        failure_reason="ValueError",
    )


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

    async def mock_apatch(path, **kwargs):
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
    async def mock_aget(path, **kwargs):
        if "checkpoints" in path:
            return []
        else:
            return {"status": "created"}

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
    mock_log_exception.assert_called_once_with(
        status_error,
        extra={
            "workflow_id": workflow_id,
            "context": "Failed to update workflow status during startup error",
        },
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
        "workflow_status": "created",  # Any status that's not INPUT_REQUIRED or PLAN_APPROVAL_REQUIRED
        "agent_privileges_names": ["read_repository"],
        "pre_approved_agent_privileges_names": [],
        "mcp_enabled": True,
        "allow_agent_to_request_user": True,
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
        use_http_response=True,
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
    }

    gitlab_workflow = GitLabWorkflow(
        http_client,
        workflow_id,
        workflow_type,
        workflow_config,  # type: ignore[arg-type]
    )
    gitlab_workflow._internal_event_client = internal_event_client

    # Mock different responses for different API calls
    def mock_aget(path, **kwargs):
        if "checkpoints" in path:
            return []
        else:
            return {"status": "running"}

    http_client.aget.side_effect = mock_aget
    http_client.apatch.return_value = GitLabHttpResponse(status_code=200, body={})

    with pytest.raises(type(exception)):
        async with gitlab_workflow:
            raise exception

    http_client.apatch.assert_called_with(
        path=f"/api/v4/ai/duo_workflows/workflows/{workflow_id}",
        body=json.dumps({"status_event": WorkflowStatusEventEnum.DROP.value}),
        parse_json=True,
        use_http_response=True,
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


@pytest.mark.asyncio
async def test_aget_tuple(
    gitlab_workflow, http_client, config, workflow_id, checkpoint_data
):
    http_client.aget.return_value = checkpoint_data

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

    http_client.aget.assert_called_once_with(
        path=f"/api/v4/ai/duo_workflows/workflows/{workflow_id}/checkpoints",
        object_hook=checkpoint_decoder,
    )


@pytest.mark.asyncio
async def test_aget_tuple_when_config_has_no_checkpoint_id_and_checkpoints_present(
    gitlab_workflow, http_client, workflow_id, checkpoint_data
):
    config: CustomRunnableConfig = {"configurable": {"thread_id": workflow_id}}

    http_client.aget.return_value = checkpoint_data

    result = await gitlab_workflow.aget_tuple(config)
    assert result is not None
    assert isinstance(result, CheckpointTuple)
    assert result.checkpoint == checkpoint_data[0]["checkpoint"]
    assert result.metadata == checkpoint_data[0]["metadata"]

    http_client.aget.assert_called_once_with(
        path=f"/api/v4/ai/duo_workflows/workflows/{workflow_id}/checkpoints?per_page=1",
        object_hook=checkpoint_decoder,
    )


@pytest.mark.asyncio
async def test_aget_tuple_no_checkpoints(gitlab_workflow, http_client, config):
    http_client.aget.return_value = []
    result = await gitlab_workflow.aget_tuple(config)
    assert result is None


@pytest.mark.asyncio
async def test_aget_tuple_when_server_returns_non_success_response(
    gitlab_workflow, workflow_id
):
    config: CustomRunnableConfig = {"configurable": {"thread_id": workflow_id}}

    gitlab_workflow._client.aget = AsyncMock(
        return_value={"status": 400, "reason": "Bad request"}
    )

    with pytest.raises(Exception) as exc_info:
        await gitlab_workflow.aget_tuple(config)

    assert (
        str(exc_info.value)
        == "Failed to fetch checkpoints: {'status': 400, 'reason': 'Bad request'}"
    )

    gitlab_workflow._client.aget.assert_called_once_with(
        path=f"/api/v4/ai/duo_workflows/workflows/{workflow_id}/checkpoints?per_page=1",
        object_hook=checkpoint_decoder,
    )


@pytest.mark.asyncio
async def test_alist(gitlab_workflow, http_client):
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
    http_client.aget.return_value = checkpoints

    results: list[CheckpointTuple] = [
        checkpoint async for checkpoint in gitlab_workflow.alist(None)
    ]

    assert len(results) == 2
    assert results[0].checkpoint == checkpoints[0]["checkpoint"]  # type: ignore
    assert results[1].checkpoint == checkpoints[1]["checkpoint"]  # type: ignore


@pytest.mark.asyncio
async def test_aput(
    gitlab_workflow, http_client, checkpoint_data, checkpoint_metadata, workflow_id
):
    config = {"configurable": {"checkpoint_id": "parent-checkpoint"}}
    checkpoint = checkpoint_data[0]["checkpoint"]
    checkpoint["channel_values"]["status"] = WorkflowStatusEnum.COMPLETED

    http_client.apatch.return_value = GitLabHttpResponse(status_code=200, body={})

    result = await gitlab_workflow.aput(
        config, checkpoint, checkpoint_metadata, ChannelVersions()
    )

    http_client.apost.assert_called_once_with(
        path=f"/api/v4/ai/duo_workflows/workflows/{workflow_id}/checkpoints",
        body=json.dumps(
            {
                "thread_ts": checkpoint["id"],
                "parent_ts": "parent-checkpoint",
                "checkpoint": checkpoint,
                "metadata": checkpoint_metadata,
            },
            cls=CustomEncoder,
        ),
        use_http_response=True,
    )

    http_client.apatch.assert_called_once_with(
        path=f"/api/v4/ai/duo_workflows/workflows/{workflow_id}",
        body=json.dumps({"status_event": WorkflowStatusEventEnum.FINISH.value}),
        parse_json=True,
        use_http_response=True,
    )

    assert result == {
        "configurable": {"thread_id": workflow_id, "checkpoint_id": checkpoint["id"]}
    }


def test_aput_with_no_status_update(
    checkpoint_data,
    checkpoint_metadata,
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
    checkpoint = checkpoint_data[0]["checkpoint"]

    # no status update in checkpoint
    checkpoint["channel_values"]["status"] = None
    status_event = workflow._get_workflow_status_event(checkpoint, checkpoint_metadata)
    assert status_event is None

    # no status update in checkpoint metadata
    checkpoint["channel_values"]["status"] = "Paused"
    checkpoint_metadata["writes"] = {"execution_handover": {"handover": []}}
    status_event = workflow._get_workflow_status_event(checkpoint, checkpoint_metadata)
    assert status_event is None


def test_aput_with_noop_status_update(
    checkpoint_data,
    checkpoint_metadata,
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
    checkpoint = checkpoint_data[0]["checkpoint"]

    # no status update in checkpoint
    checkpoint["channel_values"]["status"] = "approval_error"
    status_event = workflow._get_workflow_status_event(checkpoint, checkpoint_metadata)
    assert status_event is None


def test_aput_with_no_status_update_and_human_input(
    checkpoint_data,
    checkpoint_metadata,
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
    checkpoint = checkpoint_data[0]["checkpoint"]

    # no status update in checkpoint
    checkpoint["channel_values"]["status"] = None
    checkpoint["channel_values"]["last_human_input"] = None
    status_event = workflow._get_workflow_status_event(checkpoint, checkpoint_metadata)
    assert status_event is None

    # no status update in checkpoint metadata
    checkpoint["channel_values"]["status"] = "Paused"
    checkpoint["channel_values"]["last_human_input"] = {"event_type": "pause"}
    checkpoint_metadata["writes"] = {
        "planning_check_human_input": {"last_human_input": {"event_type": "resume"}}
    }
    status_event = workflow._get_workflow_status_event(checkpoint, checkpoint_metadata)
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
    checkpoint_metadata,
    workflow_id,
    status,
    expected_event,
):
    checkpoint = {
        "id": f"checkpoint-{status}",
        "channel_values": {"status": status},
    }
    config: RunnableConfig = {"configurable": {}}

    http_client.apatch.return_value = GitLabHttpResponse(status_code=200, body={})

    await gitlab_workflow.aput(
        config, checkpoint, checkpoint_metadata, ChannelVersions()
    )

    http_client.apatch.assert_called_once_with(
        path=f"/api/v4/ai/duo_workflows/workflows/{workflow_id}",
        body=json.dumps({"status_event": expected_event.value}),
        parse_json=True,
        use_http_response=True,
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
        path=f"/api/v4/ai/duo_workflows/workflows/123/checkpoint_writes_batch",
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
