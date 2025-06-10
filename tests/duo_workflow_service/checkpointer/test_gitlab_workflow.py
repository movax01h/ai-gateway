import base64
import json
from typing import Any, Optional, Sequence, TypedDict
from unittest.mock import AsyncMock, call, patch

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
from duo_workflow_service.internal_events import InternalEventAdditionalProperties
from duo_workflow_service.internal_events.event_enum import (
    CategoryEnum,
    EventEnum,
    EventLabelEnum,
    EventPropertyEnum,
)
from duo_workflow_service.json_encoder.encoder import CustomEncoder


class CustomRunnableConfig(TypedDict):
    configurable: Optional[dict]


@pytest.fixture
def http_client_for_retry(http_client, workflow_id):
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
            # Return any status except INPUT_REQUIRED to trigger the RETRY path
            return {"status": "finished"}
        else:
            raise ValueError(f"Unexpected path: {path}")

    http_client.aget = mock_aget
    http_client.apatch.return_value = GitLabHttpResponse(status_code=200, body={})
    http_client.apost.return_value = {"status": 200}

    return http_client


@pytest.fixture
def config() -> CustomRunnableConfig:
    return {"configurable": {"thread_id": "1234", "checkpoint_id": "5678"}}


@pytest.fixture
def http_client():
    return AsyncMock()


@pytest.fixture
def workflow_id():
    return "1234"


@pytest.fixture
def workflow_type():
    return CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT


@pytest.fixture
def gitlab_workflow(
    http_client,
    workflow_id,
    workflow_type,
):
    return GitLabWorkflow(
        http_client,
        workflow_id,
        workflow_type,
    )


@pytest.fixture
def checkpoint_data():
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


@pytest.fixture
def checkpoint_metadata():
    metadata = CheckpointMetadata()
    metadata["writes"] = {"some_node": {"status": "Created", "last_human_input": {}}}
    return metadata


@pytest.mark.asyncio
@patch("duo_workflow_service.checkpointer.gitlab_workflow.DuoWorkflowInternalEvent")
async def test_workflow_event_tracking_for_cancelled_workflow(
    mock_internal_event_tracker,
    gitlab_workflow,
    http_client,
    workflow_id,
    workflow_type,
):
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

    assert mock_internal_event_tracker.track_event.call_count == 2
    mock_internal_event_tracker.track_event.assert_has_calls(
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


@pytest.mark.asyncio
@patch("duo_workflow_service.checkpointer.gitlab_workflow.DuoWorkflowInternalEvent")
async def test_workflow_context_manager_success(
    mock_internal_event_tracker,
    gitlab_workflow,
    http_client,
    workflow_id,
    workflow_type,
):

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

    assert mock_internal_event_tracker.track_event.call_count == 2

    mock_internal_event_tracker.track_event.assert_has_calls(
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


@pytest.mark.asyncio
@patch("duo_workflow_service.checkpointer.gitlab_workflow.DuoWorkflowInternalEvent")
@patch("duo_workflow_service.checkpointer.gitlab_workflow.log_exception")
async def test_workflow_context_manager_startup_error(
    mock_log_exception,
    mock_internal_event_tracker,
    gitlab_workflow,
    http_client,
    workflow_id,
    workflow_type,
):
    http_client.aget.side_effect = ValueError("Startup error simulated")

    async def mock_apatch(path, **kwargs):
        return GitLabHttpResponse(status_code=200, body={})

    http_client.apatch.side_effect = mock_apatch
    with pytest.raises(ValueError) as exc_info:
        async with gitlab_workflow:
            pytest.fail("Context manager body should not execute")

    assert str(exc_info.value) == "Startup error simulated"

    http_client.apatch.assert_called_once_with(
        path=f"/api/v4/ai/duo_workflows/workflows/{workflow_id}",
        body=json.dumps({"status_event": WorkflowStatusEventEnum.DROP.value}),
        parse_json=True,
        use_http_response=True,
    )

    mock_internal_event_tracker.track_event.assert_called_once_with(
        event_name=EventEnum.WORKFLOW_FINISH_FAILURE.value,
        additional_properties=InternalEventAdditionalProperties(
            label=EventLabelEnum.WORKFLOW_FINISH_LABEL.value,
            property="ValueError('Startup error simulated')",
            value=workflow_id,
        ),
        category=workflow_type,
    )

    # The log_exception for status update shouldn't be called since status update succeeded
    mock_log_exception.assert_not_called()


@pytest.mark.asyncio
@patch("duo_workflow_service.checkpointer.gitlab_workflow.DuoWorkflowInternalEvent")
@patch("duo_workflow_service.checkpointer.gitlab_workflow.log_exception")
async def test_workflow_context_manager_startup_error_with_status_update_failure(
    mock_log_exception,
    mock_internal_event_tracker,
    gitlab_workflow,
    http_client,
    workflow_id,
    workflow_type,
):
    http_client.aget.side_effect = ValueError("Startup error simulated")

    status_error = ConnectionError("Status update failed")
    http_client.apatch.side_effect = status_error

    with pytest.raises(ValueError) as exc_info:
        async with gitlab_workflow:
            pytest.fail("Context manager body should not execute")

    assert str(exc_info.value) == "Startup error simulated"

    http_client.apatch.assert_called_once_with(
        path=f"/api/v4/ai/duo_workflows/workflows/{workflow_id}",
        body=json.dumps({"status_event": WorkflowStatusEventEnum.DROP.value}),
        parse_json=True,
        use_http_response=True,
    )

    mock_internal_event_tracker.track_event.assert_called_once_with(
        event_name=EventEnum.WORKFLOW_FINISH_FAILURE.value,
        additional_properties=InternalEventAdditionalProperties(
            label=EventLabelEnum.WORKFLOW_FINISH_LABEL.value,
            property="ValueError('Startup error simulated')",
            value=workflow_id,
        ),
        category=workflow_type,
    )

    mock_log_exception.assert_called_once_with(
        status_error,
        extra={
            "workflow_id": workflow_id,
            "context": "Failed to update workflow status during startup error",
        },
    )


@pytest.mark.asyncio
@patch("duo_workflow_service.checkpointer.gitlab_workflow.DuoWorkflowInternalEvent")
async def test_workflow_context_manager_resume_interrupted(
    mock_internal_event_tracker,
    gitlab_workflow,
    http_client,
    workflow_id,
    workflow_type,
):
    gitlab_workflow._status_handler = AsyncMock()
    gitlab_workflow._status_handler.get_workflow_status.side_effect = [
        WorkflowStatusEnum.INPUT_REQUIRED,
        WorkflowStatusEnum.PLANNING,
    ]

    async with gitlab_workflow as workflow:
        assert isinstance(workflow, GitLabWorkflow)

    gitlab_workflow._status_handler.update_workflow_status.assert_called_once_with(
        workflow_id, WorkflowStatusEventEnum.RESUME
    )

    mock_internal_event_tracker.track_event.assert_has_calls(
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
@patch("duo_workflow_service.checkpointer.gitlab_workflow.DuoWorkflowInternalEvent")
async def test_workflow_context_manager_resume_interrupted_approval(
    mock_internal_event_tracker,
    gitlab_workflow,
    http_client,
    workflow_id,
    workflow_type,
):
    gitlab_workflow._status_handler = AsyncMock()
    gitlab_workflow._status_handler.get_workflow_status.side_effect = [
        WorkflowStatusEnum.PLAN_APPROVAL_REQUIRED,
        WorkflowStatusEnum.EXECUTION,
    ]

    async with gitlab_workflow as workflow:
        assert isinstance(workflow, GitLabWorkflow)

    gitlab_workflow._status_handler.update_workflow_status.assert_called_once_with(
        workflow_id, WorkflowStatusEventEnum.RESUME
    )

    mock_internal_event_tracker.track_event.assert_has_calls(
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
@patch("duo_workflow_service.checkpointer.gitlab_workflow.DuoWorkflowInternalEvent")
async def test_workflow_context_manager_retry_success(
    mock_internal_event_tracker,
    gitlab_workflow,
    http_client_for_retry,
    workflow_id,
    workflow_type,
):
    async with gitlab_workflow as workflow:
        assert isinstance(workflow, GitLabWorkflow)

    http_client_for_retry.apatch.assert_called_once_with(
        path=f"/api/v4/ai/duo_workflows/workflows/{workflow_id}",
        body=json.dumps({"status_event": WorkflowStatusEventEnum.RETRY.value}),
        parse_json=True,
        use_http_response=True,
    )

    assert mock_internal_event_tracker.track_event.call_count == 2
    mock_internal_event_tracker.track_event.assert_has_calls(
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


@pytest.mark.asyncio
@patch("duo_workflow_service.checkpointer.gitlab_workflow.DuoWorkflowInternalEvent")
async def test_workflow_context_manager_error(
    mock_internal_event_tracker,
    gitlab_workflow,
    http_client,
    workflow_id,
    workflow_type,
):
    http_client.aget.return_value = []
    http_client.apatch.return_value = GitLabHttpResponse(status_code=200, body={})

    with pytest.raises(ValueError):
        async with gitlab_workflow:
            raise ValueError("Test error")

    http_client.apatch.assert_called_with(
        path=f"/api/v4/ai/duo_workflows/workflows/{workflow_id}",
        body=json.dumps({"status_event": WorkflowStatusEventEnum.DROP.value}),
        parse_json=True,
        use_http_response=True,
    )

    assert mock_internal_event_tracker.track_event.call_count == 2

    mock_internal_event_tracker.track_event.assert_has_calls(
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
                event_name=EventEnum.WORKFLOW_FINISH_FAILURE.value,
                additional_properties=InternalEventAdditionalProperties(
                    label=EventLabelEnum.WORKFLOW_FINISH_LABEL.value,
                    property="ValueError('Test error')",
                    value="1234",
                ),
                category=workflow_type,
            ),
        ]
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
):
    workflow = GitLabWorkflow(http_client, workflow_id, workflow_type=workflow_type)
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


def test_aput_with_no_status_update_and_human_input(
    checkpoint_data,
    checkpoint_metadata,
    http_client,
    workflow_id,
    workflow_type,
):
    workflow = GitLabWorkflow(
        client=http_client,
        workflow_id=workflow_id,
        workflow_type=workflow_type,
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
async def test_resume_status_event(
    gitlab_workflow, http_client, checkpoint_metadata, workflow_id
):
    checkpoint = {
        "id": "resume-checkpoint",
        "channel_values": {"last_human_input": {"event_type": "resume"}},
    }
    config: RunnableConfig = {"configurable": {}}

    http_client.apatch.return_value = GitLabHttpResponse(status_code=200, body={})

    await gitlab_workflow.aput(
        config, checkpoint, checkpoint_metadata, ChannelVersions()
    )

    http_client.apatch.assert_called_with(
        path=f"/api/v4/ai/duo_workflows/workflows/{workflow_id}",
        body=json.dumps({"status_event": WorkflowStatusEventEnum.RESUME.value}),
        parse_json=True,
        use_http_response=True,
    )


@pytest.mark.asyncio
@patch("duo_workflow_service.checkpointer.gitlab_workflow.DuoWorkflowInternalEvent")
async def test_track_workflow_completion_early_return(
    mock_internal_event_tracker, gitlab_workflow
):
    # Test with a status that doesn't match any of the specific cases
    await gitlab_workflow._track_workflow_completion("some_other_status")

    # Verify no event was tracked
    mock_internal_event_tracker.track_event.assert_not_called()


@pytest.mark.asyncio
@patch.dict("os.environ", {"USE_MEMSAVER": "true"}, clear=True)
async def test_offline_mode(http_client, workflow_id, workflow_type):
    gitlab_workflow = GitLabWorkflow(
        client=http_client,
        workflow_id=workflow_id,
        workflow_type=workflow_type,
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
