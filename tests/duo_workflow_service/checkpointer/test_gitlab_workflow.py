# pylint: disable=comparison-with-callable,import-outside-toplevel,line-too-long,no-else-raise,no-else-return,too-many-lines
import asyncio
import json
import zlib
from asyncio import CancelledError
from typing import Any, Optional, Sequence, TypedDict
from unittest.mock import ANY, AsyncMock, Mock, call, patch
from urllib.parse import parse_qs, urlparse

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
    _serialize_all_channels_full,
    _serialize_channel_blobs,
    _thread_started_at_from_id,
)
from duo_workflow_service.checkpointer.gitlab_workflow_utils import compress_checkpoint
from duo_workflow_service.entities.state import WorkflowStatusEnum
from duo_workflow_service.errors.typing import (
    InvalidRequestException,
    NotifiableException,
)
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
        "incremental_checkpoints_enabled": False,
        "allow_agent_to_request_user": True,
        "archived": False,
        "stalled": False,
    }


@pytest.fixture(name="incremental_enabled")
def incremental_enabled_fixture(workflow_config):
    """Enable incremental checkpoints on the workflow_config consumed by the gitlab_workflow fixture."""
    workflow_config["incremental_checkpoints_enabled"] = True


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


def _make_workflow_for_reconciliation(
    http_client,
    workflow_id,
    workflow_type,
    *,
    checkpoint_status: WorkflowStatusEnum,
    use_prev_channel_values: bool = True,
) -> tuple[GitLabWorkflow, AsyncMock]:
    """Build a GitLabWorkflow wired for reconciliation tests.

    Constructs the workflow with a resumable config and replaces _status_handler with an AsyncMock so tests can assert
    on update_workflow_status calls.  Returns both the workflow and the mock so callers can configure
    get_workflow_status.return_value and inspect call_args_list with proper typing.

    Does NOT configure http_client.aget — each test is responsible for setting up the HTTP responses it needs
    (checkpoint API, Rails status, etc.).

    When use_prev_channel_values=True the checkpoint status is injected into _prev_channel_values
    (incremental_checkpoints fast path).  Set it to False to leave _prev_channel_values empty, forcing the API fallback
    path.
    """
    workflow_config = {
        "first_checkpoint": {"checkpoint": "{}"},
        "workflow_status": WorkflowStatusEnum.INPUT_REQUIRED,
        "agent_privileges_names": [],
        "pre_approved_agent_privileges_names": [],
        "mcp_enabled": False,
        "allow_agent_to_request_user": True,
        "archived": False,
        "stalled": False,
    }

    http_client.apatch.return_value = GitLabHttpResponse(status_code=200, body={})

    gitlab_workflow = GitLabWorkflow(
        http_client,
        workflow_id,
        workflow_type,
        workflow_config,  # type: ignore[arg-type]
    )
    status_handler_mock = AsyncMock()
    gitlab_workflow._status_handler = status_handler_mock

    if use_prev_channel_values:
        gitlab_workflow._prev_channel_values = {"status": checkpoint_status}

    return gitlab_workflow, status_handler_mock


def _make_checkpoint_aget(
    workflow_id: str, checkpoint_status: WorkflowStatusEnum, rails_status: str
):
    """Return an async aget function that serves checkpoint and workflow status responses."""
    inner = {"id": "cp-1", "channel_values": {"status": checkpoint_status}}
    compressed = compress_checkpoint(inner)  # type: ignore[arg-type]

    async def mock_aget(path, **_kwargs):
        if "checkpoints?per_page=1" in path:
            return GitLabHttpResponse(
                status_code=200,
                body=[
                    {
                        "thread_ts": "cp-1",
                        "parent_ts": None,
                        "compressed_checkpoint": compressed,
                        "metadata": {},
                    }
                ],
            )
        if f"/workflows/{workflow_id}" in path:
            return GitLabHttpResponse(status_code=200, body={"status": rails_status})
        raise ValueError(f"Unexpected path: {path}")

    return mock_aget


@pytest.mark.asyncio
async def test_bad_request_reconciles_rails_to_checkpoint_status_via_prev_channel_values(
    http_client,
    workflow_id,
    workflow_type,
    internal_event_client: Mock,
):
    """Core bug (gitlab-org/gitlab#602799): after an empty-goal reconnect is rejected, Rails is driven back to
    INPUT_REQUIRED using the in-memory checkpoint status (_prev_channel_values, incremental_checkpoints-capable client).

    The checkpoint API must NOT be called — the status is read from memory.
    """
    gitlab_workflow, status_handler_mock = _make_workflow_for_reconciliation(
        http_client,
        workflow_id,
        workflow_type,
        checkpoint_status=WorkflowStatusEnum.INPUT_REQUIRED,
        use_prev_channel_values=True,
    )
    gitlab_workflow._internal_event_client = internal_event_client
    status_handler_mock.get_workflow_status.return_value = "running"

    checkpoint_api_calls = []

    async def tracking_aget(path, **_kwargs):
        if "checkpoints?per_page=1" in path:
            checkpoint_api_calls.append(path)
        raise ValueError(f"Unexpected path: {path}")

    http_client.aget = tracking_aget

    with pytest.raises(InvalidRequestException):
        async with gitlab_workflow:
            raise InvalidRequestException("empty goal")

    status_calls = [
        c.args[1] for c in status_handler_mock.update_workflow_status.call_args_list
    ]
    assert status_calls == [
        WorkflowStatusEventEnum.RESUME,
        WorkflowStatusEventEnum.REQUIRE_INPUT,
    ]
    assert not checkpoint_api_calls, (
        "checkpoint API must not be called when _prev_channel_values is populated"
    )


@pytest.mark.asyncio
async def test_bad_request_reconciles_rails_to_checkpoint_status_via_api_fallback(
    http_client,
    workflow_id,
    workflow_type,
    internal_event_client: Mock,
):
    """Same reconciliation but via the API fallback path (older client without incremental_checkpoints, so
    _prev_channel_values is empty).

    The checkpoint API MUST be called to fetch the status.
    """
    gitlab_workflow, status_handler_mock = _make_workflow_for_reconciliation(
        http_client,
        workflow_id,
        workflow_type,
        checkpoint_status=WorkflowStatusEnum.INPUT_REQUIRED,
        use_prev_channel_values=False,
    )
    gitlab_workflow._internal_event_client = internal_event_client
    status_handler_mock.get_workflow_status.return_value = "running"

    checkpoint_api_calls = []
    real_aget = _make_checkpoint_aget(
        workflow_id, WorkflowStatusEnum.INPUT_REQUIRED, "running"
    )

    async def tracking_aget(path, **kwargs):
        if "checkpoints?per_page=1" in path:
            checkpoint_api_calls.append(path)
        return await real_aget(path, **kwargs)

    http_client.aget = tracking_aget

    with pytest.raises(InvalidRequestException):
        async with gitlab_workflow:
            raise InvalidRequestException("empty goal")

    status_calls = [
        c.args[1] for c in status_handler_mock.update_workflow_status.call_args_list
    ]
    assert status_calls == [
        WorkflowStatusEventEnum.RESUME,
        WorkflowStatusEventEnum.REQUIRE_INPUT,
    ]
    assert len(checkpoint_api_calls) == 1, (
        "checkpoint API must be called when _prev_channel_values is empty"
    )


@pytest.mark.asyncio
async def test_bad_request_skips_reconciliation_when_rails_already_matches(
    http_client,
    workflow_id,
    workflow_type,
    internal_event_client: Mock,
):
    """No update is sent when Rails already reflects the checkpoint status."""
    gitlab_workflow, status_handler_mock = _make_workflow_for_reconciliation(
        http_client,
        workflow_id,
        workflow_type,
        checkpoint_status=WorkflowStatusEnum.INPUT_REQUIRED,
    )
    gitlab_workflow._internal_event_client = internal_event_client
    status_handler_mock.get_workflow_status.return_value = (
        WorkflowStatusEnum.INPUT_REQUIRED
    )

    with pytest.raises(InvalidRequestException):
        async with gitlab_workflow:
            raise InvalidRequestException("empty goal")

    status_calls = [
        c.args[1] for c in status_handler_mock.update_workflow_status.call_args_list
    ]
    assert status_calls == [WorkflowStatusEventEnum.RESUME]


@pytest.mark.asyncio
async def test_bad_request_skips_reconciliation_when_checkpoint_status_has_no_rails_event(
    http_client,
    workflow_id,
    workflow_type,
    internal_event_client: Mock,
):
    """No update is sent when the checkpoint status has no corresponding Rails event.

    EXECUTION maps to RUNNING which has no entry in CHECKPOINT_STATUS_TO_STATUS_EVENT.
    """
    gitlab_workflow, status_handler_mock = _make_workflow_for_reconciliation(
        http_client,
        workflow_id,
        workflow_type,
        checkpoint_status=WorkflowStatusEnum.EXECUTION,
    )
    gitlab_workflow._internal_event_client = internal_event_client
    status_handler_mock.get_workflow_status.return_value = "running"

    with pytest.raises(InvalidRequestException):
        async with gitlab_workflow:
            raise InvalidRequestException("empty goal")

    status_calls = [
        c.args[1] for c in status_handler_mock.update_workflow_status.call_args_list
    ]
    assert status_calls == [WorkflowStatusEventEnum.RESUME]


@pytest.mark.asyncio
async def test_bad_request_reconciliation_survives_checkpoint_api_error(
    http_client,
    workflow_id,
    workflow_type,
    internal_event_client: Mock,
):
    """A checkpoint API failure during reconciliation must not mask the original exception."""
    gitlab_workflow, status_handler_mock = _make_workflow_for_reconciliation(
        http_client,
        workflow_id,
        workflow_type,
        checkpoint_status=WorkflowStatusEnum.INPUT_REQUIRED,
        use_prev_channel_values=False,
    )
    gitlab_workflow._internal_event_client = internal_event_client
    status_handler_mock.get_workflow_status.return_value = "running"

    async def failing_aget(path, **_kwargs):
        if "checkpoints?per_page=1" in path:
            return GitLabHttpResponse(status_code=500, body={"error": "server error"})
        raise ValueError(f"Unexpected path: {path}")

    http_client.aget = failing_aget

    # The InvalidRequestException must still propagate despite the API error.
    with pytest.raises(InvalidRequestException):
        async with gitlab_workflow:
            raise InvalidRequestException("empty goal")

    status_calls = [
        c.args[1] for c in status_handler_mock.update_workflow_status.call_args_list
    ]
    assert status_calls == [WorkflowStatusEventEnum.RESUME]


@pytest.mark.asyncio
async def test_bad_request_skips_reconciliation_when_no_checkpoints_exist(
    http_client,
    workflow_id,
    workflow_type,
    internal_event_client: Mock,
):
    """No update is sent when the checkpoint API returns an empty list.

    Covers the ``_fetch_most_recent_checkpoint`` → ``return None`` path
    and the ``_reconcile_session_status`` early-return when
    ``_get_latest_checkpoint_status`` returns ``None``.
    """
    gitlab_workflow, status_handler_mock = _make_workflow_for_reconciliation(
        http_client,
        workflow_id,
        workflow_type,
        checkpoint_status=WorkflowStatusEnum.INPUT_REQUIRED,
        use_prev_channel_values=False,
    )
    gitlab_workflow._internal_event_client = internal_event_client

    async def empty_aget(path, **_kwargs):
        if "checkpoints?per_page=1" in path:
            return GitLabHttpResponse(status_code=200, body=[])
        raise ValueError(f"Unexpected path: {path}")

    http_client.aget = empty_aget

    with pytest.raises(InvalidRequestException):
        async with gitlab_workflow:
            raise InvalidRequestException("empty goal")

    # Only the entry RESUME — no reconciliation update because there are no checkpoints.
    status_calls = [
        c.args[1] for c in status_handler_mock.update_workflow_status.call_args_list
    ]
    assert status_calls == [WorkflowStatusEventEnum.RESUME]


@pytest.mark.asyncio
async def test_bad_request_skips_reconciliation_when_checkpoint_status_has_no_workflow_mapping(
    http_client,
    workflow_id,
    workflow_type,
    internal_event_client: Mock,
):
    """No update is sent when the checkpoint status has no WORKFLOW_STATUS_TO_CHECKPOINT_STATUS entry.

    Covers the ``checkpoint_status_str is None`` early-return in
    ``_reconcile_session_status``.  We inject a raw string value that is
    not a member of ``WorkflowStatusEnum`` directly into ``_prev_channel_values`` to
    simulate a future/unknown status that the current mapping does not cover.
    """
    gitlab_workflow, status_handler_mock = _make_workflow_for_reconciliation(
        http_client,
        workflow_id,
        workflow_type,
        checkpoint_status=WorkflowStatusEnum.INPUT_REQUIRED,
        use_prev_channel_values=False,
    )
    gitlab_workflow._internal_event_client = internal_event_client
    # Inject an unknown status value that has no entry in WORKFLOW_STATUS_TO_CHECKPOINT_STATUS.
    gitlab_workflow._prev_channel_values = {"status": "UNKNOWN_FUTURE_STATUS"}

    with pytest.raises(InvalidRequestException):
        async with gitlab_workflow:
            raise InvalidRequestException("empty goal")

    # Only the entry RESUME — no reconciliation update because the status is unmapped.
    status_calls = [
        c.args[1] for c in status_handler_mock.update_workflow_status.call_args_list
    ]
    assert status_calls == [WorkflowStatusEventEnum.RESUME]


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


def _checkpoint_saved_kwargs(logger):
    for logger_call in logger.info.call_args_list:
        if logger_call.args and logger_call.args[0] == "Checkpoint saved":
            return logger_call.kwargs
    raise AssertionError("'Checkpoint saved' was not logged")


@pytest.mark.asyncio
async def test_aput_logs_full_checkpoint_strategy(
    gitlab_workflow, http_client, checkpoint_data, checkpoint_metadata
):
    gitlab_workflow._logger = Mock()
    config = {"configurable": {"checkpoint_id": "parent-checkpoint"}}
    http_client.apost.return_value = GitLabHttpResponse(status_code=200, body={})

    await gitlab_workflow.aput(
        config, checkpoint_data[0]["checkpoint"], checkpoint_metadata, ChannelVersions()
    )

    assert _checkpoint_saved_kwargs(gitlab_workflow._logger)["checkpoint_strategy"] == (
        "full"
    )
    assert "?checkpoint_strategy=full" in http_client.apost.call_args[1]["path"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("incremental_enabled")
async def test_aput_logs_incremental_checkpoint_strategy(
    gitlab_workflow, http_client, checkpoint_data, checkpoint_metadata
):
    gitlab_workflow._logger = Mock()
    config = {"configurable": {"checkpoint_id": "parent-checkpoint"}}
    http_client.apost.return_value = GitLabHttpResponse(status_code=200, body={})

    await gitlab_workflow.aput(
        config, checkpoint_data[0]["checkpoint"], checkpoint_metadata, ChannelVersions()
    )

    assert _checkpoint_saved_kwargs(gitlab_workflow._logger)["checkpoint_strategy"] == (
        "incremental"
    )
    assert "?checkpoint_strategy=incremental" in http_client.apost.call_args[1]["path"]


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

    checkpoint = {
        "id": "ckpt1",
        "channel_values": {
            "messages": ["a", "b"],
            "status": "running",
        },
    }
    new_versions = ChannelVersions({"messages": "2.0"})

    blobs, _ = _serialize_channel_blobs(checkpoint, new_versions, {})

    assert len(blobs) == 1
    assert blobs[0]["channel"] == "messages"
    assert blobs[0]["version"] == "2.0"
    assert blobs[0]["write_type"] == "json"
    assert base64.b64decode(blobs[0]["data"])


def test_serialize_channel_blobs_skips_scalar_channels():
    checkpoint = {
        "id": "ckpt1",
        "channel_values": {
            "messages": ["a", "b"],
            "status": "running",
            "goal": "fix the bug",
        },
    }
    new_versions = ChannelVersions({"messages": "2.0", "status": "1.0", "goal": "1.0"})

    blobs, _ = _serialize_channel_blobs(checkpoint, new_versions, {})

    channels = [b["channel"] for b in blobs]
    assert "goal" not in channels
    assert "messages" in channels
    # status is always blobbed for reconstruction, even though it is a scalar
    assert "status" in channels


def test_serialize_channel_blobs_status_always_compaction_and_no_thread_bump():
    """Status blobs must carry step_action='compaction' and must not set is_compaction.

    status is a scalar channel so it bypasses the list/dict delta branches entirely. This means it always serialises as
    a full replacement (step_action='compaction') and a status-only change must not trigger is_compaction=True (which
    would incorrectly bump current_thread).
    """
    checkpoint = {
        "id": "ckpt1",
        "channel_values": {
            "status": "running",
        },
    }
    new_versions = ChannelVersions({"status": "2.0"})
    prev_channel_values = {"status": "waiting"}

    blobs, is_compaction = _serialize_channel_blobs(
        checkpoint, new_versions, prev_channel_values
    )

    assert len(blobs) == 1
    status_blob = blobs[0]
    assert status_blob["channel"] == "status"
    # Scalar path always produces a full replacement, never a delta
    assert status_blob["step_action"] == "compaction"
    # A status-only change must NOT set is_compaction — that would incorrectly bump current_thread
    assert is_compaction is False


def test_serialize_channel_blobs_list_delta():
    import base64

    checkpoint = {
        "id": "ckpt2",
        "channel_values": {"messages": ["a", "b", "c"]},
    }
    new_versions = ChannelVersions({"messages": "3.0"})
    prev_channel_values = {"messages": ["a", "b"]}

    blobs, _ = _serialize_channel_blobs(checkpoint, new_versions, prev_channel_values)

    assert len(blobs) == 1
    val = json.loads(zlib.decompress(base64.b64decode(blobs[0]["data"])))
    assert val == ["c"]


def test_serialize_channel_blobs_skips_unknown_channels():
    checkpoint = {"id": "ckpt3", "channel_values": {}}
    new_versions = ChannelVersions({"nonexistent": "1.0"})

    blobs, _ = _serialize_channel_blobs(checkpoint, new_versions, {})

    assert not blobs


def test_serialize_channel_blobs_list_unchanged_skips():
    checkpoint = {
        "id": "ckpt_unchanged",
        "channel_values": {"messages": ["a", "b"]},
    }
    new_versions = ChannelVersions({"messages": "2.0"})
    prev_channel_values = {"messages": ["a", "b"]}

    blobs, is_compaction = _serialize_channel_blobs(
        checkpoint, new_versions, prev_channel_values
    )

    assert not blobs
    assert not is_compaction


def test_serialize_channel_blobs_list_shrink_stores_full():
    import base64

    checkpoint = {
        "id": "ckpt5",
        "channel_values": {"messages": ["a"]},
    }
    new_versions = ChannelVersions({"messages": "4.0"})
    prev_channel_values = {"messages": ["a", "b", "c"]}

    blobs, is_compaction = _serialize_channel_blobs(
        checkpoint, new_versions, prev_channel_values
    )

    assert is_compaction
    assert len(blobs) == 1
    val = json.loads(zlib.decompress(base64.b64decode(blobs[0]["data"])))
    assert val == ["a"]


def test_serialize_channel_blobs_dict_channel_delta():
    import base64

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

    blobs, _ = _serialize_channel_blobs(checkpoint, new_versions, prev_channel_values)

    assert len(blobs) == 1
    assert blobs[0]["channel"] == "conversation_history"
    delta = json.loads(zlib.decompress(base64.b64decode(blobs[0]["data"])))
    assert delta == {"planner": ["msg3"]}


def test_serialize_channel_blobs_dict_unchanged_skips_blob():
    values = {"conversation_history": {"planner": ["msg1"], "executor": ["a"]}}
    checkpoint = {"id": "ckpt6", "channel_values": values}
    new_versions = ChannelVersions({"conversation_history": "2.0"})

    blobs, _ = _serialize_channel_blobs(checkpoint, new_versions, dict(values))

    assert not blobs


def test_serialize_channel_blobs_compaction_stores_full_value():
    import base64

    prev_channel_values = {
        "conversation_history": {"planner": ["msg1", "msg2", "msg3", "msg4", "msg5"]}
    }
    checkpoint = {
        "id": "ckpt7",
        "channel_values": {"conversation_history": {"planner": ["summary", "msg5"]}},
    }
    new_versions = ChannelVersions({"conversation_history": "3.0"})

    blobs, is_compaction = _serialize_channel_blobs(
        checkpoint, new_versions, prev_channel_values
    )

    assert is_compaction
    assert len(blobs) == 1
    assert blobs[0]["step_action"] == "compaction"
    val = json.loads(zlib.decompress(base64.b64decode(blobs[0]["data"])))
    assert val == {"planner": ["summary", "msg5"]}


def test_serialize_channel_blobs_dict_same_length_rewrite_is_compaction():
    prev_channel_values = {"conversation_history": {"planner": ["msg1", "msg2"]}}
    checkpoint = {
        "id": "ckpt9",
        "channel_values": {
            "conversation_history": {"planner": ["summary_a", "summary_b"]}
        },
    }
    new_versions = ChannelVersions({"conversation_history": "3.0"})

    blobs, is_compaction = _serialize_channel_blobs(
        checkpoint, new_versions, prev_channel_values
    )

    assert is_compaction
    assert len(blobs) == 1
    assert blobs[0]["step_action"] == "compaction"


def test_serialize_channel_blobs_force_rewrite_bypasses_delta():
    import base64

    checkpoint = {
        "id": "ckpt_force",
        "channel_values": {"messages": ["a", "b", "c"]},
    }
    new_versions = ChannelVersions({"messages": "2.0"})
    prev_channel_values = {"messages": ["a", "b"]}

    blobs, is_compaction = _serialize_channel_blobs(
        checkpoint,
        new_versions,
        prev_channel_values,
        force_rewrite=True,
    )

    assert not is_compaction
    assert len(blobs) == 1
    assert blobs[0]["step_action"] == "compaction"
    val = json.loads(zlib.decompress(base64.b64decode(blobs[0]["data"])))
    assert val == ["a", "b", "c"]


def test_serialize_channel_blobs_conversation_sends_delta():
    import base64

    checkpoint = {
        "id": "ckpt8",
        "channel_values": {"messages": ["a", "b", "c"]},
    }
    new_versions = ChannelVersions({"messages": "3.0"})

    blobs, is_compaction = _serialize_channel_blobs(
        checkpoint, new_versions, {"messages": ["a", "b"]}
    )

    assert not is_compaction
    assert len(blobs) == 1
    assert blobs[0]["step_action"] == "conversation"
    val = json.loads(zlib.decompress(base64.b64decode(blobs[0]["data"])))
    assert val == ["c"]


def _v6_uuid_for(unix_seconds: int, sub_second_100ns: int = 0) -> str:
    """Build a valid UUIDv6 whose embedded timestamp is unix_seconds (+ optional 100ns)."""
    gregorian_100ns = unix_seconds * 10_000_000 + sub_second_100ns + 0x01B21DD213814000
    time_high = (gregorian_100ns >> 28) & 0xFFFFFFFF
    time_mid = (gregorian_100ns >> 12) & 0xFFFF
    time_low = gregorian_100ns & 0x0FFF
    return f"{time_high:08x}-{time_mid:04x}-6{time_low:03x}-8000-000000000000"


def test_thread_started_at_from_id_decodes_v6():
    # 1_700_000_000 == 2023-11-14T22:13:20Z
    assert (
        _thread_started_at_from_id(_v6_uuid_for(1_700_000_000))
        == "2023-11-14T22:13:20+00:00"
    )


def test_thread_started_at_from_id_decodes_v7():
    # 48-bit millisecond timestamp: 1_700_000_000_000 ms
    value = (1_700_000_000_000 << 80) | (0x7 << 76) | (0x8 << 62)
    uuid = f"{value:032x}"
    dashed = f"{uuid[:8]}-{uuid[8:12]}-{uuid[12:16]}-{uuid[16:20]}-{uuid[20:]}"
    assert _thread_started_at_from_id(dashed) == "2023-11-14T22:13:20+00:00"


def test_thread_started_at_from_id_floors_to_second():
    # Sub-second 100ns component must be dropped so the marker never exceeds the
    # created_at Rails derives from the same id.
    assert (
        _thread_started_at_from_id(
            _v6_uuid_for(1_700_000_000, sub_second_100ns=9_999_999)
        )
        == "2023-11-14T22:13:20+00:00"
    )


def test_thread_started_at_from_id_returns_none_for_non_time_uuid():
    # Version 4 (random) UUID embeds no timestamp.
    assert _thread_started_at_from_id("f47ac10b-58cc-4372-a567-0e02b2c3d479") is None


def test_thread_started_at_from_id_returns_none_for_malformed():
    assert _thread_started_at_from_id("not-a-uuid") is None


def test_serialize_all_channels_full_reseeds_and_drops_non_status_scalars():
    """Group-start snapshot: list/dict channels and the status scalar are re-seeded as
    full 'compaction' blobs (versions from the checkpoint); other scalars are dropped."""
    import base64

    checkpoint = {
        "channel_values": {
            "messages": ["a", "b"],
            "conversation_history": {"planner": [1, 2]},
            "status": "Execution",
            "plan_step": 3,  # non-status scalar -> dropped
        },
        "channel_versions": {
            "messages": "2",
            "conversation_history": "1",
            "status": "1",
            "plan_step": "1",
        },
    }

    blobs = _serialize_all_channels_full(checkpoint)

    by_channel = {b["channel"]: b for b in blobs}
    assert set(by_channel) == {"messages", "conversation_history", "status"}
    assert all(b["step_action"] == "compaction" for b in blobs)
    assert by_channel["messages"]["version"] == "2"
    assert json.loads(
        zlib.decompress(base64.b64decode(by_channel["messages"]["data"]))
    ) == ["a", "b"]


def test_serialize_all_channels_full_version_defaults_to_empty_when_absent():
    """A channel present in channel_values but missing from channel_versions falls back to an empty version string
    rather than raising or dropping the channel."""
    checkpoint = {
        "channel_values": {"messages": ["a"]},
        "channel_versions": {},
    }

    blobs = _serialize_all_channels_full(checkpoint)

    assert len(blobs) == 1
    assert blobs[0]["channel"] == "messages"
    assert blobs[0]["version"] == ""


@pytest.mark.asyncio
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_aput_sends_full_checkpoint_and_channel_blobs(
    _mock_duo_workflow_metrics,
    incremental_enabled,
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
    checkpoint["channel_versions"] = {
        "conversation_history": "1.0",
        "messages": "2.1",
        "status": "1.0",
    }

    new_versions = ChannelVersions({"messages": "2.1"})
    http_client.apost.return_value = GitLabHttpResponse(status_code=200, body={})

    await gitlab_workflow.aput(config, checkpoint, checkpoint_metadata, new_versions)

    post_call_body = json.loads(http_client.apost.call_args[1]["body"])

    # Full checkpoint must still be present (Phase 1 — backward compatible reads)
    assert post_call_body["compressed_checkpoint"] == compress_checkpoint(checkpoint)

    # First checkpoint starts a self-contained group (issue 605653): every
    # reconstructable channel is re-seeded as a full compaction snapshot, not just
    # the channel in new_versions.
    blobs = post_call_body["channel_blobs"]
    by_channel = {b["channel"]: b for b in blobs}
    assert set(by_channel) == {"conversation_history", "messages", "status"}
    assert all(b["step_action"] == "compaction" for b in blobs)
    assert by_channel["messages"]["version"] == "2.1"


@pytest.mark.asyncio
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_aput_accumulates_list_deltas_across_calls(
    _mock_duo_workflow_metrics,
    incremental_enabled,
    gitlab_workflow,
    http_client,
    checkpoint_data,
    checkpoint_metadata,
):
    """Aput tracks previous channel values between calls so each blob contains only the items appended since the
    previous checkpoint."""
    import base64

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
    # First checkpoint starts a self-contained group, so messages is a full snapshot.
    body1 = json.loads(http_client.apost.call_args[1]["body"])
    messages1 = next(b for b in body1["channel_blobs"] if b["channel"] == "messages")
    val1 = json.loads(zlib.decompress(base64.b64decode(messages1["data"])))
    assert val1 == ["a", "b"]

    checkpoint["id"] = "ckpt-2"
    checkpoint["channel_values"]["messages"] = ["a", "b", "c", "d"]
    await gitlab_workflow.aput(
        {"configurable": {"checkpoint_id": "ckpt-1"}},
        checkpoint,
        checkpoint_metadata,
        ChannelVersions({"messages": "2.0"}),
    )
    # Not a new group: only the appended tail is sent.
    body2 = json.loads(http_client.apost.call_args[1]["body"])
    messages2 = next(b for b in body2["channel_blobs"] if b["channel"] == "messages")
    val2 = json.loads(zlib.decompress(base64.b64decode(messages2["data"])))
    assert val2 == ["c", "d"]


@pytest.mark.asyncio
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_aput_increments_thread_id_on_compaction(
    _mock_duo_workflow_metrics,
    incremental_enabled,
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
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_aput_compaction_reseeds_all_channels_as_full_snapshot(
    _mock_duo_workflow_metrics,
    incremental_enabled,
    gitlab_workflow,
    http_client,
    checkpoint_data,
    checkpoint_metadata,
):
    """A compaction starts a self-contained group: EVERY channel is re-seeded as a full
    'compaction' snapshot, even a channel that did not change in the compacting step, so
    the new group reconstructs without the previous group or the checkpoint header
    (issue 605653)."""
    import base64

    checkpoint = checkpoint_data[0]["checkpoint"]
    http_client.apost.return_value = GitLabHttpResponse(status_code=200, body={})
    # Group 0: both channels seeded.
    checkpoint["id"] = "ckpt-1"
    checkpoint["channel_values"]["messages"] = ["a", "b"]
    await gitlab_workflow.aput(
        {"configurable": {"checkpoint_id": None}},
        checkpoint,
        checkpoint_metadata,
        ChannelVersions({"messages": "1.0", "conversation_history": "1.0"}),
    )

    # Compaction: messages shrinks; conversation_history is unchanged this step.
    checkpoint["id"] = "ckpt-2"
    checkpoint["channel_values"]["messages"] = ["summary"]
    checkpoint["channel_versions"] = {"messages": "2.0", "conversation_history": "1.0"}
    await gitlab_workflow.aput(
        {"configurable": {"checkpoint_id": "ckpt-1"}},
        checkpoint,
        checkpoint_metadata,
        ChannelVersions({"messages": "2.0"}),
    )

    body = json.loads(http_client.apost.call_args[1]["body"])
    assert body["current_thread"] == 1
    by_channel = {b["channel"]: b for b in body["channel_blobs"]}
    # Both channels re-seeded, not just the one that compacted.
    assert set(by_channel) == {"messages", "conversation_history"}
    assert all(b["step_action"] == "compaction" for b in body["channel_blobs"])
    assert json.loads(
        zlib.decompress(base64.b64decode(by_channel["messages"]["data"]))
    ) == ["summary"]
    # The unchanged channel carries its full value, not an empty delta.
    ch_val = json.loads(
        zlib.decompress(base64.b64decode(by_channel["conversation_history"]["data"]))
    )
    assert len(ch_val["planner"]) == 3


@pytest.mark.asyncio
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_aput_pins_current_thread_started_at_to_group_start(
    _mock_duo_workflow_metrics,
    incremental_enabled,
    gitlab_workflow,
    http_client,
    checkpoint_data,
    checkpoint_metadata,
):
    """current_thread_started_at pins to the group's first checkpoint and re-pins on compaction."""
    checkpoint = checkpoint_data[0]["checkpoint"]
    http_client.apost.return_value = GitLabHttpResponse(status_code=200, body={})

    id_start = _v6_uuid_for(1_700_000_000)
    id_mid = _v6_uuid_for(1_700_000_005)
    id_compaction = _v6_uuid_for(1_700_000_010)

    # First checkpoint of the group — marker takes its start time.
    checkpoint["id"] = id_start
    checkpoint["channel_values"]["messages"] = ["a", "b"]
    await gitlab_workflow.aput(
        {"configurable": {"checkpoint_id": None}},
        checkpoint,
        checkpoint_metadata,
        ChannelVersions({"messages": "1.0"}),
    )
    body = json.loads(http_client.apost.call_args[1]["body"])
    assert body["current_thread_started_at"] == _thread_started_at_from_id(id_start)

    # Conversation step — marker stays pinned to the first checkpoint, not id_mid.
    checkpoint["id"] = id_mid
    checkpoint["channel_values"]["messages"] = ["a", "b", "c"]
    await gitlab_workflow.aput(
        {"configurable": {"checkpoint_id": id_start}},
        checkpoint,
        checkpoint_metadata,
        ChannelVersions({"messages": "2.0"}),
    )
    body = json.loads(http_client.apost.call_args[1]["body"])
    assert body["current_thread"] == 0
    assert body["current_thread_started_at"] == _thread_started_at_from_id(id_start)

    # Compaction (list shrank) starts a new group — marker re-pins to id_compaction.
    checkpoint["id"] = id_compaction
    checkpoint["channel_values"]["messages"] = ["summary"]
    await gitlab_workflow.aput(
        {"configurable": {"checkpoint_id": id_mid}},
        checkpoint,
        checkpoint_metadata,
        ChannelVersions({"messages": "3.0"}),
    )
    body = json.loads(http_client.apost.call_args[1]["body"])
    assert body["current_thread"] == 1
    assert body["current_thread_started_at"] == _thread_started_at_from_id(
        id_compaction
    )


@pytest.mark.asyncio
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_aput_repins_current_thread_started_at_on_stale_cache(
    _mock_duo_workflow_metrics,
    incremental_enabled,
    gitlab_workflow,
    http_client,
    checkpoint_data,
    checkpoint_metadata,
):
    """A stale cache (skipped parent) re-pins the marker to the stale checkpoint, not the original group start."""
    checkpoint = checkpoint_data[0]["checkpoint"]
    http_client.apost.return_value = GitLabHttpResponse(status_code=200, body={})

    id_start = _v6_uuid_for(1_700_000_000)
    id_stale = _v6_uuid_for(1_700_000_020)

    checkpoint["id"] = id_start
    checkpoint["channel_values"]["messages"] = ["a", "b"]
    await gitlab_workflow.aput(
        {"configurable": {"checkpoint_id": None}},
        checkpoint,
        checkpoint_metadata,
        ChannelVersions({"messages": "1.0"}),
    )

    # Wrong parent → stale cache → thread bumps and the marker re-pins to id_stale.
    checkpoint["id"] = id_stale
    checkpoint["channel_values"]["messages"] = ["a", "b", "c"]
    await gitlab_workflow.aput(
        {"configurable": {"checkpoint_id": "ckpt-unknown"}},
        checkpoint,
        checkpoint_metadata,
        ChannelVersions({"messages": "2.0"}),
    )
    body = json.loads(http_client.apost.call_args[1]["body"])
    assert body["current_thread"] == 1
    assert body["current_thread_started_at"] == _thread_started_at_from_id(id_stale)


@pytest.mark.asyncio
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_aput_resets_cache_on_stale_checkpoint_id(
    _mock_duo_workflow_metrics,
    incremental_enabled,
    gitlab_workflow,
    http_client,
    checkpoint_data,
    checkpoint_metadata,
):
    """When parent checkpoint_id doesn't match the cached id, cache is reset and a warning is logged."""
    import base64

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

    # Stale cache starts a new self-contained group: the full list is stored for
    # messages (no delta), not just ["c"].
    body = json.loads(http_client.apost.call_args[1]["body"])
    messages = next(b for b in body["channel_blobs"] if b["channel"] == "messages")
    val = json.loads(zlib.decompress(base64.b64decode(messages["data"])))
    assert val == ["a", "b", "c"]
    assert messages["step_action"] == "compaction"
    # Thread must be bumped so Rails starts reconstruction from this checkpoint
    assert body["current_thread"] == 1


@pytest.mark.asyncio
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_aput_omits_channel_blobs_when_incremental_disabled(
    _mock_duo_workflow_metrics,
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
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_aget_tuple_hydrates_current_thread_from_response(
    _mock_duo_workflow_metrics,
    incremental_enabled,
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
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_aget_tuple_hydrates_current_thread_started_at(
    _mock_duo_workflow_metrics,
    incremental_enabled,
    gitlab_workflow,
    http_client,
    compressed_checkpoint_data,
):
    """On resume the marker is restored so a post-restart aput doesn't re-pin it mid-group."""
    compressed_checkpoint_data[0]["current_thread_started_at"] = (
        "2026-07-08T10:00:00+00:00"
    )
    http_client.aget.return_value = GitLabHttpResponse(
        status_code=200, body=compressed_checkpoint_data
    )

    config = {"configurable": {"thread_id": "1234", "checkpoint_id": "5678"}}
    result = await gitlab_workflow.aget_tuple(config)

    assert result is not None
    assert gitlab_workflow._current_thread_started_at == "2026-07-08T10:00:00+00:00"


@pytest.mark.asyncio
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_aget_tuple_hydrates_current_thread_on_latest_fetch_path(
    _mock_duo_workflow_metrics,
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
        "incremental_checkpoints_enabled": True,
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
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_aget_tuple_hydration_tolerates_missing_current_thread(
    _mock_duo_workflow_metrics,
    incremental_enabled,
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
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_aget_tuple_hydration_tolerates_malformed_current_thread(
    _mock_duo_workflow_metrics,
    incremental_enabled,
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
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_aget_tuple_skips_hydration_when_incremental_disabled(
    _mock_duo_workflow_metrics,
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
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_aput_after_hydration_chains_delta_without_stale_cache_reset(
    _mock_duo_workflow_metrics,
    incremental_enabled,
    gitlab_workflow,
    http_client,
    compressed_checkpoint_data,
    checkpoint_metadata,
):
    """End-to-end: simulate a restart by hydrating then writing — must reuse server thread, no current_thread bump."""
    import base64

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
    val = json.loads(zlib.decompress(base64.b64decode(blob["data"])))
    assert val == ["new"]


# A GraphQL-format checkpoint dict as cached in WorkflowConfig["latest_checkpoint"].
_GQL_LATEST_CHECKPOINT = {
    "threadTs": "gql-ckpt",
    "parentTs": None,
    "checkpoint": json.dumps({"id": "gql-ckpt", "channel_values": {"x": [1]}}),
    "metadata": "{}",
    "currentThread": 4,
    "currentThreadStartedAt": "2026-07-08T10:00:00+00:00",
}


def test_decode_graphql_checkpoint_is_side_effect_free(gitlab_workflow):
    """Decoding a checkpoint must never touch the incremental write cache.

    Regression guard for the decode/hydration coupling: a caller decoding an arbitrary checkpoint (e.g. the
    pre-rollback tip during stop-recovery) must not repoint the delta baseline — a wrong baseline whose id happens to
    match the next aput's parent would silently emit wrong deltas / a rewound current_thread. Hydration is owned
    exclusively by the ``aget_tuple`` fetch paths.
    """
    gitlab_workflow._workflow_config["incremental_checkpoints_enabled"] = True
    # Seed the cache with sentinels so the assertion is "unchanged", not merely "still default".
    gitlab_workflow._prev_checkpoint_id = "sentinel-ckpt"
    gitlab_workflow._prev_channel_values = {"sentinel": True}
    gitlab_workflow._current_thread = 9
    gitlab_workflow._current_thread_started_at = "2026-01-01T00:00:00+00:00"

    result = gitlab_workflow.decode_graphql_checkpoint(dict(_GQL_LATEST_CHECKPOINT))

    assert result is not None
    assert result.checkpoint["channel_values"] == {"x": [1]}
    assert gitlab_workflow._prev_checkpoint_id == "sentinel-ckpt"
    assert gitlab_workflow._prev_channel_values == {"sentinel": True}
    assert gitlab_workflow._current_thread == 9
    assert gitlab_workflow._current_thread_started_at == "2026-01-01T00:00:00+00:00"


@pytest.mark.asyncio
async def test_aget_tuple_hydrates_from_cached_latest_checkpoint(
    http_client,
    workflow_id,
    workflow_type,
    workflow_config,
):
    """Hydration must fire on the cached-latest-checkpoint path (no checkpoint_id; latest_checkpoint in config).

    Decoding itself is side-effect free (see test_decode_graphql_checkpoint_is_side_effect_free), so this path — where the
    decoded checkpoint IS the resume baseline — hydrates explicitly, with no HTTP fetch.
    """
    workflow_config["incremental_checkpoints_enabled"] = True
    workflow_config["latest_checkpoint"] = dict(_GQL_LATEST_CHECKPOINT)
    gitlab_workflow = GitLabWorkflow(
        http_client,
        workflow_id,
        workflow_type,
        workflow_config,
    )

    result = await gitlab_workflow.aget_tuple(
        {"configurable": {"thread_id": workflow_id}}
    )

    assert result is not None
    http_client.aget.assert_not_called()  # served from the session-start cache
    assert gitlab_workflow._current_thread == 4
    assert gitlab_workflow._prev_checkpoint_id == "gql-ckpt"
    assert gitlab_workflow._prev_channel_values == {"x": [1]}
    assert gitlab_workflow._current_thread_started_at == "2026-07-08T10:00:00+00:00"


def _make_gl_checkpoint(thread_ts, status):
    """Build a compressed GitLab checkpoint dict whose channel_values carry the given status.

    Deliberately unannotated: ``compress_checkpoint`` expects langgraph's full Checkpoint TypedDict, whose remaining
    keys are irrelevant here — the same partial-dict convention other checkpoint-building helpers in this file use.
    """
    return {
        "thread_ts": thread_ts,
        "parent_ts": None,
        "compressed_checkpoint": compress_checkpoint(
            {"id": thread_ts, "channel_values": {"status": status}}
        ),
        "metadata": {},
    }


def _paginated_checkpoints_aget(pages: list[list[dict]]):
    """Return an aget double serving `pages` (newest-first) with GitLab REST pagination headers.

    Also returns the list of query-param dicts recorded per request, so tests can assert exactly which pages were
    fetched (and that no extra requests happened).
    """
    requested_params: list[dict] = []

    async def mock_aget(path, **_kwargs):
        query = {k: v[0] for k, v in parse_qs(urlparse(path).query).items()}
        requested_params.append(query)
        page_index = int(query["page"]) - 1
        if page_index >= len(pages):
            return GitLabHttpResponse(status_code=404, body=[], headers={})
        headers = {}
        if page_index + 1 < len(pages):
            headers["X-Next-Page"] = str(page_index + 2)
        return GitLabHttpResponse(
            status_code=200, body=pages[page_index], headers=headers
        )

    return mock_aget, requested_params


@pytest.fixture(name="paginated_checkpoint_pages")
def paginated_checkpoint_pages_fixture() -> list[list[dict]]:
    """Two pages of checkpoints, newest-first, with the only INPUT_REQUIRED boundary on page one."""
    return [
        [
            _make_gl_checkpoint("cp-4", WorkflowStatusEnum.EXECUTION),
            _make_gl_checkpoint("cp-3", WorkflowStatusEnum.INPUT_REQUIRED),
        ],
        [
            _make_gl_checkpoint("cp-2", WorkflowStatusEnum.PLANNING),
            _make_gl_checkpoint("cp-1", WorkflowStatusEnum.NOT_STARTED),
        ],
    ]


@pytest.mark.asyncio
async def test_iter_checkpoint_pages_stops_without_fetching_further_pages(
    gitlab_workflow, http_client, paginated_checkpoint_pages
):
    mock_aget, requested_params = _paginated_checkpoints_aget(
        paginated_checkpoint_pages
    )
    http_client.aget = mock_aget

    page_iterator = gitlab_workflow._iter_checkpoint_pages()
    first_page = await anext(page_iterator)
    await page_iterator.aclose()

    assert [cp["thread_ts"] for cp in first_page] == ["cp-4", "cp-3"]
    assert len(requested_params) == 1, (
        "X-Next-Page must only be followed if the consumer keeps iterating"
    )
    assert requested_params[0]["page"] == "1"


@pytest.mark.asyncio
async def test_iter_checkpoint_pages_follows_next_page_header(
    gitlab_workflow, http_client, paginated_checkpoint_pages
):
    mock_aget, requested_params = _paginated_checkpoints_aget(
        paginated_checkpoint_pages
    )
    http_client.aget = mock_aget

    collected = [
        cp["thread_ts"]
        async for page in gitlab_workflow._iter_checkpoint_pages(per_page=2)
        for cp in page
    ]

    assert collected == ["cp-4", "cp-3", "cp-2", "cp-1"]
    # Pages are requested in sequence via page/per_page params, stopping when
    # X-Next-Page is absent.
    assert [params["page"] for params in requested_params] == ["1", "2"]
    assert all(params["per_page"] == "2" for params in requested_params)
    assert all(params["accept_compressed"] == "true" for params in requested_params)


@pytest.mark.asyncio
async def test_checkpoints_reversed_applies_predicate(
    gitlab_workflow, http_client, paginated_checkpoint_pages
):
    mock_aget, _ = _paginated_checkpoints_aget(paginated_checkpoint_pages)
    http_client.aget = mock_aget

    matched = [
        checkpoint_tuple
        async for checkpoint_tuple in gitlab_workflow.checkpoints_reversed(
            matches=lambda cv: (
                cv.get("status")
                in (WorkflowStatusEnum.INPUT_REQUIRED, WorkflowStatusEnum.NOT_STARTED)
            )
        )
    ]

    assert [
        checkpoint_tuple.config["configurable"]["checkpoint_id"]
        for checkpoint_tuple in matched
    ] == ["cp-3", "cp-1"]
    assert all(isinstance(t, CheckpointTuple) for t in matched)


@pytest.mark.asyncio
async def test_checkpoints_reversed_no_predicate_yields_all(
    gitlab_workflow, http_client, paginated_checkpoint_pages
):
    mock_aget, _ = _paginated_checkpoints_aget(paginated_checkpoint_pages)
    http_client.aget = mock_aget

    results = [
        checkpoint_tuple
        async for checkpoint_tuple in gitlab_workflow.checkpoints_reversed()
    ]

    assert [
        checkpoint_tuple.config["configurable"]["checkpoint_id"]
        for checkpoint_tuple in results
    ] == ["cp-4", "cp-3", "cp-2", "cp-1"]


@pytest.mark.asyncio
async def test_checkpoints_reversed_stops_after_first_match_no_extra_requests(
    gitlab_workflow, http_client, paginated_checkpoint_pages
):
    """Early-exit (as _resolve_stop_recovery does) must not request pages beyond the match."""
    mock_aget, requested_params = _paginated_checkpoints_aget(
        paginated_checkpoint_pages
    )
    http_client.aget = mock_aget

    checkpoint_iterator = gitlab_workflow.checkpoints_reversed(
        matches=lambda cv: cv.get("status") == WorkflowStatusEnum.INPUT_REQUIRED
    )
    boundary = await anext(checkpoint_iterator)
    await checkpoint_iterator.aclose()

    assert boundary.config["configurable"]["checkpoint_id"] == "cp-3"
    assert len(requested_params) == 1, (
        "no HTTP requests may be issued beyond the page containing the match"
    )


@pytest.mark.asyncio
async def test_iter_checkpoint_pages_stops_on_error_mid_pagination(
    gitlab_workflow, http_client, paginated_checkpoint_pages
):
    """An HTTP error on page 2 must stop iteration after yielding page 1; no further requests are made."""
    call_count = 0

    async def mock_aget(path, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return GitLabHttpResponse(
                status_code=200,
                body=paginated_checkpoint_pages[0],
                headers={"X-Next-Page": "2"},
            )
        return GitLabHttpResponse(status_code=500, body=None, headers={})

    http_client.aget = mock_aget

    pages = [page async for page in gitlab_workflow._iter_checkpoint_pages()]

    assert len(pages) == 1
    assert [cp["thread_ts"] for cp in pages[0]] == ["cp-4", "cp-3"]
    assert call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_code,body",
    [
        (404, [{"thread_ts": "cp-1"}]),
        (200, None),
        (200, []),
    ],
    ids=["non_success_status", "no_body", "empty_body"],
)
async def test_iter_checkpoint_pages_stops_on_failed_or_empty_response(
    gitlab_workflow, http_client, status_code, body
):
    """_iter_checkpoint_pages must stop iteration when the response is not successful or has no body."""

    async def mock_aget(path, **_kwargs):
        return GitLabHttpResponse(status_code=status_code, body=body, headers={})

    http_client.aget = mock_aget

    pages = [page async for page in gitlab_workflow._iter_checkpoint_pages()]

    assert pages == [], (
        "no pages should be yielded when the response is not successful or has no body"
    )


@pytest.mark.asyncio
async def test_checkpoints_reversed_skips_malformed_checkpoint_and_continues(
    gitlab_workflow, http_client
):
    """A malformed checkpoint (uncompress_checkpoint raises ValueError) must be skipped and logged; iteration must
    continue to yield the remaining valid checkpoints rather than aborting."""
    valid_cp = _make_gl_checkpoint("cp-valid", WorkflowStatusEnum.EXECUTION)
    malformed_cp = {
        "thread_ts": "cp-malformed",
        "parent_ts": None,
        # Invalid base64/zlib data — uncompress_checkpoint will raise ValueError.
        "compressed_checkpoint": "!!!not-valid-base64!!!",
        "metadata": {},
    }
    pages = [[malformed_cp, valid_cp]]
    mock_aget, _ = _paginated_checkpoints_aget(pages)
    http_client.aget = mock_aget

    with patch(
        "duo_workflow_service.checkpointer.gitlab_workflow.log_exception"
    ) as mock_log_exception:
        results = [
            checkpoint_tuple
            async for checkpoint_tuple in gitlab_workflow.checkpoints_reversed()
        ]

    assert [
        checkpoint_tuple.config["configurable"]["checkpoint_id"]
        for checkpoint_tuple in results
    ] == ["cp-valid"], "only the valid checkpoint should be yielded"
    mock_log_exception.assert_called_once()
    _, kwargs = mock_log_exception.call_args
    assert kwargs.get("extra", {}).get("context") == "Skipping malformed checkpoint"


@pytest.mark.asyncio
async def test_get_initial_status_event_stopped_returns_stop_recovery(
    http_client, workflow_id, workflow_type
):
    workflow_config = {
        "first_checkpoint": {"checkpoint": "{}"},
        "latest_checkpoint": None,
        "workflow_status": WorkflowStatusEnum.STOPPED,
        "agent_privileges_names": [],
        "pre_approved_agent_privileges_names": [],
        "mcp_enabled": False,
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

    status_event, event_property = await gitlab_workflow._get_initial_status_event(
        {"configurable": {}}
    )

    assert status_event == WorkflowStatusEventEnum.STOP_RECOVERY
    assert event_property == EventPropertyEnum.WORKFLOW_RESUME_BY_USER
    # Pure detection: no checkpoint/HTTP access at all.
    http_client.aget.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_event,expected_rails_event",
    [
        (WorkflowStatusEventEnum.STOP_RECOVERY, WorkflowStatusEventEnum.RETRY),
        (WorkflowStatusEventEnum.RESUME, WorkflowStatusEventEnum.RESUME),
    ],
    ids=["stop_recovery_translated_to_retry", "regular_event_passed_through"],
)
async def test_update_workflow_status_translates_stop_recovery_to_retry(
    gitlab_workflow, workflow_id, status_event, expected_rails_event
):
    gitlab_workflow._status_handler = AsyncMock()

    await gitlab_workflow._update_workflow_status(status_event)

    gitlab_workflow._status_handler.update_workflow_status.assert_called_once_with(
        workflow_id, expected_rails_event
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_code,body",
    [
        (404, []),
        (200, None),
        (200, []),
    ],
    ids=["non_success", "no_body", "empty_body"],
)
async def test_checkpoints_reversed_yields_nothing_on_failed_or_empty_response(
    gitlab_workflow, http_client, status_code, body
):
    """checkpoints_reversed must yield nothing when the underlying HTTP response is non-success or empty."""

    async def mock_aget(path, **_kwargs):
        return GitLabHttpResponse(status_code=status_code, body=body, headers={})

    http_client.aget = mock_aget

    results = [t async for t in gitlab_workflow.checkpoints_reversed()]

    assert results == []


@pytest.mark.asyncio
@patch("duo_workflow_service.checkpointer.gitlab_workflow.duo_workflow_metrics")
async def test_workflow_context_manager_stop_recovery(
    mock_duo_workflow_metrics,
    http_client_for_retry,
    workflow_id,
    workflow_type,
    internal_event_client: Mock,
):
    """A stopped workflow enters as STOP_RECOVERY: Rails receives the wire-valid `retry` event and the session is
    tracked with the same labels the old catch-all RETRY branch used."""
    workflow_config: dict[str, Any] = {
        "first_checkpoint": {"checkpoint": "{}"},
        "latest_checkpoint": None,
        "workflow_status": WorkflowStatusEnum.STOPPED,
        "agent_privileges_names": [],
        "pre_approved_agent_privileges_names": [],
        "mcp_enabled": False,
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
        assert workflow.initial_status_event == WorkflowStatusEventEnum.STOP_RECOVERY

    # Rails only ever sees `retry` — never the DWS-internal `stop_recovery`.
    http_client_for_retry.apatch.assert_called_once_with(
        path=f"/api/v4/ai/duo_workflows/workflows/{workflow_id}",
        body=json.dumps({"status_event": WorkflowStatusEventEnum.RETRY.value}),
        parse_json=True,
    )

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
        ]
    )
    mock_duo_workflow_metrics.count_agent_platform_session_retry.assert_called_once_with(
        flow_type=workflow_type.value,
    )
