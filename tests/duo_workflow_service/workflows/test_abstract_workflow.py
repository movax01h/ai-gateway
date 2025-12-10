import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from contract import contract_pb2
from duo_workflow_service.tools import UNTRUSTED_MCP_WARNING
from duo_workflow_service.workflows.abstract_workflow import (
    AbstractWorkflow,
    TraceableException,
)
from duo_workflow_service.workflows.type_definitions import (
    AIO_CANCEL_STOP_WORKFLOW_REQUEST,
)
from lib.internal_events import InternalEventAdditionalProperties
from lib.internal_events.event_enum import CategoryEnum, EventEnum


# Concrete implementation for testing
class MockGraph:
    async def astream(self, input, config, stream_mode):
        yield "updates", {"step1": {"key": "value"}}


class MockWorkflow(AbstractWorkflow):
    def _compile(self, goal, tools_registry, checkpointer):
        return MockGraph()

    def get_workflow_state(self, goal):
        return {"goal": goal, "state": "initial"}

    async def _handle_workflow_failure(self, error, compiled_graph, graph_config):
        print(error)

    def log_workflow_elements(self, element):
        print(element)


@pytest.fixture(autouse=True)
def prepare_container(mock_duo_workflow_service_container):
    pass


@pytest.fixture(name="workflow")
def workflow_fixture(user):
    workflow_id = "test-workflow-id"
    metadata = {
        "extended_logging": True,
        "git_url": "https://example.com",
        "git_sha": "abc123",
    }
    workflow_type = CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT
    return MockWorkflow(
        workflow_id,
        metadata,
        workflow_type,
        user,
    )


@pytest.fixture(name="mock_project")
def mock_project_fixture():
    return {
        "id": MagicMock(),
        "description": MagicMock(),
        "name": MagicMock(),
        "http_url_to_repo": MagicMock(),
        "web_url": "https://example.com/project",
    }


@pytest.fixture(name="mock_namespace")
def mock_namespace_fixture():
    return {
        "id": MagicMock(),
        "description": MagicMock(),
        "name": MagicMock(),
        "web_url": "https://example.com/group",
    }


@pytest.fixture
def mock_action():
    return contract_pb2.Action()


@pytest.mark.asyncio
async def test_init(user):
    # Test initialization
    workflow_id = "test-workflow-id"
    metadata = {"key": "value"}
    mcp_tools = [
        contract_pb2.McpTool(name="get_issue", description="Tool to get issue")
    ]
    user = MagicMock()
    workflow = MockWorkflow(
        workflow_id,
        metadata,
        CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        user,
        {},
        mcp_tools,
        user,
    )

    assert workflow._workflow_id == workflow_id
    assert workflow._workflow_metadata == metadata
    assert workflow.is_done is False
    assert len(workflow._mcp_tools) == 1
    tool = workflow._mcp_tools[0](metadata={})
    assert tool.name == "get_issue"
    assert tool.description == f"{UNTRUSTED_MCP_WARNING}\n\nTool to get issue"
    assert workflow._user == user


@pytest.mark.asyncio
async def test_get_from_outbox(workflow, mock_action):
    # Put an item in the outbox
    workflow._outbox.put_action(mock_action)

    # Get the item
    item = await workflow.get_from_outbox()

    assert item == mock_action
    assert workflow._outbox._queue.empty()


@pytest.mark.asyncio
async def test_fail_outbox_action(workflow, mock_action):
    workflow._outbox.put_action(mock_action)

    item = await workflow.get_from_outbox()
    workflow.fail_outbox_action(item.requestID, "Something went wrong")

    assert item.requestID not in workflow._outbox._action_response


@pytest.mark.asyncio
async def test_set_action_response(workflow):
    request_id = workflow._outbox.put_action(contract_pb2.Action())

    # Set action response that has the corresponding request ID
    workflow.set_action_response(
        contract_pb2.ClientEvent(
            actionResponse=contract_pb2.ActionResponse(
                response="the response", requestID=request_id
            )
        )
    )


@pytest.mark.asyncio
@patch(
    "duo_workflow_service.workflows.abstract_workflow.fetch_workflow_and_container_data"
)
@patch(
    "duo_workflow_service.workflows.abstract_workflow.convert_mcp_tools_to_langchain_tool_classes"
)
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow")
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry.configure")
@pytest.mark.parametrize("mcp_enabled", [True, False])
async def test_compile_and_run_graph(
    mock_tools_registry,
    mock_gitlab_workflow,
    mock_convert_mcp_tools,
    mock_fetch_workflow,
    mock_project,
    mcp_enabled,
    user,
):
    # Setup mocks
    mock_tools_registry.return_value = MagicMock()
    mock_checkpointer = AsyncMock()
    mock_checkpointer.aget_tuple = AsyncMock(return_value=None)
    mock_checkpointer.initial_status_event = "START"
    mock_gitlab_workflow.return_value.__aenter__.return_value = mock_checkpointer
    mock_fetch_workflow.return_value = (
        mock_project,
        None,
        {
            "project_id": 1,
            "mcp_enabled": mcp_enabled,
            "agent_privileges_names": [],
            "pre_approved_agent_privileges_names": [],
            "allow_agent_to_request_user": False,
            "first_checkpoint": None,
            "workflow_status": "",
            "gitlab_host": "example.com",
        },
    )

    mcp_tool = MagicMock()
    mock_convert_mcp_tools.return_value = [mcp_tool]

    workflow = MockWorkflow(
        "id",
        {},
        CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        user,
        {},
        [mcp_tool],
    )

    # Run the method
    await workflow._compile_and_run_graph("Test goal")

    # Assertions
    assert workflow.is_done
    assert (
        workflow._session_url
        == "https://example.com/project/-/automate/agent-sessions/id"
    )
    mock_fetch_workflow.assert_called_once()
    mock_tools_registry.assert_called_once_with(
        outbox=workflow._outbox,
        workflow_config=workflow._workflow_config,
        gl_http_client=workflow._http_client,
        project=mock_project,
        mcp_tools=[mcp_tool] if mcp_enabled else [],
        language_server_version=None,
    )


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.abstract_workflow.log_exception")
async def test_cleanup(mock_log_exception, workflow):
    # Add items to queues
    workflow._outbox.put_action(contract_pb2.Action())

    # Run cleanup
    await workflow.cleanup(workflow._workflow_id)

    # Check queues are empty
    assert workflow._outbox._queue.empty()
    assert workflow.is_done is True
    mock_log_exception.assert_not_called()


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.abstract_workflow.log_exception")
async def test_cleanup_with_exception(mock_log_exception, workflow):
    workflow._outbox.put_action(contract_pb2.Action())
    # Make drain_queue raise an exception
    with patch.object(
        workflow._outbox._queue, "get_nowait", side_effect=RuntimeError("Test error")
    ):
        # Catch exception raised during cleanup
        try:
            await workflow.cleanup(workflow._workflow_id)
        except RuntimeError:
            pass

    # Two log_exception calls: one from _drain_queue and one from cleanup
    assert mock_log_exception.call_count == 1

    # Check the first call (from cleanup)
    first_call = mock_log_exception.call_args_list[0]
    args, kwargs = first_call
    assert isinstance(args[0], RuntimeError)
    assert kwargs["extra"]["workflow_id"] == workflow._workflow_id
    assert kwargs["extra"]["context"] == "Workflow cleanup failed"


def test_track_internal_event(workflow, internal_event_client: Mock):
    # Test tracking an internal event
    event_name = EventEnum.WORKFLOW_START
    workflow_type = CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT
    additional_properties = InternalEventAdditionalProperties()
    workflow._internal_event_client = internal_event_client

    workflow._track_internal_event(
        event_name=event_name,
        additional_properties=additional_properties,
        category=workflow_type,
    )

    internal_event_client.track_event.assert_called_once_with(
        event_name=event_name.value,
        additional_properties=additional_properties,
        category=workflow_type,
    )


@pytest.mark.asyncio
@patch(
    "duo_workflow_service.workflows.abstract_workflow.fetch_workflow_and_container_data"
)
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow")
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry.configure")
async def test_compile_and_run_graph_with_exception(
    mock_tools_registry,
    mock_gitlab_workflow,
    mock_fetch_workflow,
    workflow,
    mock_project,
):
    # Setup mocks to raise an exception
    mock_tools_registry.side_effect = Exception("Test exception")
    mock_fetch_workflow.return_value = (mock_project, None, {"project_id": 1})

    with pytest.raises(TraceableException) as exc_info:
        await workflow._compile_and_run_graph("Test goal")

    mock_tools_registry.assert_called_once()
    mock_fetch_workflow.assert_called_once()
    assert workflow.is_done
    assert isinstance(exc_info.value.original_exception, Exception)
    assert str(exc_info.value.original_exception) == "Test exception"


@pytest.mark.asyncio
@patch(
    "duo_workflow_service.workflows.abstract_workflow.fetch_workflow_and_container_data"
)
@patch("duo_workflow_service.workflows.abstract_workflow.UserInterface")
async def test_compile_and_run_graph_with_cancellation_during_fetch(
    mock_user_interface,
    mock_fetch_workflow,
    workflow,
):
    from duo_workflow_service.entities.state import WorkflowStatusEnum

    mock_fetch_workflow.side_effect = asyncio.CancelledError(
        AIO_CANCEL_STOP_WORKFLOW_REQUEST
    )

    mock_notifier = AsyncMock()
    mock_user_interface.return_value = mock_notifier

    with pytest.raises(TraceableException) as exc_info:
        await workflow._compile_and_run_graph("Test goal")

    mock_fetch_workflow.assert_called_once()
    assert workflow.is_done
    assert isinstance(exc_info.value.original_exception, asyncio.CancelledError)
    assert str(exc_info.value.original_exception) == AIO_CANCEL_STOP_WORKFLOW_REQUEST

    mock_notifier.send_event.assert_called_once_with(
        type="values",
        state={"status": WorkflowStatusEnum.CANCELLED, "ui_chat_log": []},
        stream=workflow._stream,
    )


@pytest.mark.asyncio
# pylint: disable=direct-environment-variable-reference
@patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"})
# pylint: enable=direct-environment-variable-reference
@patch.object(MockWorkflow, "_compile_and_run_graph")
async def test_run_passes_correct_metadata_to_langsmith_extra(
    mock_compile_and_run_graph, workflow
):
    await workflow.run("Test goal")

    call_args = mock_compile_and_run_graph.call_args
    args, kwargs = call_args

    metadata = kwargs["langsmith_extra"]["metadata"]
    assert metadata["git_url"] == "https://example.com"
    assert metadata["git_sha"] == "abc123"
    assert metadata["workflow_type"] == CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT.value


@pytest.mark.asyncio
@patch(
    "duo_workflow_service.workflows.abstract_workflow.fetch_workflow_and_container_data"
)
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow")
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry.configure")
async def test_namespace_level_workflow(
    mock_tools_registry,
    mock_gitlab_workflow,
    mock_fetch_workflow,
    workflow,
    mock_namespace,
):
    # Setup mocks
    mock_tools_registry.return_value = MagicMock()
    mock_checkpointer = AsyncMock()
    mock_checkpointer.aget_tuple = AsyncMock(return_value=None)
    mock_checkpointer.initial_status_event = "START"
    mock_gitlab_workflow.return_value.__aenter__.return_value = mock_checkpointer
    mock_fetch_workflow.return_value = (
        None,
        mock_namespace,
        {
            "namespace_id": 1,
            "agent_privileges_names": [],
            "pre_approved_agent_privileges_names": [],
            "allow_agent_to_request_user": False,
            "mcp_enabled": False,
            "first_checkpoint": None,
            "workflow_status": "",
            "gitlab_host": "example.com",
        },
    )

    # Run the method
    with pytest.raises(TraceableException) as exc_info:
        await workflow._compile_and_run_graph("Test goal")

    # Assertions
    assert workflow.is_done
    assert workflow._session_url is None
    assert isinstance(exc_info.value.original_exception, Exception)
    assert (
        str(exc_info.value.original_exception)
        == "This workflow software_development does not support namespace-level workflow"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("env_var_value", "extended_logging", "expected_tracing_enabled"),
    [
        ("false", True, False),  # Explicitly disabled
        ("false", False, False),  # Explicitly disabled regardless of extended_logging
        ("true", True, True),  # Explicitly enabled with extended_logging
        ("true", False, False),  # Explicitly enabled but extended_logging is False
        ("", True, True),  # Not set, follows extended_logging
        ("", False, False),  # Not set, follows extended_logging
    ],
)
@patch("duo_workflow_service.workflows.abstract_workflow.tracing_context")
@patch.object(MockWorkflow, "_compile_and_run_graph")
async def test_tracing_enabled_based_on_env_and_extended_logging(
    mock_compile_and_run_graph,
    mock_tracing_context,
    env_var_value,
    extended_logging,
    expected_tracing_enabled,
    user,
):
    """Test that tracing is enabled/disabled based on LANGSMITH_TRACING_V2 and extended_logging."""
    workflow_id = "test-workflow-id"
    metadata = {
        "extended_logging": extended_logging,
        "git_url": "https://example.com",
        "git_sha": "abc123",
    }
    workflow_type = CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT
    workflow = MockWorkflow(workflow_id, metadata, workflow_type, user)

    with patch.dict(os.environ, {"LANGSMITH_TRACING_V2": env_var_value}, clear=False):
        await workflow.run("Test goal")

    # Verify tracing_context was called with the expected enabled value
    mock_tracing_context.assert_called_once()
    call_kwargs = mock_tracing_context.call_args[1]
    assert call_kwargs["enabled"] == expected_tracing_enabled


@pytest.mark.asyncio
@patch(
    "duo_workflow_service.workflows.abstract_workflow.fetch_workflow_and_container_data"
)
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow")
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry.configure")
@patch("duo_workflow_service.workflows.abstract_workflow.duo_workflow_metrics")
async def test_compile_and_run_graph_records_first_response_on_first_graph_update(
    mock_metrics,
    mock_tools_registry,
    mock_gitlab_workflow,
    mock_fetch_workflow,
    mock_project,
    user,
):
    """Test that _compile_and_run_graph records time to first response when graph is updated."""

    class MockGraphWithMessages:
        async def astream(self, input, config, stream_mode):
            yield "updates", {"step1": {"key": "value"}}
            yield "values", {"status": "running", "ui_chat_log": []}
            yield "messages", {"message": "second message"}

    mock_tools_registry.return_value = MagicMock()
    mock_checkpointer = AsyncMock()
    mock_checkpointer.aget_tuple = AsyncMock(return_value=None)
    mock_checkpointer.initial_status_event = "START"
    mock_gitlab_workflow.return_value.__aenter__.return_value = mock_checkpointer
    mock_fetch_workflow.return_value = (
        mock_project,
        None,
        {
            "project_id": 1,
            "agent_privileges_names": [],
            "pre_approved_agent_privileges_names": [],
            "allow_agent_to_request_user": False,
            "mcp_enabled": False,
            "first_checkpoint": None,
            "workflow_status": "",
            "gitlab_host": "example.com",
        },
    )

    workflow = MockWorkflow(
        "id",
        {},
        CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        user,
    )

    workflow._compile = MagicMock(return_value=MockGraphWithMessages())

    await workflow._compile_and_run_graph("Test goal")

    assert mock_metrics.record_time_to_first_response.call_count == 1
    assert workflow._first_response_metric_recorded is True
