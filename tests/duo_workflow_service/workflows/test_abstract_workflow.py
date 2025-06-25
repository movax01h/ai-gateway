import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from contract import contract_pb2
from duo_workflow_service.internal_events import InternalEventAdditionalProperties
from duo_workflow_service.internal_events.event_enum import CategoryEnum, EventEnum
from duo_workflow_service.llm_factory import AnthropicConfig, VertexConfig
from duo_workflow_service.workflows.abstract_workflow import (
    AbstractWorkflow,
    TraceableException,
)
from duo_workflow_service.workflows.chat import Workflow


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


@pytest.fixture
def workflow():
    workflow_id = "test-workflow-id"
    metadata = {
        "extended_logging": True,
        "git_url": "https://example.com",
        "git_sha": "abc123",
    }
    workflow_type = CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT
    context_elements = []
    return MockWorkflow(workflow_id, metadata, workflow_type, context_elements)


@pytest.fixture
def mock_project():
    return {
        "id": MagicMock(),
        "description": MagicMock(),
        "name": MagicMock(),
        "http_url_to_repo": MagicMock(),
        "web_url": "https://example.com/project",
    }


@pytest.mark.asyncio
async def test_init():
    # Test initialization
    workflow_id = "test-workflow-id"
    metadata = {"key": "value"}
    context_elements = [{"type": 1, "name": "test", "contents": "test content"}]
    mcp_tools = [
        contract_pb2.McpTool(name="get_issue", description="Tool to get issue")
    ]
    user = MagicMock()
    workflow = MockWorkflow(
        workflow_id,
        metadata,
        CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        context_elements,
        {},
        mcp_tools,
        user,
    )

    assert workflow._workflow_id == workflow_id
    assert workflow._workflow_metadata == metadata
    assert workflow._context_elements == context_elements
    assert workflow.is_done is False
    assert workflow._outbox.maxsize == 1
    assert workflow._inbox.maxsize == 1
    assert len(workflow._additional_tools) == 1
    tool = workflow._additional_tools[0]
    assert tool.name == "get_issue"
    assert tool.description == "Tool to get issue"
    assert workflow._user == user


@pytest.mark.asyncio
async def test_outbox_empty(workflow):
    await workflow._outbox.put("test_item")
    assert not workflow.outbox_empty()

    item = await workflow.get_from_outbox()

    assert item == "test_item"
    assert workflow.outbox_empty()


@pytest.mark.asyncio
async def test_get_from_outbox(workflow):
    # Put an item in the outbox
    await workflow._outbox.put("test_item")

    # Get the item
    item = await workflow.get_from_outbox()

    assert item == "test_item"
    assert workflow._outbox.empty()


@pytest.mark.asyncio
async def test_get_from_streaming_outbox(workflow):
    await workflow._streaming_outbox.put("test_item")

    item = workflow.get_from_streaming_outbox()

    assert item == "test_item"
    assert workflow._streaming_outbox.empty()


@pytest.mark.asyncio
async def test_add_to_inbox(workflow):
    # Create a mock event
    mock_event = MagicMock()

    # Add to inbox
    workflow.add_to_inbox(mock_event)

    # Check if it was added
    assert workflow._inbox.qsize() == 1
    assert await workflow._inbox.get() == mock_event


@pytest.mark.asyncio
@patch(
    "duo_workflow_service.workflows.abstract_workflow.fetch_project_data_with_workflow_id"
)
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow")
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry.configure")
async def test_compile_and_run_graph(
    mock_tools_registry,
    mock_gitlab_workflow,
    mock_fetch_project,
    workflow,
    mock_project,
):
    # Setup mocks
    mock_tools_registry.return_value = MagicMock()
    mock_checkpointer = AsyncMock()
    mock_checkpointer.aget_tuple = AsyncMock(return_value=None)
    mock_checkpointer.initial_status_event = "START"
    mock_gitlab_workflow.return_value.__aenter__.return_value = mock_checkpointer
    mock_fetch_project.return_value = (mock_project, {"project_id": 1})

    # Run the method
    await workflow._compile_and_run_graph("Test goal")

    # Assertions
    assert workflow.is_done
    mock_tools_registry.assert_called_once()
    mock_fetch_project.assert_called_once()


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.abstract_workflow.log_exception")
async def test_cleanup(mock_log_exception, workflow):
    # Add items to queues
    await workflow._outbox.put("outbox_item")
    await workflow._inbox.put("inbox_item")

    # Run cleanup
    await workflow.cleanup(workflow._workflow_id)

    # Check queues are empty
    assert workflow._outbox.empty()
    assert workflow._inbox.empty()
    assert workflow.is_done is True
    mock_log_exception.assert_not_called()


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.abstract_workflow.log_exception")
async def test_cleanup_with_exception(mock_log_exception, workflow):
    await workflow._outbox.put("test_item")
    # Make drain_queue raise an exception
    with patch.object(
        workflow._outbox, "get_nowait", side_effect=RuntimeError("Test error")
    ):
        # Catch exception raised during cleanup
        try:
            await workflow.cleanup(workflow._workflow_id)
        except RuntimeError:
            pass

    # Two log_exception calls: one from _drain_queue and one from cleanup
    assert mock_log_exception.call_count == 2

    # Check the first call (from _drain_queue)
    first_call = mock_log_exception.call_args_list[0]
    args, kwargs = first_call
    assert isinstance(args[0], RuntimeError)
    assert str(args[0]) == "Test error"
    assert kwargs["extra"]["workflow_id"] == workflow._workflow_id
    assert kwargs["extra"]["context"] == "Error draining outbox queue"

    # Check the second call (from cleanup)
    second_call = mock_log_exception.call_args_list[1]
    args, kwargs = second_call
    assert isinstance(args[0], RuntimeError)
    assert kwargs["extra"]["workflow_id"] == workflow._workflow_id
    assert kwargs["extra"]["context"] == "Workflow cleanup failed"


@patch(
    "duo_workflow_service.workflows.abstract_workflow.DuoWorkflowInternalEvent.track_event"
)
def test_track_internal_event(mock_track_event, workflow):
    # Test tracking an internal event
    event_name = EventEnum.WORKFLOW_START
    workflow_type = CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT
    additional_properties = InternalEventAdditionalProperties()

    workflow._track_internal_event(
        event_name=event_name,
        additional_properties=additional_properties,
        category=workflow_type,
    )

    mock_track_event.assert_called_once_with(
        event_name=event_name.value,
        additional_properties=additional_properties,
        category=workflow_type,
    )


@pytest.mark.asyncio
@patch(
    "duo_workflow_service.workflows.abstract_workflow.fetch_project_data_with_workflow_id"
)
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow")
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry.configure")
async def test_compile_and_run_graph_with_exception(
    mock_tools_registry,
    mock_gitlab_workflow,
    mock_fetch_project,
    workflow,
    mock_project,
):
    # Setup mocks to raise an exception
    mock_tools_registry.side_effect = Exception("Test exception")
    mock_fetch_project.return_value = (mock_project, {"project_id": 1})
    workflow._inbox.get = AsyncMock(
        return_value=MagicMock(actionResponse=MagicMock(requestID="", response=""))
    )
    workflow._inbox.task_done = AsyncMock()

    with pytest.raises(TraceableException) as exc_info:
        await workflow._compile_and_run_graph("Test goal")

    mock_tools_registry.assert_called_once()
    mock_fetch_project.assert_called_once()
    assert workflow.is_done
    assert isinstance(exc_info.value.original_exception, Exception)
    assert str(exc_info.value.original_exception) == "Test exception"


@pytest.mark.asyncio
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
@patch.dict(os.environ, {"DUO_WORKFLOW__VERTEX_PROJECT_ID": ""})
async def test_workflow_get_chat_model_without_vertex():
    """Test _get_model_config returns standard model when VERTEX_PROJECT_ID is not set."""
    workflow = Workflow(
        "123",
        {},
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )

    anthropic_model: AnthropicConfig = workflow._get_model_config()
    assert anthropic_model.model_name == "claude-sonnet-4-20250514"


@pytest.mark.asyncio
@patch.dict(
    os.environ,
    {
        "DUO_WORKFLOW__VERTEX_PROJECT_ID": "123",
        "DUO_WORKFLOW__VERTEX_LOCATION": "us-central",
    },
)
async def test_workflow_get_chat_model_with_vertex():
    """Test _get_model_config returns standard model when VERTEX_PROJECT_ID is not set."""
    workflow = Workflow(
        "123",
        {},
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )

    vertex_model: VertexConfig = workflow._get_model_config()
    assert vertex_model.model_name == "claude-sonnet-4@20250514"
