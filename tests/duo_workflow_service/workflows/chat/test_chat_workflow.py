from datetime import datetime
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from contract.contract_pb2 import ContextElement, ContextElementType
from duo_workflow_service.agents.prompts import CHAT_SYSTEM_PROMPT
from duo_workflow_service.components.tools_registry import ToolsRegistry
from duo_workflow_service.entities import (
    MessageTypeEnum,
    ToolStatus,
    WorkflowStatusEnum,
)
from duo_workflow_service.entities.state import ChatWorkflowState
from duo_workflow_service.internal_events.event_enum import CategoryEnum
from duo_workflow_service.workflows.chat.workflow import (
    AGENT_NAME,
    CHAT_MUTATION_TOOLS,
    CHAT_READ_ONLY_TOOLS,
    Routes,
    Workflow,
)


@pytest.fixture
def mock_state():
    return {
        "plan": {"steps": []},
        "status": WorkflowStatusEnum.NOT_STARTED,
        "conversation_history": {},
        "ui_chat_log": [],
        "last_human_input": None,
        "context_elements": [],
    }


@pytest.fixture
def mock_tools_registry():
    mock_registry = MagicMock(spec=ToolsRegistry)
    mock_registry.get_batch = Mock(return_value=[Mock(name="test_tool")])
    mock_registry.get_handlers = Mock(return_value=[Mock(name="test_tool_handler")])
    mock_registry.configure = AsyncMock(return_value=mock_registry)
    return mock_registry


@pytest.fixture
def context_element():
    return ContextElement(
        type=ContextElementType.FILE,
        name="Test file",
        contents="Test file contents",
    )


@pytest.fixture
def workflow_with_project(context_element):
    workflow = Workflow(
        workflow_id="test-id",
        workflow_metadata={},
        workflow_type=CategoryEnum.WORKFLOW_CHAT,
        context_elements=[context_element],
    )
    workflow._project = {
        "id": 123,
        "name": "test-project",
        "http_url_to_repo": "https://example.com",
        "web_url": "https://example.com/test-project",
        "description": "A test project",
    }
    workflow._http_client = MagicMock()
    return workflow


@pytest.mark.asyncio
async def test_workflow_initialization(workflow_with_project, context_element):
    initial_state = workflow_with_project.get_workflow_state("Test chat goal")
    expected_system_prompt = CHAT_SYSTEM_PROMPT.format(
        current_date=datetime.now().strftime("%Y-%m-%d"),
        project_id="123",
        project_name="test-project",
        project_url="https://example.com/test-project",
    )

    assert initial_state["status"] == WorkflowStatusEnum.NOT_STARTED
    assert (
        initial_state["conversation_history"][AGENT_NAME][0].content
        == expected_system_prompt
    )
    assert initial_state["plan"] == {"steps": []}
    assert len(initial_state["ui_chat_log"]) == 1
    assert initial_state["ui_chat_log"][0]["message_type"] == MessageTypeEnum.TOOL
    assert "Starting chat: Test chat goal" in initial_state["ui_chat_log"][0]["content"]
    assert initial_state["ui_chat_log"][0]["status"] == ToolStatus.SUCCESS
    assert initial_state["context_elements"] == [context_element]


@pytest.mark.asyncio
async def test_execute_agent(workflow_with_project):
    # Setup test data
    test_message = "Test response"
    mock_agent_result = {
        "conversation_history": {
            AGENT_NAME: [
                SystemMessage(content="test system"),
                AIMessage(content=test_message),
            ]
        },
        "status": WorkflowStatusEnum.EXECUTION,
    }

    workflow_with_project._agent = AsyncMock()
    workflow_with_project._agent.run.return_value = mock_agent_result
    workflow_with_project._context_elements = []

    state = ChatWorkflowState(
        plan={"steps": []},
        status=WorkflowStatusEnum.EXECUTION,
        conversation_history={AGENT_NAME: []},
        ui_chat_log=[],
        last_human_input=None,
        context_elements=[],
    )

    result = await workflow_with_project._execute_agent(state)

    assert result["status"] == WorkflowStatusEnum.INPUT_REQUIRED
    assert len(result["ui_chat_log"]) == 1
    assert result["ui_chat_log"][0]["content"] == test_message
    assert "context_elements" in result["ui_chat_log"][0]
    assert result["ui_chat_log"][0]["context_elements"] == []
    assert result["ui_chat_log"][0]["message_type"] == MessageTypeEnum.AGENT
    assert result["ui_chat_log"][0]["status"] == ToolStatus.SUCCESS


@pytest.mark.asyncio
async def test_execute_agent_with_tools(workflow_with_project):
    # Setup test data
    test_message = "Test response"
    mock_agent_result = {
        "conversation_history": {
            AGENT_NAME: [
                SystemMessage(content="test system"),
                AIMessage(
                    content=test_message,
                    tool_calls=[
                        {
                            "name": "list_issues",
                            "args": {"project_id": "123"},
                            "id": "1",
                        }
                    ],
                ),
            ]
        },
        "status": WorkflowStatusEnum.EXECUTION,
    }

    workflow_with_project._agent = AsyncMock()
    workflow_with_project._agent.run.return_value = mock_agent_result
    workflow_with_project._context_elements = []

    state = ChatWorkflowState(
        plan={"steps": []},
        status=WorkflowStatusEnum.EXECUTION,
        conversation_history={AGENT_NAME: []},
        ui_chat_log=[],
        last_human_input=None,
        context_elements=[],
    )

    result = await workflow_with_project._execute_agent(state)

    assert result["status"] == WorkflowStatusEnum.INPUT_REQUIRED
    assert "ui_chat_log" not in result


@pytest.mark.parametrize(
    "message_content, expected_result",
    [
        (
            "Just text without tool calls",
            Routes.STOP,
        ),
        (
            [{"type": "text", "text": "Just text without tool calls"}],
            Routes.STOP,
        ),
    ],
    ids=[
        "Test with simple string content",
        "Test with list content but no tool_use",
    ],
)
def test_are_tools_called_with_various_content(message_content, expected_result):
    workflow = Workflow(
        workflow_id="test-id",
        workflow_metadata={},
        workflow_type=CategoryEnum.WORKFLOW_CHAT,
    )
    workflow._context_elements = []

    state: ChatWorkflowState = {
        "conversation_history": {AGENT_NAME: [AIMessage(content=message_content)]},
        "plan": {"steps": []},
        "status": WorkflowStatusEnum.EXECUTION,
        "ui_chat_log": [],
        "last_human_input": None,
        "context_elements": [],
    }
    assert workflow._are_tools_called(state) == expected_result

    # Test cancelled state
    state["status"] = WorkflowStatusEnum.CANCELLED
    assert workflow._are_tools_called(state) == Routes.STOP

    # Test error state
    state["status"] = WorkflowStatusEnum.ERROR
    assert workflow._are_tools_called(state) == Routes.STOP


def test_are_tools_called_with_tool_use():
    workflow = Workflow(
        workflow_id="test-id",
        workflow_metadata={},
        workflow_type=CategoryEnum.WORKFLOW_CHAT,
    )
    workflow._context_elements = []

    tool_message = AIMessage(content="Using tools")
    tool_message.tool_calls = [
        {
            "id": "toolu_random_id",
            "args": {"project_id": 3, "sort": "desc", "order_by": "created_at"},
            "name": "list_issues",
        }
    ]

    state: ChatWorkflowState = {
        "conversation_history": {AGENT_NAME: [tool_message]},
        "plan": {"steps": []},
        "status": WorkflowStatusEnum.EXECUTION,
        "ui_chat_log": [],
        "last_human_input": None,
        "context_elements": [],
    }
    assert workflow._are_tools_called(state) == Routes.TOOL_USE


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.chat.workflow.new_chat_client")
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry")
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow", autospec=True)
@patch(
    "duo_workflow_service.workflows.abstract_workflow.fetch_project_data_with_workflow_id"
)
@patch(
    "duo_workflow_service.workflows.abstract_workflow.fetch_workflow_config",
)
@patch("duo_workflow_service.workflows.abstract_workflow.UserInterface", autospec=True)
async def test_workflow_run(
    mock_user_interface,
    mock_fetch_workflow_config,
    mock_fetch_project_data,
    mock_gitlab_workflow,
    mock_tools_registry,
    mock_chat_client,
):
    mock_user_interface_instance = mock_user_interface.return_value
    mock_tools_registry.configure = AsyncMock(
        return_value=MagicMock(spec=ToolsRegistry)
    )
    mock_fetch_project_data.return_value = {
        "id": 1,
        "name": "test-project",
        "description": "Test project",
        "http_url_to_repo": "https://example.com/project",
        "web_url": "https://example.com/project",
    }

    mock_git_lab_workflow_instance = mock_gitlab_workflow.return_value
    mock_git_lab_workflow_instance.__aenter__.return_value = (
        mock_git_lab_workflow_instance
    )
    mock_git_lab_workflow_instance.__aexit__.return_value = None
    mock_git_lab_workflow_instance.aget_tuple = AsyncMock(return_value=None)
    mock_git_lab_workflow_instance.aput = AsyncMock(
        return_value={
            "configurable": {"thread_id": "test-id", "checkpoint_id": "checkpoint1"}
        }
    )

    state = {"status": "Not Started", "ui_chat_log": []}

    class AsyncIterator:
        def __init__(self):
            self.call_count = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.call_count += 1
            if self.call_count > 1:
                raise StopAsyncIteration
            else:
                return ("values", state)

    with patch(
        "duo_workflow_service.workflows.chat.workflow.StateGraph"
    ) as mock_graph_cls:
        compiled_graph = MagicMock()
        compiled_graph.astream.return_value = AsyncIterator()
        mock_graph = mock_graph_cls.return_value
        mock_graph.compile.return_value = compiled_graph

        workflow = Workflow(
            workflow_id="test-id",
            workflow_metadata={},
            workflow_type=CategoryEnum.WORKFLOW_CHAT,
        )

        await workflow.run("Test chat goal")

        assert workflow.is_done

        mock_user_interface_instance.send_event.assert_called_with(
            type="values", state=state, stream=True
        )
        assert mock_user_interface_instance.send_event.call_count == 1


@pytest.mark.parametrize(
    "feature_flag_value, expected_tools",
    [
        ("duo_workflow_chat_mutation_tools", CHAT_MUTATION_TOOLS),
        ("", CHAT_READ_ONLY_TOOLS),
    ],
)
@patch("duo_workflow_service.workflows.chat.workflow.current_feature_flag_context")
@patch("duo_workflow_service.components.tools_registry.ToolsRegistry.toolset")
@patch("duo_workflow_service.workflows.chat.workflow.Agent")
def test_tools_registry_interaction(
    mock_agent,
    mock_toolset,
    mock_feature_flag_context,
    feature_flag_value,
    expected_tools,
):
    mock_feature_flag_context.get.return_value = (
        [feature_flag_value] if feature_flag_value else []
    )

    mock_toolset.return_value = [Mock(name=f"mock_{tool}") for tool in expected_tools]

    workflow = Workflow(
        workflow_id="test-id",
        workflow_metadata={},
        workflow_type=CategoryEnum.WORKFLOW_CHAT,
    )
    workflow._context_elements = []
    tools_registry = MagicMock(spec=ToolsRegistry)
    checkpointer = MagicMock()

    with patch("duo_workflow_service.workflows.chat.workflow.new_chat_client"):
        workflow._compile("Test goal", tools_registry, checkpointer)

    assert tools_registry.toolset.called

    args, _ = tools_registry.toolset.call_args
    tools_passed_to_get_batch = args[0]

    for tool in expected_tools:
        assert tool in tools_passed_to_get_batch

    # Verify Agent initialization parameters
    mock_agent.assert_called_once()
    _, kwargs = mock_agent.call_args
    assert kwargs.get("check_events") is False
