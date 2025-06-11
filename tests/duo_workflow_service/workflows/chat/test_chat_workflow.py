from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from dependency_injector import containers
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from contract import contract_pb2
from contract.contract_pb2 import ContextElement, ContextElementType
from duo_workflow_service.agents.chat_agent import ChatAgent
from duo_workflow_service.checkpointer.gitlab_workflow import WorkflowStatusEventEnum
from duo_workflow_service.components.tools_registry import ToolsRegistry
from duo_workflow_service.entities import (
    MessageTypeEnum,
    ToolStatus,
    WorkflowStatusEnum,
)
from duo_workflow_service.entities.state import ChatWorkflowState
from duo_workflow_service.internal_events.event_enum import CategoryEnum
from duo_workflow_service.workflows.chat.workflow import (
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
def prompt_class():
    return ChatAgent


@pytest.fixture
def config_values():
    yield {"mock_model_responses": True}


@pytest.fixture
def workflow_with_project(
    context_element, mock_container: containers.Container, prompt: ChatAgent
):
    workflow = Workflow(
        workflow_id="test-id",
        workflow_metadata={},
        workflow_type=CategoryEnum.WORKFLOW_CHAT,
        context_elements=[context_element],
        mcp_tools=[contract_pb2.McpTool(name="extra_tool", description="Extra tool")],
    )
    workflow._project = {
        "id": 123,
        "name": "test-project",
        "http_url_to_repo": "https://example.com",
        "web_url": "https://example.com/test-project",
        "description": "A test project",
    }
    workflow._http_client = MagicMock()
    workflow._context_elements = []
    workflow._agent = prompt
    return workflow


@pytest.mark.asyncio
async def test_workflow_initialization(workflow_with_project):
    initial_state = workflow_with_project.get_workflow_state("Test chat goal")

    assert initial_state["status"] == WorkflowStatusEnum.NOT_STARTED
    assert initial_state["plan"] == {"steps": []}
    assert len(initial_state["ui_chat_log"]) == 1
    assert initial_state["ui_chat_log"][0]["message_type"] == MessageTypeEnum.TOOL
    assert "Starting chat: Test chat goal" in initial_state["ui_chat_log"][0]["content"]
    assert initial_state["ui_chat_log"][0]["status"] == ToolStatus.SUCCESS
    assert initial_state["context_elements"] == []
    assert initial_state["project"]["name"] == "test-project"


@pytest.mark.asyncio
async def test_execute_agent(workflow_with_project):
    workflow_with_project._context_elements = []

    state = ChatWorkflowState(
        plan={"steps": []},
        status=WorkflowStatusEnum.EXECUTION,
        conversation_history={"test_prompt": []},
        ui_chat_log=[],
        last_human_input=None,
        context_elements=[],
    )

    result = await workflow_with_project._agent.run(state)

    assert result["status"] == WorkflowStatusEnum.INPUT_REQUIRED
    assert len(result["ui_chat_log"]) == 1
    assert result["ui_chat_log"][0]["content"] == "Hello there!"
    assert "context_elements" in result["ui_chat_log"][0]
    assert result["ui_chat_log"][0]["context_elements"] == []
    assert result["ui_chat_log"][0]["message_type"] == MessageTypeEnum.AGENT
    assert result["ui_chat_log"][0]["status"] == ToolStatus.SUCCESS


class TestExecuteAgentWithTools:
    @pytest.fixture
    def model_response(self):
        return ToolMessage(content="tool calling", tool_call_id="random_id")

    @pytest.fixture
    def model_disable_streaming(self):
        return "tool_calling"

    @pytest.mark.asyncio
    async def test_execute_agent_with_tools(self, workflow_with_project):
        workflow_with_project._context_elements = []

        state = ChatWorkflowState(
            plan={"steps": []},
            status=WorkflowStatusEnum.EXECUTION,
            conversation_history={"test_prompt": [HumanMessage(content="hi")]},
            ui_chat_log=[],
            last_human_input=None,
            context_elements=[],
        )

        result = await workflow_with_project._agent.run(state)

        assert result["status"] == WorkflowStatusEnum.INPUT_REQUIRED
        assert len(result["ui_chat_log"]) == 1
        assert result["ui_chat_log"][0]["content"] == "tool calling"
        assert "context_elements" in result["ui_chat_log"][0]
        assert result["ui_chat_log"][0]["context_elements"] == []
        assert result["ui_chat_log"][0]["message_type"] == MessageTypeEnum.AGENT
        assert result["ui_chat_log"][0]["status"] == ToolStatus.SUCCESS


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
def test_are_tools_called_with_various_content(
    workflow_with_project, message_content, expected_result
):
    workflow = workflow_with_project
    workflow._context_elements = []

    state: ChatWorkflowState = {
        "conversation_history": {"test_prompt": [AIMessage(content=message_content)]},
        "plan": {"steps": []},
        "status": WorkflowStatusEnum.EXECUTION,
        "ui_chat_log": [],
        "last_human_input": None,
        "context_elements": [],
        "project": None,
    }
    assert workflow._are_tools_called(state) == expected_result

    # Test cancelled state
    state["status"] = WorkflowStatusEnum.CANCELLED
    assert workflow._are_tools_called(state) == Routes.STOP

    # Test error state
    state["status"] = WorkflowStatusEnum.ERROR
    assert workflow._are_tools_called(state) == Routes.STOP


def test_are_tools_called_with_tool_use(workflow_with_project):
    workflow = workflow_with_project
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
        "conversation_history": {"test_prompt": [tool_message]},
        "plan": {"steps": []},
        "status": WorkflowStatusEnum.EXECUTION,
        "ui_chat_log": [],
        "last_human_input": None,
        "context_elements": [],
        "project": None,
    }
    assert workflow._are_tools_called(state) == Routes.TOOL_USE


@pytest.mark.asyncio
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
    workflow_with_project,
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

        workflow = workflow_with_project

        await workflow.run("Test chat goal")

        assert workflow.is_done

        mock_user_interface_instance.send_event.assert_called_with(
            type="values", state=state, stream=True
        )
        assert mock_user_interface_instance.send_event.call_count == 1


@pytest.mark.parametrize(
    ("feature_flags", "workflow_config", "expected_tools"),
    [
        (
            ["duo_workflow_chat_mutation_tools"],
            {},
            CHAT_READ_ONLY_TOOLS + CHAT_MUTATION_TOOLS,
        ),
        ([], {}, CHAT_READ_ONLY_TOOLS),
        (["duo_workflow_mcp_support"], {}, CHAT_READ_ONLY_TOOLS + ["extra_tool"]),
        (
            [],
            {"mcp_enabled": True},
            CHAT_READ_ONLY_TOOLS + ["extra_tool"],
        ),
        (
            ["duo_workflow_chat_mutation_tools", "duo_workflow_mcp_support"],
            {"mcp_enabled": True},
            CHAT_READ_ONLY_TOOLS + CHAT_MUTATION_TOOLS + ["extra_tool"],
        ),
    ],
)
@patch("duo_workflow_service.workflows.chat.workflow.current_feature_flag_context")
@patch("duo_workflow_service.components.tools_registry.ToolsRegistry.toolset")
def test_tools_registry_interaction(
    mock_toolset,
    mock_feature_flag_context,
    feature_flags,
    workflow_config,
    expected_tools,
    workflow_with_project,
):
    mock_feature_flag_context.get.return_value = feature_flags

    mock_toolset.return_value = [Mock(name=f"mock_{tool}") for tool in expected_tools]

    workflow = workflow_with_project
    workflow._context_elements = []
    workflow._workflow_config = workflow_config
    tools_registry = MagicMock(spec=ToolsRegistry)
    checkpointer = MagicMock()

    workflow._compile("Test goal", tools_registry, checkpointer)

    assert tools_registry.toolset.called

    args, _ = tools_registry.toolset.call_args
    tools_passed_to_get_batch = args[0]

    for tool in expected_tools:
        assert tool in tools_passed_to_get_batch


@pytest.mark.asyncio
async def test_get_graph_input_start(workflow_with_project):
    result = await workflow_with_project.get_graph_input(
        "Test goal", WorkflowStatusEventEnum.START
    )

    assert result["status"] == WorkflowStatusEnum.NOT_STARTED
    assert result["conversation_history"]["test_prompt"][0].content == "Test goal"


@pytest.mark.asyncio
async def test_get_graph_input_resume(workflow_with_project):
    result = await workflow_with_project.get_graph_input(
        "New input", WorkflowStatusEventEnum.RESUME
    )

    assert result.goto == "agent"
    assert result.update["status"] == WorkflowStatusEnum.EXECUTION
    assert (
        result.update["conversation_history"]["test_prompt"][0].content == "New input"
    )


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.chat.workflow.log_exception")
async def test_handle_workflow_failure(mock_log_exception, workflow_with_project):
    error = Exception("Test error")
    compiled_graph = MagicMock()
    graph_config = MagicMock()

    await workflow_with_project._handle_workflow_failure(
        error=error, compiled_graph=compiled_graph, graph_config=graph_config
    )

    mock_log_exception.assert_called_once_with(
        error, extra={"workflow_id": workflow_with_project._workflow_id}
    )


@patch("logging.Logger.info")
def test_log_workflow_elements(mock_logger_info, workflow_with_project):
    element = {
        "ui_chat_log": [
            {
                "message_type": MessageTypeEnum.AGENT,
                "content": "Test message content",
                "timestamp": datetime.now().isoformat(),
                "status": ToolStatus.SUCCESS,
            }
        ]
    }

    workflow_with_project.log = Mock()
    workflow_with_project.log_workflow_elements(element)

    workflow_with_project.log.info.assert_any_call("###############################")

    format_call_args = workflow_with_project.log.info.call_args_list[1][0]
    assert format_call_args[0].startswith(
        "%s"
    )  # Format string starts with message type
    assert "Test message content" in format_call_args[2]  # Second arg is content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "conversation_content,response_content,tool_calls,expected_status,has_ui_log",
    [
        (
            [HumanMessage(content="List issues")],
            "I'll help you with that.",
            [{"id": "call_123", "name": "list_issues", "args": {"project_id": 123}}],
            WorkflowStatusEnum.EXECUTION,
            False,
        ),
        (
            [HumanMessage(content="Hello")],
            "Here's my response to your question.",
            None,
            WorkflowStatusEnum.INPUT_REQUIRED,
            True,
        ),
        (
            [HumanMessage(content="Simple question")],
            "No tools needed for this response.",
            [],
            WorkflowStatusEnum.INPUT_REQUIRED,
            True,
        ),
        (
            [],
            "Hello there!",
            None,
            WorkflowStatusEnum.INPUT_REQUIRED,
            True,
        ),
    ],
    ids=[
        "with_tool_calls_sets_execution_status",
        "without_tool_calls_sets_input_required_status",
        "with_empty_tool_calls_sets_input_required_status",
        "without_tools_returns_input_required",
    ],
)
@patch("ai_gateway.prompts.base.Prompt.ainvoke")
async def test_chat_agent_status_handling(
    mock_ainvoke,
    workflow_with_project,
    conversation_content,
    response_content,
    tool_calls,
    expected_status,
    has_ui_log,
):
    workflow_with_project._context_elements = []

    state = ChatWorkflowState(
        plan={"steps": []},
        status=WorkflowStatusEnum.EXECUTION,
        conversation_history={"test_prompt": conversation_content},
        ui_chat_log=[],
        last_human_input=None,
        context_elements=[],
    )

    ai_response = AIMessage(content=response_content)
    if tool_calls is not None:
        ai_response.tool_calls = tool_calls
    mock_ainvoke.return_value = ai_response

    result = await workflow_with_project._agent.run(state)

    assert result["status"] == expected_status

    if has_ui_log:
        assert "ui_chat_log" in result
        assert len(result["ui_chat_log"]) == 1
        assert result["ui_chat_log"][0]["content"] == response_content
        if conversation_content:  # Only check these for non-empty conversation
            assert result["ui_chat_log"][0]["message_type"] == MessageTypeEnum.AGENT
            assert result["ui_chat_log"][0]["status"] == ToolStatus.SUCCESS
        else:  # For empty conversation case
            assert "context_elements" in result["ui_chat_log"][0]
            assert result["ui_chat_log"][0]["context_elements"] == []
            assert result["ui_chat_log"][0]["message_type"] == MessageTypeEnum.AGENT
            assert result["ui_chat_log"][0]["status"] == ToolStatus.SUCCESS
    else:
        assert "ui_chat_log" not in result


@pytest.mark.asyncio
@patch("ai_gateway.prompts.base.Prompt.ainvoke")
async def test_chat_workflow_status_flow_integration(
    mock_ainvoke, workflow_with_project
):
    workflow_with_project._context_elements = []

    # Test sequence: agent with tools -> tools execution -> agent final response
    # 1. Agent responds with tool calls (should be EXECUTION status)
    state_1 = ChatWorkflowState(
        plan={"steps": []},
        status=WorkflowStatusEnum.EXECUTION,
        conversation_history={"test_prompt": [HumanMessage(content="List issues")]},
        ui_chat_log=[],
        last_human_input=None,
        context_elements=[],
    )

    ai_response_with_tools = AIMessage(content="I'll list the issues for you.")
    ai_response_with_tools.tool_calls = [
        {"id": "call_123", "name": "list_issues", "args": {}}
    ]
    mock_ainvoke.return_value = ai_response_with_tools

    result_1 = await workflow_with_project._agent.run(state_1)
    assert result_1["status"] == WorkflowStatusEnum.EXECUTION

    # 2. After tools execute, agent provides final response (should be INPUT_REQUIRED)
    state_2 = ChatWorkflowState(
        plan={"steps": []},
        status=WorkflowStatusEnum.EXECUTION,
        conversation_history={
            "test_prompt": [
                HumanMessage(content="List issues"),
                ai_response_with_tools,
            ]
        },
        ui_chat_log=[],
        last_human_input=None,
        context_elements=[],
    )

    ai_response_final = AIMessage(content="Here are the issues I found: ...")
    mock_ainvoke.return_value = ai_response_final

    result_2 = await workflow_with_project._agent.run(state_2)
    assert result_2["status"] == WorkflowStatusEnum.INPUT_REQUIRED
    assert "ui_chat_log" in result_2
