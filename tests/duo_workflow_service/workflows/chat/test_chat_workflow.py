from unittest.mock import MagicMock, Mock, patch

import pytest
from dependency_injector import containers
from gitlab_cloud_connector import CloudConnectorUser, UserClaims, WrongUnitPrimitives
from google.protobuf import struct_pb2
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_gateway.model_metadata import ModelMetadata
from contract import contract_pb2
from duo_workflow_service.agents.chat_agent import ChatAgent
from duo_workflow_service.checkpointer.gitlab_workflow import WorkflowStatusEventEnum
from duo_workflow_service.components.tools_registry import ToolsRegistry
from duo_workflow_service.entities import (
    MessageTypeEnum,
    ToolStatus,
    WorkflowStatusEnum,
)
from duo_workflow_service.entities.state import (
    ApprovalStateRejection,
    ChatWorkflowState,
)
from duo_workflow_service.workflows.chat.workflow import (
    CHAT_GITLAB_MUTATION_TOOLS,
    CHAT_MUTATION_TOOLS,
    CHAT_READ_ONLY_TOOLS,
    RUN_COMMAND_TOOLS,
    Routes,
    Workflow,
)
from duo_workflow_service.workflows.type_definitions import AdditionalContext
from lib.feature_flags import FeatureFlag, current_feature_flag_context
from lib.internal_events.event_enum import CategoryEnum


@pytest.fixture(name="prompt_class")
def prompt_class_fixture():
    return ChatAgent


@pytest.fixture(name="config_values")
def config_values_fixture():
    yield {"mock_model_responses": True}


@pytest.fixture(name="user")
def user_fixture():
    return CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(
            scopes=["duo_chat"],
            issuer="gitlab-duo-workflow-service",
        ),
    )


@pytest.fixture(name="workflow_with_project")
def workflow_with_project_fixture(
    mock_duo_workflow_service_container: containers.Container,
    prompt: ChatAgent,
    user: CloudConnectorUser,
    mock_tools_registry: Mock,
):
    workflow = Workflow(
        workflow_id="test-id",
        workflow_metadata={},
        workflow_type=CategoryEnum.WORKFLOW_CHAT,
        mcp_tools=[contract_pb2.McpTool(name="extra_tool", description="Extra tool")],
        user=user,
    )
    additional_context = [
        AdditionalContext(
            category="file",
            id="test-file-id",
            content="test content",
            metadata={"path": "/test/file.py"},
        )
    ]
    workflow._project = {
        "id": 123,
        "name": "test-project",
        "http_url_to_repo": "https://example.com",
        "web_url": "https://example.com/test-project",
        "description": "A test project",
        "languages": [{"name": "Python", "share": 1.0}],
        "default_branch": "main",
        "exclusion_rules": None,
    }
    workflow._namespace = None
    workflow._additional_context = additional_context
    workflow._http_client = MagicMock()
    prompt.tools_registry = mock_tools_registry
    workflow._agent = prompt
    return workflow


@pytest.fixture(name="workflow_with_approval")
def workflow_with_approval_fixture(workflow_with_project):
    workflow = workflow_with_project
    workflow._approval = contract_pb2.Approval(
        approval=contract_pb2.Approval.Approved()
    )

    return workflow


@pytest.mark.asyncio
async def test_workflow_initialization(workflow_with_project):
    initial_state = workflow_with_project.get_workflow_state("Test chat goal")

    assert initial_state["status"] == WorkflowStatusEnum.NOT_STARTED
    assert initial_state["plan"] == {"steps": []}
    assert len(initial_state["ui_chat_log"]) == 1
    assert initial_state["ui_chat_log"][0]["message_type"] == MessageTypeEnum.USER
    assert "Test chat goal" in initial_state["ui_chat_log"][0]["content"]
    assert initial_state["ui_chat_log"][0]["status"] == ToolStatus.SUCCESS
    assert len(initial_state["ui_chat_log"][0]["additional_context"]) == 1
    assert initial_state["ui_chat_log"][0]["additional_context"][0].category == "file"
    assert initial_state["project"]["name"] == "test-project"


@pytest.mark.asyncio
async def test_workflow_initialization_with_additional_context(workflow_with_project):
    additional_context = [
        AdditionalContext(
            category="file",
            id="file1",
            content="file content 1",
            metadata={"path": "/path/to/file1"},
        ),
        AdditionalContext(
            category="issue",
            id="issue123",
            content="issue description",
            metadata={"title": "Bug report", "state": "open"},
        ),
        AdditionalContext(
            category="terminal",
            content="command output",
            metadata={"command": "ls -la"},
        ),
    ]
    workflow_with_project._additional_context = additional_context

    initial_state = workflow_with_project.get_workflow_state("Test chat goal")

    assert initial_state["status"] == WorkflowStatusEnum.NOT_STARTED
    assert initial_state["ui_chat_log"][0]["additional_context"] == additional_context
    assert len(initial_state["ui_chat_log"][0]["additional_context"]) == 3
    assert initial_state["ui_chat_log"][0]["additional_context"][0].category == "file"
    assert initial_state["ui_chat_log"][0]["additional_context"][1].category == "issue"
    assert (
        initial_state["ui_chat_log"][0]["additional_context"][2].category == "terminal"
    )
    assert (
        initial_state["conversation_history"]["test_prompt"][0].additional_kwargs[
            "additional_context"
        ]
        == additional_context
    )


@pytest.mark.asyncio
async def test_execute_agent(workflow_with_project):
    state = ChatWorkflowState(
        plan={"steps": []},
        status=WorkflowStatusEnum.EXECUTION,
        conversation_history={"test_prompt": []},
        ui_chat_log=[],
        last_human_input=None,
    )

    result = await workflow_with_project._agent.run(state)

    assert result["status"] == WorkflowStatusEnum.INPUT_REQUIRED
    assert len(result["ui_chat_log"]) == 1
    assert result["ui_chat_log"][0]["content"] == "Hello there!"
    assert result["ui_chat_log"][0]["message_type"] == MessageTypeEnum.AGENT
    assert result["ui_chat_log"][0]["status"] == ToolStatus.SUCCESS


class TestExecuteAgentWithTools:
    @pytest.fixture(name="model_response")
    def model_response_fixture(self):
        return [ToolMessage(content="tool calling", tool_call_id="random_id")]

    @pytest.fixture(name="model_disable_streaming")
    def model_disable_streaming_fixture(self):
        return "tool_calling"

    @pytest.mark.asyncio
    async def test_execute_agent_with_tools(self, workflow_with_project):
        state = ChatWorkflowState(
            plan={"steps": []},
            status=WorkflowStatusEnum.EXECUTION,
            conversation_history={"test_prompt": [HumanMessage(content="hi")]},
            ui_chat_log=[],
            last_human_input=None,
        )

        result = await workflow_with_project._agent.run(state)

        assert result["status"] == WorkflowStatusEnum.INPUT_REQUIRED
        assert len(result["ui_chat_log"]) == 1
        assert result["ui_chat_log"][0]["content"] == "tool calling"
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

    state: ChatWorkflowState = {
        "conversation_history": {"test_prompt": [AIMessage(content=message_content)]},
        "plan": {"steps": []},
        "status": WorkflowStatusEnum.EXECUTION,
        "ui_chat_log": [],
        "last_human_input": None,
        "goal": "test goal",
        "project": None,
        "namespace": None,
        "approval": None,
        "preapproved_tools": None,
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
        "goal": "test goal",
        "project": None,
        "namespace": None,
        "approval": None,
        "preapproved_tools": None,
    }
    assert workflow._are_tools_called(state) == Routes.TOOL_USE


@pytest.mark.asyncio
@pytest.mark.usefixtures(
    "mock_tools_registry_cls",
    "mock_git_lab_workflow_instance",
    "mock_fetch_workflow_and_container_data",
)
@patch("duo_workflow_service.workflows.chat.workflow.current_model_metadata_context")
async def test_workflow_run(
    mock_model_metadata_context,
    mock_checkpoint_notifier,
    workflow_with_project,
):
    mock_model_metadata = MagicMock()
    mock_model_metadata_context.get.return_value = mock_model_metadata

    mock_user_interface_instance = mock_checkpoint_notifier.return_value
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

        mock_agent = MagicMock()
        with patch.object(
            workflow._prompt_registry, "get_on_behalf", return_value=mock_agent
        ) as mock_get_on_behalf:
            await workflow.run("Test chat goal")

            assert workflow.is_done

            mock_get_on_behalf.assert_called_once()
            call_args = mock_get_on_behalf.call_args

            assert call_args.kwargs["model_metadata"] == mock_model_metadata
            assert call_args.kwargs["user"] == workflow._user
            assert call_args.kwargs["prompt_id"] == "chat/agent"
            assert call_args.kwargs["prompt_version"] == "^1.0.0"
            assert (
                call_args.kwargs["internal_event_category"]
                == "duo_workflow_service.workflows.chat.workflow"
            )

        mock_user_interface_instance.send_event.assert_called_with(
            type="values", state=state, stream=True
        )
        assert mock_user_interface_instance.send_event.call_count == 1


class TestUnauthorizedChatExecution:
    @pytest.fixture(name="user")
    def user_fixture(self):
        return CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(
                scopes=["unknown_scope"],
                issuer="gitlab-duo-workflow-service",
            ),
        )

    def test_workflow_run(
        self,
        workflow_with_project,
    ):
        with pytest.raises(WrongUnitPrimitives):
            workflow_with_project._compile("Test goal", MagicMock(), MagicMock())


@pytest.mark.parametrize(
    ("feature_flags", "workflow_config", "expected_tools"),
    [
        (
            [],
            {},
            CHAT_READ_ONLY_TOOLS
            + CHAT_MUTATION_TOOLS
            + RUN_COMMAND_TOOLS
            + CHAT_GITLAB_MUTATION_TOOLS,
        ),
    ],
)
@patch("duo_workflow_service.components.tools_registry.ToolsRegistry.toolset")
def test_tools_registry_interaction(
    mock_toolset,
    feature_flags,
    workflow_config,
    expected_tools,
    workflow_with_project,
):
    current_feature_flag_context.set(set(feature_flags))

    mock_toolset.return_value = [Mock(name=f"mock_{tool}") for tool in expected_tools]

    workflow = workflow_with_project
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
    assert result["goal"] == "Test goal"
    assert result["conversation_history"]["test_prompt"][0].content == "Test goal"
    assert result["ui_chat_log"][0]["message_type"] == MessageTypeEnum.USER
    assert "Test goal" in result["ui_chat_log"][0]["content"]
    assert len(result["ui_chat_log"][0]["additional_context"]) == 1
    assert result["ui_chat_log"][0]["additional_context"][0].category == "file"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status", [WorkflowStatusEventEnum.RETRY, WorkflowStatusEventEnum.RESUME]
)
async def test_get_graph_input(workflow_with_project, status):
    result = await workflow_with_project.get_graph_input("New input", status)

    assert result.goto == "agent"
    assert result.update["status"] == WorkflowStatusEnum.EXECUTION
    assert (
        result.update["conversation_history"]["test_prompt"][0].content == "New input"
    )
    assert result.update["ui_chat_log"][-1]["message_type"] == MessageTypeEnum.USER
    assert result.update["ui_chat_log"][-1]["content"] == "New input"
    assert len(result.update["ui_chat_log"][-1]["additional_context"]) == 1
    assert result.update["ui_chat_log"][-1]["additional_context"][0].category == "file"


@pytest.mark.asyncio
async def test_get_graph_input_resume_with_approval(workflow_with_approval):
    """Test graph input with approved tool calls."""
    result = await workflow_with_approval.get_graph_input(
        "", WorkflowStatusEventEnum.RESUME
    )

    assert result.goto == "run_tools"
    assert result.update["status"] == WorkflowStatusEnum.EXECUTION
    assert "conversation_history" not in result.update
    assert "ui_chat_log" not in result.update


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rejection_message", ["", "null", "Rejected the tool usage because it's not safe"]
)
async def test_get_graph_input_resume_with_rejected_approval(
    rejection_message, workflow_with_project
):
    """Test graph input with rejected tool calls."""

    workflow_with_rejected_approval = workflow_with_project
    workflow_with_rejected_approval._approval = contract_pb2.Approval(
        rejection=contract_pb2.Approval.Rejected(
            message=rejection_message,
        )
    )

    result = await workflow_with_rejected_approval.get_graph_input(
        "", WorkflowStatusEventEnum.RESUME
    )

    assert result.goto == "agent"
    assert result.update["status"] == WorkflowStatusEnum.EXECUTION
    assert "conversation_history" not in result.update
    assert result.update["approval"].message == rejection_message

    if rejection_message and rejection_message != "null":
        result.update["ui_chat_log"][-1]["content"] == rejection_message
    else:
        assert "ui_chat_log" not in result.update


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
        error,
        extra={
            "workflow_id": workflow_with_project._workflow_id,
            "source": "duo_workflow_service.workflows.chat.workflow",
        },
    )


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
    state = ChatWorkflowState(
        plan={"steps": []},
        status=WorkflowStatusEnum.EXECUTION,
        conversation_history={"test_prompt": conversation_content},
        ui_chat_log=[],
        last_human_input=None,
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
            assert result["ui_chat_log"][0]["message_type"] == MessageTypeEnum.AGENT
            assert result["ui_chat_log"][0]["status"] == ToolStatus.SUCCESS
    else:
        assert "ui_chat_log" not in result


@pytest.mark.asyncio
@patch("ai_gateway.prompts.base.Prompt.ainvoke")
async def test_chat_workflow_status_flow_integration(
    mock_ainvoke, workflow_with_project
):
    # Test sequence: agent with tools -> tools execution -> agent final response
    # 1. Agent responds with tool calls (should be EXECUTION status)
    state_1 = ChatWorkflowState(
        plan={"steps": []},
        status=WorkflowStatusEnum.EXECUTION,
        conversation_history={"test_prompt": [HumanMessage(content="List issues")]},
        ui_chat_log=[],
        last_human_input=None,
        project=None,
        approval=None,
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
        project=None,
        approval=None,
    )

    ai_response_final = AIMessage(content="Here are the issues I found: ...")
    mock_ainvoke.return_value = ai_response_final

    result_2 = await workflow_with_project._agent.run(state_2)
    assert result_2["status"] == WorkflowStatusEnum.INPUT_REQUIRED
    assert "ui_chat_log" in result_2


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_approval_required", [True])
@pytest.mark.usefixtures("mock_fetch_workflow_and_container_data")
async def test_agent_run_with_tool_approval_required(workflow_with_project):
    """Test agent run method when tools require approval."""

    state = ChatWorkflowState(
        plan={"steps": []},
        status=WorkflowStatusEnum.EXECUTION,
        conversation_history={"test_prompt": [HumanMessage(content="Create a file")]},
        ui_chat_log=[],
        last_human_input=None,
        project=None,
        approval=None,
    )

    ai_message = AIMessage(content="I'll create the file for you")
    ai_message.tool_calls = [
        {
            "id": "toolu_approval_id",
            "args": {"path": "/test/file.txt", "content": "Test content"},
            "name": "create_file_with_contents",
        }
    ]

    with patch("ai_gateway.prompts.base.Prompt.ainvoke", return_value=ai_message):
        result = await workflow_with_project._agent.run(state)

    assert result["status"] == WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED
    assert "ui_chat_log" in result
    assert len(result["ui_chat_log"]) == 1
    assert result["ui_chat_log"][0]["message_type"] == MessageTypeEnum.REQUEST
    assert "requires approval" in result["ui_chat_log"][0]["content"]
    assert result["ui_chat_log"][0]["tool_info"]["name"] == "create_file_with_contents"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_approval_required", [True])
@pytest.mark.usefixtures("mock_fetch_workflow_and_container_data")
async def test_agent_run_with_preapproved_tools(workflow_with_project):
    """Test agent run method when executed with preapproved tools."""

    state = ChatWorkflowState(
        plan={"steps": []},
        status=WorkflowStatusEnum.EXECUTION,
        conversation_history={"test_prompt": [HumanMessage(content="Create a file")]},
        ui_chat_log=[],
        last_human_input=None,
        project=None,
        approval=None,
        preapproved_tools=["create_file_with_contents"],
    )

    ai_message = AIMessage(content="I'll create the file for you")
    ai_message.tool_calls = [
        {
            "id": "toolu_approval_id",
            "args": {"path": "/test/file.txt", "content": "Test content"},
            "name": "create_file_with_contents",
        }
    ]

    with patch("ai_gateway.prompts.base.Prompt.ainvoke", return_value=ai_message):
        result = await workflow_with_project._agent.run(state)

    assert result["status"] == WorkflowStatusEnum.EXECUTION


@pytest.mark.asyncio
async def test_agent_resume_with_updated_preapproved_tools(workflow_with_project):
    """Test that preapproved_tools are updated in state when resuming."""
    # Set up workflow with new preapproved tools (simulating a new startRequest)
    workflow_with_project._preapproved_tools = [
        "tool_1",
        "tool_2",
        "tool_3",
    ]

    # Resume the workflow
    result = await workflow_with_project.get_graph_input(
        "", WorkflowStatusEventEnum.RESUME
    )

    # Verify the state update includes the new preapproved_tools
    assert result.goto == "agent"
    assert result.update["status"] == WorkflowStatusEnum.EXECUTION
    assert "preapproved_tools" in result.update
    assert result.update["preapproved_tools"] == [
        "tool_1",
        "tool_2",
        "tool_3",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cancel_tool_message", "expected_tool_message"),
    [
        (
            "I don't want this file created",
            "Tool is cancelled temporarily as user has a comment. Comment: I don't want this file created",
        ),
        (
            None,
            "Tool is cancelled by user. Don't run the command and stop tool execution in progress.",
        ),
    ],
    ids=[
        "Test with simple string content",
        "Test with list content but no tool_use",
    ],
)
async def test_agent_run_with_cancel_tool_message(
    workflow_with_project, cancel_tool_message, expected_tool_message
):
    """Test agent run method when a tool is cancelled with a message."""
    # Setup a state with a previous AI message containing tool calls
    ai_message_with_tools = AIMessage(content="I'll use a tool")
    ai_message_with_tools.tool_calls = [
        {
            "id": "toolu_cancelled_id",
            "args": {"project_id": 3},
            "name": "create_file_with_contents",
        }
    ]

    state = ChatWorkflowState(
        plan={"steps": []},
        status=WorkflowStatusEnum.EXECUTION,
        conversation_history={
            "test_prompt": [
                HumanMessage(content="Create a file"),
                ai_message_with_tools,
            ]
        },
        ui_chat_log=[],
        last_human_input=None,
        project=None,
        approval=ApprovalStateRejection(message=cancel_tool_message),
    )

    ai_response_after_cancel = AIMessage(
        content="I understand you don't want the file created"
    )

    with patch(
        "ai_gateway.prompts.base.Prompt.ainvoke", return_value=ai_response_after_cancel
    ):
        result = await workflow_with_project._agent.run(state)

    assert result["status"] == WorkflowStatusEnum.INPUT_REQUIRED
    assert "ui_chat_log" in result

    tool_messages = [
        msg
        for msg in state["conversation_history"]["test_prompt"]
        if hasattr(msg, "tool_call_id")
    ]
    assert len(tool_messages) == 1
    assert expected_tool_message == tool_messages[0].content


@pytest.mark.asyncio
async def test_workflow_with_approval_object():
    """Test creating a workflow with an approval object."""
    approval = contract_pb2.Approval(approval=contract_pb2.Approval.Approved())
    start_request = contract_pb2.StartWorkflowRequest()
    start_request.preapproved_tools.extend(["get_issue", "read_file"])

    workflow = Workflow(
        workflow_id="test-id",
        workflow_metadata={},
        workflow_type=CategoryEnum.WORKFLOW_CHAT,
        approval=approval,
        preapproved_tools=list(start_request.preapproved_tools),
    )

    assert workflow._approval is not None
    assert workflow._approval.WhichOneof("user_decision") == "approval"
    assert workflow._preapproved_tools == list(start_request.preapproved_tools)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "feature_flags,context_model_metadata,expected_prompt_id,expected_prompt_version,expected_model_metadata",
    [
        # Test case 1: Feature flag enabled, no model metadata - should use GPT-5
        (
            [FeatureFlag.DUO_AGENTIC_CHAT_OPENAI_GPT_5],
            None,
            "chat/agent/gpt_5",
            "^1.0.0",
            ModelMetadata(provider="gitlab", name="gpt_5"),
        ),
        # Test case 2: Feature flag disabled, no model metadata - should use default prompt
        (
            [],
            None,
            "chat/agent",
            "^1.0.0",
            None,
        ),
        # Test case 3: Feature flag enabled, but model metadata already provided - should use provided metadata
        (
            [FeatureFlag.DUO_AGENTIC_CHAT_OPENAI_GPT_5],
            ModelMetadata(name="llama3", provider="test"),
            "chat/agent",
            "^1.0.0",
            ModelMetadata(name="llama3", provider="test"),
        ),
    ],
)
@patch("duo_workflow_service.workflows.chat.workflow.current_model_metadata_context")
@patch.object(Workflow, "_get_tools")
@patch("duo_workflow_service.components.tools_registry.ToolsRegistry.toolset")
async def test_workflow_model_selection(
    mock_toolset,
    mock_get_tools,
    mock_model_metadata_context,
    feature_flags,
    context_model_metadata,
    expected_prompt_id,
    expected_prompt_version,
    expected_model_metadata,
    workflow_with_project,
):
    """Test that model selection works correctly with the GPT-5 feature flag."""
    # Setup
    current_feature_flag_context.set(set(feature_flags))
    mock_model_metadata_context.get.return_value = context_model_metadata

    mock_get_tools.return_value = ["list_issues", "get_issue"]
    mock_toolset.return_value.bindable = ["list_issues", "get_issue"]

    # Create tools registry and checkpointer mocks
    tools_registry = MagicMock(spec=ToolsRegistry)
    checkpointer = MagicMock()

    # Setup the mock for get_on_behalf to verify what it's called with
    mock_agent = MagicMock()
    with patch.object(
        workflow_with_project._prompt_registry, "get_on_behalf", return_value=mock_agent
    ) as mock_get_on_behalf:
        workflow_with_project._compile("Test goal", tools_registry, checkpointer)

        mock_get_on_behalf.assert_called_once()
        call_args = mock_get_on_behalf.call_args

        # Check the prompt ID matches expectations
        # For the GPT-5 case, we're expecting to see the prompt loaded from chat/agent/gpt_5
        if expected_prompt_id == "chat/agent/gpt_5":
            assert call_args.kwargs["prompt_id"] == "chat/agent"
            assert call_args.kwargs["model_metadata"].provider == "gitlab"
            assert call_args.kwargs["model_metadata"].name == "gpt_5"
        else:
            assert call_args.kwargs["prompt_id"] == expected_prompt_id
            assert call_args.kwargs["model_metadata"] == expected_model_metadata

        assert call_args.kwargs["prompt_version"] == expected_prompt_version
        assert call_args.kwargs["user"] == workflow_with_project._user


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_fetch_workflow_and_container_data")
@patch("duo_workflow_service.workflows.chat.workflow.current_model_metadata_context")
@patch.object(Workflow, "_get_tools")
@patch("duo_workflow_service.components.tools_registry.ToolsRegistry.toolset")
@patch("duo_workflow_service.workflows.chat.workflow.StateGraph")
async def test_compile_with_tools_override_and_flow_config(
    mock_state_graph,
    mock_toolset,
    mock_get_tools,
    mock_model_metadata_context,
    mock_duo_workflow_service_container,
):
    mock_model_metadata_context.get.return_value = None
    mock_get_tools.return_value = ["default_tool1", "default_tool2"]

    mock_agents_toolset = MagicMock()
    mock_agents_toolset.bindable = ["tool1", "tool2"]
    mock_toolset.return_value = mock_agents_toolset

    tools_registry = MagicMock()
    tools_registry.toolset.return_value = mock_agents_toolset
    checkpointer = MagicMock()

    user = CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(
            scopes=["duo_chat"],
            issuer="gitlab-duo-workflow-service",
        ),
    )

    workflow = Workflow(
        workflow_id="test-id",
        workflow_metadata={},
        workflow_type=CategoryEnum.WORKFLOW_CHAT,
        user=user,
        tools_override=["tool1", "tool2"],
        prompt_template_id_override="custom/prompt",
        prompt_template_version_override="2.0.0",
        prompt_template_override={"prompt_id": "custom/prompt", "content": "test"},
    )

    mock_prompt_runnable = MagicMock()
    mock_agent = MagicMock()

    mock_graph = MagicMock()
    mock_state_graph.return_value = mock_graph

    with (
        patch.object(
            workflow._prompt_registry, "get", return_value=mock_prompt_runnable
        ) as mock_get,
        patch.object(
            workflow._prompt_registry, "get_on_behalf", return_value=mock_agent
        ) as mock_get_on_behalf,
    ):
        workflow._compile("Test goal", tools_registry, checkpointer)

        tools_registry.toolset.assert_called_once_with(["tool1", "tool2"])
        mock_get_tools.assert_not_called()

        mock_get.assert_called_once_with(
            user=user,
            prompt_id="custom/prompt",
            prompt_version="2.0.0",
            model_metadata=None,
            internal_event_category="duo_workflow_service.workflows.chat.workflow",
            tools=mock_agents_toolset.bindable,
            preapproved_tools=[],
        )

        mock_get_on_behalf.assert_called_once_with(
            user=user,
            prompt_id="chat/agent",
            prompt_version="^1.0.0",
            model_metadata=None,
            internal_event_category="duo_workflow_service.workflows.chat.workflow",
            tools=mock_agents_toolset.bindable,
        )

        mock_graph.add_node.assert_any_call("agent", mock_agent.run)

        assert mock_agent.tools_registry == tools_registry
        assert mock_agent.prompt_runnable == mock_prompt_runnable


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.chat.workflow.current_model_metadata_context")
@patch.object(Workflow, "_get_tools")
@patch("duo_workflow_service.components.tools_registry.ToolsRegistry.toolset")
async def test_compile_without_overrides(
    mock_toolset, mock_get_tools, mock_model_metadata_context, workflow_with_project
):
    mock_model_metadata_context.get.return_value = None
    mock_get_tools.return_value = ["default_tool1", "default_tool2"]

    mock_agents_toolset = MagicMock()
    mock_agents_toolset.bindable = ["default_tool1", "default_tool2"]
    mock_toolset.return_value = mock_agents_toolset

    tools_registry = MagicMock()
    tools_registry.toolset.return_value = mock_agents_toolset
    checkpointer = MagicMock()

    mock_agent = MagicMock()

    with patch.object(
        workflow_with_project._prompt_registry, "get_on_behalf", return_value=mock_agent
    ) as mock_get_on_behalf:
        workflow_with_project._compile("Test goal", tools_registry, checkpointer)

        # Verify _get_tools was called since no tools_override
        mock_get_tools.assert_called_once()
        tools_registry.toolset.assert_called_once_with(
            ["default_tool1", "default_tool2"]
        )

        # Verify no prompt_runnable was created (no flow config)
        assert workflow_with_project._agent.prompt_runnable is None
