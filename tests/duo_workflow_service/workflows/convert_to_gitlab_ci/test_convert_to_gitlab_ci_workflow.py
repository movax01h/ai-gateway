from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from duo_workflow_service.components.tools_registry import (
    _AGENT_PRIVILEGES,
    ToolsRegistry,
)
from duo_workflow_service.entities import (
    MAX_CONTEXT_TOKENS,
    Plan,
    WorkflowState,
    WorkflowStatusEnum,
)
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.internal_events.event_enum import CategoryEnum
from duo_workflow_service.workflows.convert_to_gitlab_ci import Workflow
from duo_workflow_service.workflows.convert_to_gitlab_ci.prompts import (
    CI_PIPELINES_MANAGER_FILE_USER_MESSAGE,
    CI_PIPELINES_MANAGER_SYSTEM_MESSAGE,
    CI_PIPELINES_MANAGER_USER_GUIDELINES,
)
from duo_workflow_service.workflows.convert_to_gitlab_ci.workflow import (
    Routes,
    _load_file_contents,
    _router,
    _tools_execution_requested,
)


@pytest.fixture
def tools_registry_with_all_privileges(tool_metadata):
    return ToolsRegistry(
        enabled_tools=list(_AGENT_PRIVILEGES.keys()),
        preapproved_tools=list(_AGENT_PRIVILEGES.keys()),
        tool_metadata=tool_metadata,
    )


@pytest.fixture
def mock_checkpointer():
    """Create a mock checkpointer."""
    return Mock()


@pytest.fixture
def mock_state():
    """Create a mock workflow state."""
    return WorkflowState(
        plan=Plan(steps=[]),
        status=WorkflowStatusEnum.NOT_STARTED,
        conversation_history={},
        handover=[],
        last_human_input=None,
        ui_chat_log=[],
    )


@pytest.fixture
def mock_tools_registry():
    """Create a mock tools registry."""
    registry = Mock()
    registry.get = Mock(return_value=Mock(name="test_tool"))
    registry.get_batch = Mock(return_value=[Mock(name="test_tool")])
    registry.get_handlers = Mock(return_value=[Mock(name="test_tool")])
    return registry


@pytest.fixture
def mock_workflow():
    return Workflow(
        workflow_id="test_id",
        workflow_metadata={},
        workflow_type=CategoryEnum.WORKFLOW_CONVERT_TO_GITLAB_CI,
    )


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.new_chat_client")
async def test_translation_tools(
    tools_registry_with_all_privileges, mock_checkpointer, mock_workflow
):
    """Test that all tools used by the gitlab ci translator agent are available in the tools registry."""

    captured_tool_names = []

    # The ci translator agent is initialized with tools via `tools=tools_registry.get_batch(translator_tools),`
    with patch.object(
        tools_registry_with_all_privileges,
        "get_batch",
        side_effect=lambda tool_names: captured_tool_names.extend(tool_names),
    ):
        mock_workflow._compile(
            goal="/test/path",
            tools_registry=tools_registry_with_all_privileges,
            checkpointer=mock_checkpointer,
        )

    missing_tools = []
    for tool_name in captured_tool_names:
        if tools_registry_with_all_privileges.get(tool_name) is None:
            missing_tools.append(tool_name)

    assert (
        not missing_tools
    ), f"The following tools are missing from the tools registry: {missing_tools}"


def test_router(mock_state):
    mock_state["conversation_history"] = {
        "ci_pipelines_manager_agent": [
            AIMessage(
                content="test", tool_calls=[{"id": "123", "name": "test", "args": {}}]
            )
        ]
    }
    assert _router(mock_state) == Routes.END

    mock_state["conversation_history"] = {
        "ci_pipelines_manager_agent": [
            AIMessage(
                content="test",
                tool_calls=[{"id": "123", "name": "read_file", "args": {}}],
            ),
            ToolMessage(
                content="test",
                tool_call_id="1",
            ),
        ]
    }
    assert _router(mock_state) == Routes.AGENT

    mock_state["status"] = WorkflowStatusEnum.CANCELLED
    assert _router(mock_state) == Routes.END


def test_tools_execution_requested(mock_state):
    assert _tools_execution_requested(mock_state) == Routes.END

    mock_state["conversation_history"] = {
        "ci_pipelines_manager_agent": [
            AIMessage(
                content="test", tool_calls=[{"id": "123", "name": "test", "args": {}}]
            )
        ]
    }
    assert _tools_execution_requested(mock_state) == Routes.CONTINUE

    mock_state["status"] = WorkflowStatusEnum.CANCELLED
    assert _tools_execution_requested(mock_state) == Routes.END


@patch(
    "duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.ApproximateTokenCounter"
)
@pytest.mark.asyncio
async def test_file_content_too_large(mock_token_counter, mock_state):
    mock_state["conversation_history"] = {"ci_pipelines_manager_agent": []}
    mock_token_counter.return_value.count_tokens.return_value = MAX_CONTEXT_TOKENS + 1

    result = _load_file_contents(["large file content"], mock_state)

    human_prompt = CI_PIPELINES_MANAGER_FILE_USER_MESSAGE.format(
        file_content="large file content"
    )
    mock_token_counter.assert_called_once_with("ci_pipelines_manager_agent")
    mock_token_counter.return_value.count_tokens.assert_called_once_with(
        [
            SystemMessage(content=CI_PIPELINES_MANAGER_SYSTEM_MESSAGE),
            HumanMessage(content=CI_PIPELINES_MANAGER_USER_GUIDELINES),
            HumanMessage(content=human_prompt),
        ]
    )
    assert "File too large" in result["ui_chat_log"][0]["content"]
    assert result["conversation_history"].get("ci_pipelines_manager_agent") == []


@pytest.mark.asyncio
async def test_workflow_initialization():
    """Test workflow initialization and state setup."""
    workflow = Workflow(
        workflow_id="test_id",
        workflow_metadata={},
        workflow_type=CategoryEnum.WORKFLOW_CONVERT_TO_GITLAB_CI,
    )
    initial_state = workflow.get_workflow_state("/test/path")
    assert initial_state["status"] == WorkflowStatusEnum.NOT_STARTED
    assert len(initial_state["ui_chat_log"]) == 1
    assert initial_state["conversation_history"] == {}


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.new_chat_client")
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.Agent")
async def test_workflow_compilation(
    mock_agent, mock_new_chat_client, mock_tools_registry, mock_checkpointer
):
    """Test workflow compilation process."""
    workflow = Workflow(
        workflow_id="test_id",
        workflow_metadata={},
        workflow_type=CategoryEnum.WORKFLOW_CONVERT_TO_GITLAB_CI,
    )

    # Compile the workflow graph
    compiled_graph = workflow._compile(
        goal="/test/path",
        tools_registry=mock_tools_registry,
        checkpointer=mock_checkpointer,
    )

    assert compiled_graph is not None
    mock_agent.assert_called_with(
        goal="N/A",
        system_prompt="N/A",
        name="ci_pipelines_manager_agent",
        tools=mock_tools_registry.get_batch.return_value,
        model=mock_new_chat_client.return_value,
        workflow_id="test_id",
        http_client=workflow._http_client,
        workflow_type=CategoryEnum.WORKFLOW_CONVERT_TO_GITLAB_CI.value,
    )
    mock_tools_registry.get.assert_called()  # Should call get() for tools
    mock_tools_registry.get_batch.assert_called_once()  # Should get batch of tools for agent
    mock_tools_registry.get_handlers.assert_called_once()  # Should get handlers for executor
