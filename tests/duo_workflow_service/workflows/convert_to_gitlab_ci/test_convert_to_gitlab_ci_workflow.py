import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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
from duo_workflow_service.internal_events.event_enum import CategoryEnum
from duo_workflow_service.workflows.convert_to_gitlab_ci import Workflow
from duo_workflow_service.workflows.convert_to_gitlab_ci.prompts import (
    CI_PIPELINES_MANAGER_FILE_USER_MESSAGE,
    CI_PIPELINES_MANAGER_SYSTEM_MESSAGE,
    CI_PIPELINES_MANAGER_USER_GUIDELINES,
)
from duo_workflow_service.workflows.convert_to_gitlab_ci.workflow import (
    _load_file_contents,
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


@pytest.fixture
def mock_agent_response():
    return {
        "plan": Plan(steps=[]),
        "status": WorkflowStatusEnum.EXECUTION,
        "conversation_history": {
            "ci_pipelines_manager_agent": [
                SystemMessage(content="system message"),
                HumanMessage(content="human message"),
                AIMessage(
                    content="I'll help translate this Jenkins pipeline to a GitLab CI/CD configuration",
                    tool_calls=[
                        {
                            "id": "1",
                            "name": "create_file_with_contents",
                            "args": {
                                "file_path": "gitlab-ci.yml",
                                "contents": "Translated .gitlab-ci file content",
                            },
                        }
                    ],
                ),
            ],
        },
    }


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.create_chat_model")
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
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.create_chat_model")
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
        toolset=mock_tools_registry.toolset.return_value,
        model=mock_new_chat_client.return_value,
        workflow_id="test_id",
        http_client=workflow._http_client,
        workflow_type=CategoryEnum.WORKFLOW_CONVERT_TO_GITLAB_CI.value,
    )
    mock_tools_registry.get.assert_called()  # Should call get() for tools
    mock_tools_registry.toolset.assert_called()


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry", autospec=True)
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.Agent")
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.HandoverAgent")
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.ToolsExecutor")
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.RunToolNode")
@patch(
    "duo_workflow_service.workflows.abstract_workflow.fetch_project_data_with_workflow_id"
)
@patch("duo_workflow_service.workflows.abstract_workflow.fetch_workflow_config")
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.create_chat_model")
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow", autospec=True)
@patch("duo_workflow_service.workflows.abstract_workflow.UserInterface", autospec=True)
async def test_workflow_run(
    mock_checkpoint_notifier,
    mock_gitlab_workflow,
    mock_chat_client,
    mock_fetch_workflow_config,
    mock_fetch_project_data_with_workflow_id,
    mock_run_tool_node_generic_class,
    mock_tools_executor,
    mock_handover_agent,
    mock_agent,
    mock_tools_registry_cls,
    mock_agent_response,
    mock_state,
):
    mock_checkpoint_notifier_instance = mock_checkpoint_notifier.return_value
    mock_tools_registry = MagicMock(spec=ToolsRegistry)
    mock_tools_registry_cls.configure = AsyncMock(return_value=mock_tools_registry)
    mock_fetch_project_data_with_workflow_id.return_value = {
        "id": 1,
        "name": "test-project",
        "description": "This is a test project",
        "http_url_to_repo": "https://example.com/project",
        "web_url": "https://example.com/project",
    }

    mock_git_lab_workflow_instance = mock_gitlab_workflow.return_value
    mock_git_lab_workflow_instance.__aenter__.return_value = (
        mock_git_lab_workflow_instance
    )
    mock_git_lab_workflow_instance.__aexit__.return_value = None
    mock_git_lab_workflow_instance._offline_mode = False
    mock_git_lab_workflow_instance.aget_tuple = AsyncMock(return_value=None)
    mock_git_lab_workflow_instance.alist = AsyncMock(return_value=[])
    mock_git_lab_workflow_instance.aput = AsyncMock(
        return_value={
            "configurable": {"thread_id": "123", "checkpoint_id": "checkpoint1"}
        }
    )
    mock_git_lab_workflow_instance.get_next_version = MagicMock(return_value=1)

    mock_tools_executor.return_value.run.return_value = {
        "plan": Plan(steps=[]),
        "status": WorkflowStatusEnum.EXECUTION,
        "conversation_history": {},
    }

    mock_run_tool_node_class = mock_run_tool_node_generic_class.__getitem__.return_value
    mock_run_tool_node_class.return_value.run.side_effect = [
        {
            "file_contents": ["test string"],
            "state": mock_state,
            "ui_chat_log": [],
        },
        {
            "command_output": ["test string"],
            "state": mock_state,
            "ui_chat_log": [],
        },
    ]

    mock_handover_agent.return_value.run.return_value = {
        "plan": Plan(steps=[]),
        "status": WorkflowStatusEnum.COMPLETED,
        "conversation_history": {},
    }

    mock_agent.return_value.run.return_value = mock_agent_response

    workflow = Workflow(
        "123",
        {},
        workflow_type=CategoryEnum.WORKFLOW_CONVERT_TO_GITLAB_CI,
    )
    await workflow.run("test-file-path")

    assert mock_agent.call_count == 1
    assert mock_agent.return_value.run.call_count == 1

    assert mock_tools_executor.call_count == 1
    assert mock_tools_executor.return_value.run.call_count == 1

    assert mock_handover_agent.call_count == 1
    assert mock_handover_agent.return_value.run.call_count == 1

    assert mock_run_tool_node_class.call_count == 2
    assert mock_run_tool_node_class.return_value.run.call_count == 1

    assert mock_git_lab_workflow_instance.aput.call_count >= 2
    assert mock_git_lab_workflow_instance.aget_tuple.call_count == 2

    assert mock_checkpoint_notifier_instance.send_event.call_count >= 2

    assert workflow.is_done


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry", autospec=True)
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.Agent")
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.RunToolNode")
@patch(
    "duo_workflow_service.workflows.abstract_workflow.fetch_project_data_with_workflow_id"
)
@patch("duo_workflow_service.workflows.abstract_workflow.fetch_workflow_config")
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.create_chat_model")
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow", autospec=True)
@patch("duo_workflow_service.workflows.abstract_workflow.UserInterface", autospec=True)
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.log_exception")
async def test_workflow_run_with_file_not_found(
    mock_log_exception,
    mock_checkpoint_notifier,
    mock_gitlab_workflow,
    mock_chat_client,
    mock_fetch_workflow_config,
    mock_fetch_project_data_with_workflow_id,
    mock_run_tool_node_generic_class,
    mock_agent,
    mock_tools_registry_cls,
    mock_state,
):
    mock_checkpoint_notifier_instance = mock_checkpoint_notifier.return_value
    mock_tools_registry = MagicMock(spec=ToolsRegistry)
    mock_tools_registry_cls.configure = AsyncMock(return_value=mock_tools_registry)
    mock_fetch_project_data_with_workflow_id.return_value = {
        "id": 1,
        "name": "test-project",
        "description": "This is a test project",
        "http_url_to_repo": "https://example.com/project",
        "web_url": "https://example.com/project",
    }

    mock_git_lab_workflow_instance = mock_gitlab_workflow.return_value
    mock_git_lab_workflow_instance.__aenter__.return_value = (
        mock_git_lab_workflow_instance
    )
    mock_git_lab_workflow_instance.__aexit__.return_value = None
    mock_git_lab_workflow_instance._offline_mode = False
    mock_git_lab_workflow_instance.aget_tuple = AsyncMock(return_value=None)
    mock_git_lab_workflow_instance.alist = AsyncMock(return_value=[])
    mock_git_lab_workflow_instance.aput = AsyncMock(
        return_value={
            "configurable": {"thread_id": "123", "checkpoint_id": "checkpoint1"}
        }
    )
    mock_git_lab_workflow_instance.get_next_version = MagicMock(return_value=1)

    mock_run_tool_node_class = mock_run_tool_node_generic_class.__getitem__.return_value
    mock_run_tool_node_class.return_value.run = MagicMock(
        side_effect=RuntimeError(
            "Failed to load file contents, ensure that file is present"
        )
    )

    workflow = Workflow(
        "123",
        {},
        workflow_type=CategoryEnum.WORKFLOW_CONVERT_TO_GITLAB_CI,
    )
    await workflow.run("test-file-path")

    assert mock_log_exception.call_count == 1
    assert mock_run_tool_node_class.call_count == 2
    assert mock_run_tool_node_class.return_value.run.call_count == 1

    assert mock_git_lab_workflow_instance.aput.call_count == 2
    assert mock_git_lab_workflow_instance.aget_tuple.call_count == 2

    assert mock_checkpoint_notifier_instance.send_event.call_count == 1

    assert workflow.is_done


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry", autospec=True)
@patch(
    "duo_workflow_service.workflows.abstract_workflow.fetch_project_data_with_workflow_id"
)
@patch("duo_workflow_service.workflows.abstract_workflow.fetch_workflow_config")
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.create_chat_model")
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow", autospec=True)
async def test_workflow_run_with_exception(
    mock_gitlab_workflow,
    mock_chat_client,
    mock_fetch_workflow_config,
    mock_fetch_project_data_with_workflow_id,
    mock_tools_registry_cls,
    mock_state,
):
    mock_tools_registry = MagicMock(spec=ToolsRegistry)
    mock_tools_registry_cls.configure = AsyncMock(return_value=mock_tools_registry)
    mock_fetch_project_data_with_workflow_id.return_value = {
        "id": 1,
        "name": "test-project",
        "description": "This is a test project",
        "http_url_to_repo": "https://example.com/project",
        "web_url": "https://example.com/project",
    }

    mock_git_lab_workflow_instance = mock_gitlab_workflow.return_value
    mock_git_lab_workflow_instance.__aenter__.return_value = (
        mock_git_lab_workflow_instance
    )
    mock_git_lab_workflow_instance.__aexit__.return_value = None
    mock_git_lab_workflow_instance._offline_mode = False
    mock_git_lab_workflow_instance.aget_tuple = AsyncMock(return_value=None)
    mock_git_lab_workflow_instance.alist = AsyncMock(return_value=[])
    mock_git_lab_workflow_instance.aput = AsyncMock(
        return_value={
            "configurable": {"thread_id": "123", "checkpoint_id": "checkpoint1"}
        }
    )
    mock_git_lab_workflow_instance.get_next_version = MagicMock(return_value=1)

    class AsyncIterator:
        def __init__(self):
            pass

        def __aiter__(self):
            return self

        def __anext__(self):
            raise asyncio.CancelledError()

    workflow = Workflow(
        "123",
        {},
        workflow_type=CategoryEnum.WORKFLOW_CONVERT_TO_GITLAB_CI,
    )
    with patch(
        "duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.StateGraph"
    ) as graph:
        compiled_graph = MagicMock()
        compiled_graph.aget_state = AsyncMock(return_value=None)
        compiled_graph.astream.return_value = AsyncIterator()
        instance = graph.return_value
        instance.compile.return_value = compiled_graph
        await workflow.run("test-file-path")

    assert workflow.is_done
