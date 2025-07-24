import asyncio
import json
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
)
from lib.internal_events.event_enum import CategoryEnum


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
def workflow():
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
    tools_registry_with_all_privileges, mock_checkpointer, workflow
):
    """Test that all tools used by the gitlab ci translator agent are available in the tools registry."""

    captured_tool_names = []

    # The ci translator agent is initialized with tools via `tools=tools_registry.get_batch(translator_tools),`
    with patch.object(
        tools_registry_with_all_privileges,
        "get_batch",
        side_effect=lambda tool_names: captured_tool_names.extend(tool_names),
    ):
        workflow._compile(
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
async def test_file_content_too_large(mock_token_counter, workflow_state):
    workflow_state["conversation_history"] = {"ci_pipelines_manager_agent": []}
    mock_token_counter.return_value.count_tokens.return_value = MAX_CONTEXT_TOKENS + 1

    result = _load_file_contents(["large file content"], workflow_state)

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
async def test_workflow_initialization(workflow):
    """Test workflow initialization and state setup."""
    initial_state = workflow.get_workflow_state("/test/path")
    assert initial_state["status"] == WorkflowStatusEnum.NOT_STARTED
    assert len(initial_state["ui_chat_log"]) == 1
    assert initial_state["conversation_history"] == {}


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.create_chat_model")
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.Agent")
async def test_workflow_compilation(
    mock_agent, mock_new_chat_client, mock_tools_registry, mock_checkpointer, workflow
):
    """Test workflow compilation process."""
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
@pytest.mark.usefixtures(
    "mock_tools_registry_cls", "mock_fetch_workflow_and_container_data"
)
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.Agent")
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.HandoverAgent")
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.ToolsExecutor")
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.RunToolNode")
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.create_chat_model")
async def test_workflow_run(
    mock_chat_client,
    mock_run_tool_node_generic_class,
    mock_tools_executor,
    mock_handover_agent,
    mock_agent,
    mock_checkpoint_notifier,
    mock_agent_response,
    workflow_state,
    mock_git_lab_workflow_instance,
    workflow,
):
    mock_checkpoint_notifier_instance = mock_checkpoint_notifier.return_value

    mock_tools_executor.return_value.run.return_value = {
        "plan": Plan(steps=[]),
        "status": WorkflowStatusEnum.EXECUTION,
        "conversation_history": {},
    }

    mock_run_tool_node_class = mock_run_tool_node_generic_class.__getitem__.return_value
    mock_run_tool_node_class.return_value.run.side_effect = [
        {
            "file_contents": ["test string"],
            "state": workflow_state,
            "ui_chat_log": [],
        },
        {
            "command_output": ["test string"],
            "state": workflow_state,
            "ui_chat_log": [],
        },
    ]

    mock_handover_agent.return_value.run.return_value = {
        "plan": Plan(steps=[]),
        "status": WorkflowStatusEnum.COMPLETED,
        "conversation_history": {},
    }

    mock_agent.return_value.run.return_value = mock_agent_response

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
@pytest.mark.usefixtures(
    "mock_tools_registry_cls", "mock_fetch_workflow_and_container_data"
)
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.Agent")
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.RunToolNode")
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.create_chat_model")
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.log_exception")
async def test_workflow_run_with_file_not_found(
    mock_log_exception,
    mock_chat_client,
    mock_run_tool_node_generic_class,
    mock_agent,
    mock_checkpoint_notifier,
    mock_git_lab_workflow_instance,
    workflow,
):
    mock_checkpoint_notifier_instance = mock_checkpoint_notifier.return_value

    mock_run_tool_node_class = mock_run_tool_node_generic_class.__getitem__.return_value
    mock_run_tool_node_class.return_value.run = MagicMock(
        side_effect=RuntimeError(
            "Failed to load file contents, ensure that file is present"
        )
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
@pytest.mark.usefixtures(
    "mock_tools_registry_cls",
    "mock_fetch_workflow_and_container_data",
    "mock_git_lab_workflow_instance",
)
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.create_chat_model")
async def test_workflow_run_with_exception(mock_chat_client, workflow):

    class AsyncIterator:
        def __init__(self):
            pass

        def __aiter__(self):
            return self

        def __anext__(self):
            raise asyncio.CancelledError()

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


def test_router_ci_linter_validation_success():
    """Test router handles successful ci_linter validation."""
    state = WorkflowState(
        status=WorkflowStatusEnum.EXECUTION,
        conversation_history={
            "ci_pipelines_manager_agent": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ci_linter",
                            "args": {"project_id": 123, "content": "..."},
                            "id": "1",
                        }
                    ],
                ),
                AIMessage(content='{"valid": true}'),
            ]
        },
        plan=Plan(steps=[]),
        handover=[],
        last_human_input=None,
        ui_chat_log=[],
    )

    assert _router(state) == Routes.COMMIT_CHANGES


def test_router_ci_linter_validation_failure():
    """Test router handles failed ci_linter validation."""
    state = WorkflowState(
        status=WorkflowStatusEnum.EXECUTION,
        conversation_history={
            "ci_pipelines_manager_agent": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ci_linter",
                            "args": {"project_id": 123, "content": "..."},
                            "id": "1",
                        }
                    ],
                ),
                AIMessage(content='{"valid": false, "errors": ["syntax error"]}'),
            ]
        },
        plan=Plan(steps=[]),
        handover=[],
        last_human_input=None,
        ui_chat_log=[],
    )

    assert _router(state) == Routes.AGENT


def test_router_ci_linter_max_attempts():
    """Test router handles max validation attempts."""
    messages = []
    # Add 3 ci_linter calls
    for i in range(3):
        messages.extend(
            [
                AIMessage(
                    content=f"attempt {i}",
                    tool_calls=[
                        {
                            "name": "ci_linter",
                            "args": {"project_id": 123, "content": "..."},
                            "id": str(i),
                        }
                    ],
                ),
                AIMessage(content='{"valid": false}'),
            ]
        )

    state = WorkflowState(
        status=WorkflowStatusEnum.EXECUTION,
        conversation_history={"ci_pipelines_manager_agent": messages},
        plan=Plan(steps=[]),
        handover=[],
        last_human_input=None,
        ui_chat_log=[],
    )

    assert _router(state) == Routes.COMMIT_CHANGES


def test_router_create_file_returns_to_agent():
    """Test router returns to agent after file creation for validation."""
    state = WorkflowState(
        status=WorkflowStatusEnum.EXECUTION,
        conversation_history={
            "ci_pipelines_manager_agent": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "create_file_with_contents",
                            "args": {"file_path": ".gitlab-ci.yml", "contents": "..."},
                            "id": "1",
                        }
                    ],
                ),
                AIMessage(content="File created"),
            ]
        },
        plan=Plan(steps=[]),
        handover=[],
        last_human_input=None,
        ui_chat_log=[],
    )

    assert _router(state) == Routes.AGENT


@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.log_exception")
def test_router_ci_linter_json_parsing_error(mock_log_exception):
    """Test router handles JSON parsing errors in ci_linter responses."""
    state = WorkflowState(
        status=WorkflowStatusEnum.EXECUTION,
        conversation_history={
            "ci_pipelines_manager_agent": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ci_linter",
                            "args": {"project_id": 123, "content": "..."},
                            "id": "1",
                        }
                    ],
                ),
                AIMessage(content="This is not valid JSON"),
            ]
        },
        plan=Plan(steps=[]),
        handover=[],
        last_human_input=None,
        ui_chat_log=[],
        files_changed=[],
    )

    assert _router(state) == Routes.AGENT

    mock_log_exception.assert_called_once()
    args, kwargs = mock_log_exception.call_args

    assert isinstance(args[0], json.JSONDecodeError)
    assert kwargs["extra"]["tool_name"] == "ci_linter"
    assert kwargs["extra"]["last_msg"] == "This is not valid JSON"
    assert kwargs["extra"]["error_type"] == "json_parsing_error"


def test_router_create_file_max_attempts():
    """Test router prevents infinite file creation loops."""
    messages = []
    for i in range(3):
        messages.extend(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "create_file_with_contents",
                            "args": {"file_path": f"file{i}.yml", "contents": "..."},
                            "id": str(i),
                        }
                    ],
                ),
                AIMessage(content="File created"),
            ]
        )

    state = WorkflowState(
        status=WorkflowStatusEnum.EXECUTION,
        conversation_history={"ci_pipelines_manager_agent": messages},
        plan=Plan(steps=[]),
        handover=[],
        last_human_input=None,
        ui_chat_log=[],
        files_changed=[],
    )

    assert _router(state) == Routes.COMMIT_CHANGES
