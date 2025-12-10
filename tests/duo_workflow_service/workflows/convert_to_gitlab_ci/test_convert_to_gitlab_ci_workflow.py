import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from duo_workflow_service.components.tools_registry import (
    _AGENT_PRIVILEGES,
    ToolsRegistry,
)
from duo_workflow_service.entities import Plan, WorkflowState, WorkflowStatusEnum
from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.workflows.convert_to_gitlab_ci import Workflow
from duo_workflow_service.workflows.convert_to_gitlab_ci.workflow import (
    DEFAULT_MAX_CONTEXT_TOKENS,
    Routes,
    _router,
)
from duo_workflow_service.workflows.type_definitions import AdditionalContext
from lib.internal_events.event_enum import CategoryEnum


@pytest.fixture(name="tools_registry_with_all_privileges")
def tools_registry_with_all_privileges_fixture(tool_metadata):
    return ToolsRegistry(
        enabled_tools=list(_AGENT_PRIVILEGES.keys()),
        preapproved_tools=list(_AGENT_PRIVILEGES.keys()),
        tool_metadata=tool_metadata,
    )


@pytest.fixture(name="mock_checkpointer")
def mock_checkpointer_fixture():
    """Create a mock checkpointer."""
    return Mock()


@pytest.fixture(name="mock_build_agent")
def mock_build_agent():
    with patch(
        "duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.build_agent"
    ) as mock:
        yield mock


@pytest.fixture(name="workflow_type")
def workflow_type_fixture() -> CategoryEnum:
    return CategoryEnum.WORKFLOW_CONVERT_TO_GITLAB_CI


@pytest.fixture(name="workflow")
def workflow_fixture(
    mock_duo_workflow_service_container: Mock,
    workflow_type: CategoryEnum,
    user: CloudConnectorUser,
    gl_http_client: GitlabHttpClient,
    project: Project,
):
    with patch(
        "duo_workflow_service.workflows.abstract_workflow.get_http_client",
        return_value=gl_http_client,
    ):
        workflow = Workflow(
            workflow_id="test_id",
            workflow_metadata={},
            workflow_type=workflow_type,
            user=user,
        )
        workflow._project = project
        return workflow


@pytest.fixture(name="workflow_with_source_branch")
def workflow_with_source_branch_fixture(workflow):
    """Create a workflow instance with source branch in additional context."""
    workflow._additional_context = [
        AdditionalContext(
            category="agent_user_environment",
            content='{"source_branch": "feature-branch"}',
        )
    ]
    return workflow


@pytest.fixture(name="mock_agent_response")
def mock_agent_response_fixture():
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


@pytest.fixture(name="mock_agent")
def mock_agent_fixture(mock_agent_response: dict[str, Any]):
    with patch("duo_workflow_service.agents.agent.Agent") as mock:
        mock.return_value.run.return_value = mock_agent_response
        yield mock


@pytest.fixture(name="mock_run_tool_node_class")
def mock_run_tool_node_class_fixture():
    with patch(
        "duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.RunToolNode"
    ) as mock:
        yield mock.__getitem__.return_value


@pytest.fixture(name="mock_log_exception")
def mock_log_exception_fixture():
    with patch(
        "duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.log_exception"
    ) as mock:
        yield mock


@pytest.fixture(name="anthropic_env")
def setup_anthropic_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def test_get_source_branch_with_context(workflow_with_source_branch):
    """Test get_source_branch returns correct branch when context is present."""
    assert workflow_with_source_branch.get_source_branch() == "feature-branch"


def test_get_source_branch_without_context(workflow):
    """Test get_source_branch returns None when no additional context."""

    assert workflow.get_source_branch() is None


def test_get_source_branch_with_different_category(workflow):
    """Test get_source_branch returns None when context has different category."""
    workflow._additional_context = [
        AdditionalContext(
            category="other_category", content='{"source_branch": "main"}'
        )
    ]

    assert workflow.get_source_branch() is None


@pytest.mark.asyncio
async def test_git_push_with_source_branch(
    mock_run_tool_node_class,
    mock_agent,
    mock_tools_registry,
    mock_checkpointer,
    workflow_with_source_branch,
):
    """Test git push command with merge request target when source branch exists."""
    push_command = _get_push_command(
        mock_run_tool_node_class,
        mock_tools_registry,
        mock_checkpointer,
        workflow_with_source_branch,
    )

    expected_args = (
        "-o merge_request.create "
        "-o merge_request.title='Duo Agent: Convert to GitLab CI' "
        "-o merge_request.description='Created by Duo Agent, session: test_id' "
        "-o merge_request.target=feature-branch"
    )
    assert push_command["command"] == "push"
    assert push_command["args"].strip() == expected_args


@pytest.mark.asyncio
async def test_git_push_without_source_branch(
    mock_run_tool_node_class,
    mock_agent,
    mock_tools_registry,
    mock_checkpointer,
    workflow,
):
    """Test git push command without merge request target when no source branch."""
    push_command = _get_push_command(
        mock_run_tool_node_class,
        mock_tools_registry,
        mock_checkpointer,
        workflow,
    )

    expected_args = (
        "-o merge_request.create "
        "-o merge_request.title='Duo Agent: Convert to GitLab CI' "
        "-o merge_request.description='Created by Duo Agent, session: test_id'"
    )
    assert push_command["command"] == "push"
    assert push_command["args"].strip() == expected_args


def _get_push_command(
    mock_run_tool_node_class,
    mock_tools_registry,
    mock_checkpointer,
    workflow,
):
    """Helper function to extract push command from workflow compilation."""
    mock_node = Mock(run=Mock(return_value={}))
    mock_run_tool_node_class.return_value = mock_node

    workflow._compile(
        goal="/test/path",
        tools_registry=mock_tools_registry,
        checkpointer=mock_checkpointer,
    )

    git_call = next(
        (
            call
            for call in mock_run_tool_node_class.call_args_list
            if "_git_output" in str(call.kwargs.get("output_parser", ""))
        ),
        None,
    )

    if git_call is None:
        raise ValueError("No git call found in mock calls")

    git_commands = git_call.kwargs["input_parser"](None)

    push_cmd = next((cmd for cmd in git_commands if cmd["command"] == "push"), None)
    if push_cmd is None:
        raise ValueError("No push command found in git commands")

    return push_cmd


@pytest.mark.asyncio
async def test_translation_tools(
    anthropic_env, tools_registry_with_all_privileges, mock_checkpointer, workflow
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


@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.TikTokenCounter")
@pytest.mark.asyncio
async def test_file_content_too_large(mock_token_counter, workflow_state, workflow):
    max_single_message_tokens = int(DEFAULT_MAX_CONTEXT_TOKENS * 0.65)
    mock_token_counter.return_value.count_string_content.return_value = (
        max_single_message_tokens + 1
    )

    result = workflow._load_file_contents(["large file content"], workflow_state)

    mock_token_counter.assert_called_once_with("ci_pipelines_manager_agent")
    mock_token_counter.return_value.count_string_content.assert_called_once_with(
        "large file content"
    )
    assert "File too large" in result["ui_chat_log"][0]["content"]
    assert "conversation_history" not in result
    assert "additional_context" not in result


@pytest.mark.asyncio
async def test_workflow_initialization(workflow):
    """Test workflow initialization and state setup."""
    initial_state = workflow.get_workflow_state("/test/path")
    assert initial_state["status"] == WorkflowStatusEnum.NOT_STARTED
    assert len(initial_state["ui_chat_log"]) == 1
    assert initial_state["conversation_history"] == {}


@pytest.mark.asyncio
async def test_workflow_compilation(
    mock_build_agent,
    mock_tools_registry,
    mock_checkpointer,
    workflow_type,
    workflow,
):
    """Test workflow compilation process."""
    # Compile the workflow graph
    compiled_graph = workflow._compile(
        goal="/test/path",
        tools_registry=mock_tools_registry,
        checkpointer=mock_checkpointer,
    )

    assert compiled_graph is not None
    mock_build_agent.assert_called_with(
        "ci_pipelines_manager_agent",
        workflow._prompt_registry,
        workflow._user,
        "workflow/convert_to_gitlab_ci",
        "^1.0.0",
        tools=mock_tools_registry.toolset.return_value.bindable,
        workflow_id=workflow._workflow_id,
        workflow_type=workflow_type,
        http_client=workflow._http_client,
    )
    mock_tools_registry.get.assert_called()  # Should call get() for tools
    mock_tools_registry.toolset.assert_called()


@pytest.mark.asyncio
@pytest.mark.usefixtures(
    "mock_tools_registry_cls", "mock_fetch_workflow_and_container_data"
)
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.HandoverAgent")
@patch("duo_workflow_service.workflows.convert_to_gitlab_ci.workflow.ToolsExecutor")
async def test_workflow_run(
    mock_tools_executor,
    mock_handover_agent,
    mock_run_tool_node_class,
    mock_agent,
    mock_checkpoint_notifier,
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
    assert mock_git_lab_workflow_instance.aget_tuple.call_count == 1

    assert mock_checkpoint_notifier_instance.send_event.call_count >= 2

    assert workflow.is_done


@pytest.mark.asyncio
@pytest.mark.usefixtures(
    "mock_tools_registry_cls", "mock_fetch_workflow_and_container_data"
)
async def test_workflow_run_with_file_not_found(
    mock_log_exception,
    mock_run_tool_node_class,
    mock_agent,
    mock_checkpoint_notifier,
    mock_git_lab_workflow_instance,
    workflow,
):
    mock_checkpoint_notifier_instance = mock_checkpoint_notifier.return_value

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
    assert mock_git_lab_workflow_instance.aget_tuple.call_count == 1

    assert mock_checkpoint_notifier_instance.send_event.call_count == 1

    assert workflow.is_done


@pytest.mark.asyncio
@pytest.mark.usefixtures(
    "mock_tools_registry_cls",
    "mock_fetch_workflow_and_container_data",
    "mock_git_lab_workflow_instance",
)
async def test_workflow_run_with_exception(workflow):

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
