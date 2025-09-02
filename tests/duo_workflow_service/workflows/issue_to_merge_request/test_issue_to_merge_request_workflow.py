import asyncio
import os
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue
from langgraph.checkpoint.memory import MemorySaver

from ai_gateway.models.mock import FakeModel
from duo_workflow_service.agents.prompts import HANDOVER_TOOL_NAME
from duo_workflow_service.entities import Plan, WorkflowStatusEnum
from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.workflows.issue_to_merge_request.prompts import (
    BUILD_CONTEXT_SYSTEM_MESSAGE,
)
from duo_workflow_service.workflows.issue_to_merge_request.workflow import Workflow
from lib.internal_events.event_enum import CategoryEnum


@pytest.fixture(name="mock_executor_component")
def mock_executor_component_fixture():
    with patch(
        "duo_workflow_service.workflows.issue_to_merge_request.workflow.ExecutorComponent",
        autospec=True,
    ) as mock:
        mock.return_value.attach.return_value = "git_actions"
        yield mock


@pytest.fixture(name="mock_planner_component")
def mock_planner_component_fixture():
    with patch(
        "duo_workflow_service.workflows.issue_to_merge_request.workflow.PlannerComponent",
        autospec=True,
    ) as mock:
        mock.return_value.attach.return_value = "set_status_to_execution"
        yield mock


@pytest.fixture(name="mock_tools_approval_component")
def mock_tools_approval_component_fixture():
    with patch(
        "duo_workflow_service.workflows.issue_to_merge_request.workflow.ToolsApprovalComponent",
        autospec=True,
    ) as mock:
        yield mock


@pytest.fixture(name="mock_handover_agent")
def mock_handover_agent_fixture():
    with patch(
        "duo_workflow_service.workflows.issue_to_merge_request.workflow.HandoverAgent"
    ) as mock:
        mock.return_value.run.return_value = {
            "plan": Plan(steps=[]),
            "status": WorkflowStatusEnum.COMPLETED,
            "conversation_history": {},
        }
        yield mock


@pytest.fixture(name="agent_responses")
def agent_responses_fixture() -> list[dict[str, Any]]:
    return [
        {
            "plan": Plan(steps=[]),
            "status": WorkflowStatusEnum.NOT_STARTED,
            "conversation_history": {
                "context_builder": [
                    AIMessage(
                        content="Tool calls are present, route to planner tools execution",
                        tool_calls=[
                            {
                                "id": "1",
                                "name": "test_tool",
                                "args": {"test": "test"},
                            }
                        ],
                    ),
                ],
            },
        },
        {
            "plan": Plan(steps=[]),
            "status": WorkflowStatusEnum.PLANNING,
            "conversation_history": {
                "context_builder": [
                    AIMessage(
                        content="HandoverTool call is present, route to the next agent",
                        tool_calls=[
                            {
                                "id": "1",
                                "name": "handover_tool",
                                "args": {"summary": "done"},
                            }
                        ],
                    ),
                ],
            },
        },
    ]


@pytest.fixture(name="mock_agent")
def mock_agent_fixture(
    agent_responses: list[dict[str, Any]], duo_workflow_prompt_registry_enabled: bool
):
    if duo_workflow_prompt_registry_enabled:
        factory = "ai_gateway.prompts.registry.LocalPromptRegistry.get_on_behalf"
    else:
        factory = "duo_workflow_service.workflows.issue_to_merge_request.workflow.Agent"

    with patch(factory) as mock:
        mock.return_value.run.side_effect = agent_responses
        yield mock


@pytest.fixture(name="mock_run_tool_node_class")
def mock_run_tool_node_class_fixture():
    with patch(
        "duo_workflow_service.workflows.issue_to_merge_request.workflow.RunToolNode"
    ) as mock:
        yield mock.__getitem__.return_value


@pytest.fixture(name="mock_tools_executor")
def mock_tools_executor_fixture():
    with patch(
        "duo_workflow_service.workflows.issue_to_merge_request.workflow.ToolsExecutor"
    ) as mock:
        mock.return_value.run.return_value = {
            "plan": Plan(steps=[]),
            "status": WorkflowStatusEnum.NOT_STARTED,
            "conversation_history": {},
        }

        yield mock


@pytest.fixture(name="mock_chat_client")
def mock_chat_client_fixture():
    with patch(
        "duo_workflow_service.workflows.issue_to_merge_request.workflow.create_chat_model"
    ) as mock:
        mock.return_value = mock
        mock.bind_tools.return_value = mock
        yield mock


@pytest.fixture(name="mock_model_ainvoke")
def mock_model_ainvoke_fixture(
    duo_workflow_prompt_registry_enabled: bool, mock_chat_client: Mock
):
    end_message = AIMessage("done")

    if duo_workflow_prompt_registry_enabled:
        with patch.object(FakeModel, "ainvoke") as mock:
            mock.return_value = end_message
            yield mock
    else:
        mock_chat_client.ainvoke = AsyncMock(return_value=end_message)
        yield mock_chat_client.ainvoke


@pytest.fixture(name="workflow_type")
def workflow_type_fixture() -> CategoryEnum:
    return CategoryEnum.WORKFLOW_ISSUE_TO_MERGE_REQUEST


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
            workflow_metadata={"git_branch": "test-branch"},
            workflow_type=workflow_type,
            user=user,
        )
        workflow._project = project
        return workflow


@pytest.mark.asyncio
@pytest.mark.parametrize("offline_mode", [True])
@pytest.mark.parametrize("duo_workflow_prompt_registry_enabled", [False, True])
@patch("duo_workflow_service.workflows.abstract_workflow.UserInterface", autospec=True)
async def test_workflow_run(
    mock_checkpoint_notifier,
    mock_git_lab_workflow_instance,
    mock_chat_client,
    mock_fetch_workflow_and_container_data,
    mock_run_tool_node_class,
    mock_tools_executor,
    mock_tools_approval_component,
    mock_planner_component,
    mock_executor_component,
    mock_handover_agent,
    mock_agent,
    checkpoint_tuple,
    graph_input,
    workflow,
):
    mock_user_interface_instance = mock_checkpoint_notifier.return_value

    mock_tools_approval = mock_tools_approval_component.return_value
    mock_git_lab_workflow_instance.__aenter__.return_value = MemorySaver()

    mock_run_tool_node_class.return_value.run.return_value = {
        "command_output": ["test string"],
        "state": graph_input,
        "ui_chat_log": [],
    }

    await workflow.run("https://example.com/project/-/issues/1")

    mock_planner_component.return_value.attach.assert_called_once_with(
        graph=ANY,
        next_node="set_status_to_execution",
        exit_node="plan_terminator",
        approval_component=None,
    )

    mock_executor_component.return_value.attach.assert_called_once_with(
        graph=ANY,
        next_node="git_actions",
        exit_node="plan_terminator",
        approval_component=mock_tools_approval,
    )

    assert mock_agent.call_count == 1
    assert mock_agent.return_value.run.call_count == 2

    assert mock_tools_executor.call_count == 1
    assert mock_tools_executor.return_value.run.call_count >= 1

    assert mock_handover_agent.call_count == 2
    assert mock_handover_agent.return_value.run.call_count == 2

    assert mock_git_lab_workflow_instance.aput.call_count == 0
    assert mock_git_lab_workflow_instance.aget_tuple.call_count == 0

    mock_user_interface_instance.send_event.assert_called_with(
        type=ANY, state=ANY, stream=False
    )
    assert mock_user_interface_instance.send_event.call_count >= 2
    assert mock_run_tool_node_class.call_count == 1
    assert mock_run_tool_node_class.return_value.run.call_count == 1

    assert workflow.is_done


@pytest.mark.asyncio
@pytest.mark.parametrize("duo_workflow_prompt_registry_enabled", [False, True])
async def test_workflow_run_when_exception(
    mock_git_lab_workflow_instance,
    mock_chat_client,
    mock_fetch_workflow_and_container_data,
    mock_run_tool_node_class,
    mock_tools_executor,
    mock_handover_agent,
    mock_agent,
    mock_planner_component,
    mock_executor_component,
    workflow,
):
    class AsyncIterator:
        def __init__(self):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise asyncio.CancelledError()

    with patch(
        "duo_workflow_service.workflows.issue_to_merge_request.workflow.StateGraph"
    ) as graph:
        compiled_graph = MagicMock()
        compiled_graph.aget_state = AsyncMock(return_value=None)
        compiled_graph.astream.return_value = AsyncIterator()
        instance = graph.return_value
        instance.compile.return_value = compiled_graph
        await workflow.run("https://example.com/project/-/issues/1")

    assert workflow.is_done


@pytest.mark.asyncio
@pytest.mark.parametrize("duo_workflow_prompt_registry_enabled", [False, True])
@pytest.mark.usefixtures(
    "mock_gitlab_version",
    "mock_fetch_workflow_and_container_data",
    "mock_git_lab_workflow_instance",
)
async def test_messages_to_model(
    mock_model_ainvoke,
    goal,
    project,
    workflow,
):
    await workflow.run(goal)

    ainvoke_messages = mock_model_ainvoke.call_args.args[0]

    if isinstance(ainvoke_messages, ChatPromptValue):
        ainvoke_messages = ainvoke_messages.messages

    assert ainvoke_messages == [
        SystemMessage(
            content=BUILD_CONTEXT_SYSTEM_MESSAGE.format(
                handover_tool_name=HANDOVER_TOOL_NAME,
                issue_url=goal,
                current_branch=workflow._workflow_metadata["git_branch"],
                default_branch=project["default_branch"],  # type: ignore[index]
                project_id=project["id"],  # type: ignore[index]
                workflow_id=workflow._workflow_id,
                session_url=workflow._session_url,
            )
        ),
        HumanMessage(
            content=f"Your goal is: Consider the following issue url: {goal}. Build context and identify development "
            "tasks from the issue requirements."
        ),
    ]
