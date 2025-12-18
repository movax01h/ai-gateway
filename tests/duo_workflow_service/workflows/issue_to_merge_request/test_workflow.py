import asyncio
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from duo_workflow_service.entities import Plan, WorkflowStatusEnum
from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
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


@pytest.fixture(name="workflow_type")
def workflow_type_fixture() -> CategoryEnum:
    return CategoryEnum.WORKFLOW_ISSUE_TO_MERGE_REQUEST


@pytest.fixture(name="workflow")
def workflow_fixture(
    mock_duo_workflow_service_container: Mock,  # pylint: disable=unused-argument
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
@patch("duo_workflow_service.workflows.abstract_workflow.UserInterface", autospec=True)
async def test_workflow_run(
    mock_checkpoint_notifier,
    mock_git_lab_workflow_instance,
    mock_fetch_workflow_and_container_data,  # pylint: disable=unused-argument
    mock_run_tool_node_class,
    mock_tools_executor,
    mock_tools_approval_component,
    mock_planner_component,
    mock_executor_component,
    mock_handover_agent,
    mock_agent,
    checkpoint_tuple,  # pylint: disable=unused-argument
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

    assert mock_agent.run.call_count == 2

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
async def test_workflow_run_when_exception(
    mock_fetch_workflow_and_container_data,  # pylint: disable=unused-argument
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
