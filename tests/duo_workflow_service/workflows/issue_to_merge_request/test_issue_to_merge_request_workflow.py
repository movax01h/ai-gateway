import asyncio
import os
from unittest.mock import ANY, AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.memory import MemorySaver

from duo_workflow_service.components import ToolsRegistry
from duo_workflow_service.entities import Plan, WorkflowStatusEnum
from duo_workflow_service.workflows.issue_to_merge_request.workflow import Workflow
from lib.internal_events.event_enum import CategoryEnum


@pytest.fixture(name="checkpoint_tuple")
def checkpoint_tuple_fixture():
    return CheckpointTuple(
        config={"configurable": {"thread_id": "123", "checkpoint_id": str(uuid4())}},
        checkpoint={
            "channel_values": {"status": WorkflowStatusEnum.NOT_STARTED},
            "id": str(uuid4()),
            "channel_versions": {},
            "pending_sends": [],
            "versions_seen": {},
            "ts": "",
            "v": 0,
        },
        metadata={"step": 0},
        parent_config={"configurable": {"thread_id": "123", "checkpoint_id": None}},
    )


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry", autospec=True)
@patch("duo_workflow_service.workflows.issue_to_merge_request.workflow.Agent")
@patch("duo_workflow_service.workflows.issue_to_merge_request.workflow.HandoverAgent")
@patch("duo_workflow_service.workflows.issue_to_merge_request.workflow.ToolsExecutor")
@patch("duo_workflow_service.workflows.issue_to_merge_request.workflow.RunToolNode")
@patch(
    "duo_workflow_service.workflows.abstract_workflow.fetch_workflow_and_container_data"
)
@patch(
    "duo_workflow_service.workflows.issue_to_merge_request.workflow.create_chat_model"
)
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow", autospec=True)
@patch(
    "duo_workflow_service.workflows.issue_to_merge_request.workflow.ExecutorComponent",
    autospec=True,
)
@patch(
    "duo_workflow_service.workflows.issue_to_merge_request.workflow.PlannerComponent",
    autospec=True,
)
@patch(
    "duo_workflow_service.workflows.issue_to_merge_request.workflow.ToolsApprovalComponent",
    autospec=True,
)
@patch("duo_workflow_service.workflows.abstract_workflow.UserInterface", autospec=True)
@patch.dict(os.environ, {"DW_INTERNAL_EVENT__ENABLED": "true"})
async def test_workflow_run(
    mock_checkpoint_notifier,
    mock_tools_approval_component,
    mock_planner_component,
    mock_executor_component,
    mock_gitlab_workflow,
    mock_chat_client,
    mock_fetch_workflow_and_container_data,
    mock_run_tool_node_generic_class,
    mock_tools_executor,
    mock_handover_agent,
    mock_agent,
    mock_tools_registry_cls,
    checkpoint_tuple,
    graph_input,
):
    mock_user_interface_instance = mock_checkpoint_notifier.return_value
    mock_tools_registry = MagicMock(spec=ToolsRegistry)
    mock_tools_registry_cls.configure = AsyncMock(return_value=mock_tools_registry)
    mock_tools_registry.approval_required.return_value = False
    mock_fetch_workflow_and_container_data.return_value = (
        {
            "id": 1,
            "name": "test-project",
            "description": "This is a test project",
            "http_url_to_repo": "https://example.com/project",
            "web_url": "https://example.com/project",
            "default_branch": "main",
        },
        None,
        {"id": 1, "project_id": 1},
    )

    mock_tools_approval = mock_tools_approval_component.return_value
    mock_git_lab_workflow_instance = mock_gitlab_workflow.return_value
    mock_git_lab_workflow_instance.__aenter__.return_value = MemorySaver()
    mock_git_lab_workflow_instance.__aexit__.return_value = None
    mock_git_lab_workflow_instance._offline_mode = True
    mock_git_lab_workflow_instance.aget_tuple = AsyncMock(return_value=None)
    mock_git_lab_workflow_instance.alist = AsyncMock(return_value=[])
    mock_git_lab_workflow_instance.aput = AsyncMock(
        return_value={
            "configurable": {"thread_id": "123", "checkpoint_id": "checkpoint1"}
        }
    )

    mock_handover_agent.return_value.run.return_value = {
        "plan": Plan(steps=[]),
        "status": WorkflowStatusEnum.COMPLETED,
        "conversation_history": {},
    }

    mock_tools_executor.return_value.run.return_value = {
        "plan": Plan(steps=[]),
        "status": WorkflowStatusEnum.NOT_STARTED,
        "conversation_history": {},
    }

    mock_agent.return_value.run.side_effect = [
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

    mock_planner_component.return_value.attach.return_value = "set_status_to_execution"
    mock_executor_component.return_value.attach.return_value = "git_actions"
    mock_run_tool_node_class = mock_run_tool_node_generic_class.__getitem__.return_value
    mock_run_tool_node_class.return_value.run.return_value = {
        "command_output": ["test string"],
        "state": graph_input,
        "ui_chat_log": [],
    }

    workflow = Workflow(
        "123",
        workflow_type=CategoryEnum.WORKFLOW_ISSUE_TO_MERGE_REQUEST,
        workflow_metadata={"git_branch": "test-branch"},
    )
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
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry", autospec=True)
@patch("duo_workflow_service.workflows.issue_to_merge_request.workflow.Agent")
@patch("duo_workflow_service.workflows.issue_to_merge_request.workflow.HandoverAgent")
@patch("duo_workflow_service.workflows.issue_to_merge_request.workflow.ToolsExecutor")
@patch("duo_workflow_service.workflows.issue_to_merge_request.workflow.RunToolNode")
@patch(
    "duo_workflow_service.workflows.abstract_workflow.fetch_workflow_and_container_data"
)
@patch(
    "duo_workflow_service.workflows.issue_to_merge_request.workflow.create_chat_model"
)
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow", autospec=True)
@patch(
    "duo_workflow_service.workflows.issue_to_merge_request.workflow.ExecutorComponent",
    autospec=True,
)
@patch(
    "duo_workflow_service.workflows.issue_to_merge_request.workflow.PlannerComponent",
    autospec=True,
)
@patch.dict(os.environ, {"DW_INTERNAL_EVENT__ENABLED": "true"})
async def test_workflow_run_when_exception(
    mock_planner_component,
    mock_executor_component,
    mock_gitlab_workflow,
    mock_chat_client,
    mock_fetch_workflow_and_container_data,
    mock_run_tool_node_generic_class,
    mock_tools_executor,
    mock_handover_agent,
    mock_agent,
    mock_tools_registry,
):
    mock_tools_registry.configure = AsyncMock(
        return_value=MagicMock(spec=ToolsRegistry)
    )
    mock_fetch_workflow_and_container_data.return_value = (
        {
            "id": 1,
            "name": "test-project",
            "description": "This is a test project",
            "http_url_to_repo": "https://example.com/project",
            "web_url": "https://example.com/project",
        },
        None,
        {"id": 1, "project_id": 1},
    )

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

        async def __anext__(self):
            raise asyncio.CancelledError()

    workflow = Workflow(
        "123",
        {},
        workflow_type=CategoryEnum.WORKFLOW_ISSUE_TO_MERGE_REQUEST,
    )
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
