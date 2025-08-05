import asyncio
import os
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from duo_workflow_service.entities import Plan, WorkflowState, WorkflowStatusEnum
from duo_workflow_service.workflows.fix_pipeline.workflow import Workflow
from lib.internal_events.event_enum import CategoryEnum


@pytest.fixture
def mock_tools_approval_component():
    with patch(
        "duo_workflow_service.workflows.fix_pipeline.workflow.ToolsApprovalComponent",
        autospec=True,
    ) as mock:
        yield mock


@pytest.fixture
def mock_handover_agent():
    with patch(
        "duo_workflow_service.workflows.fix_pipeline.workflow.HandoverAgent"
    ) as mock:
        mock.return_value.run.return_value = {
            "plan": Plan(steps=[]),
            "status": WorkflowStatusEnum.COMPLETED,
            "conversation_history": {},
        }
        yield mock


@pytest.fixture
def mock_planner_component():
    with patch(
        "duo_workflow_service.workflows.fix_pipeline.workflow.PlannerComponent",
        autospec=True,
    ) as mock:
        mock.return_value.attach.return_value = "set_status_to_execution"
        yield mock


@pytest.fixture
def mock_executor_component():
    with patch(
        "duo_workflow_service.workflows.fix_pipeline.workflow.ExecutorComponent",
        autospec=True,
    ) as mock:
        mock.return_value.attach.return_value = "git_actions"
        yield mock


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.fix_pipeline.workflow.Agent")
@patch("duo_workflow_service.workflows.fix_pipeline.workflow.ToolsExecutor")
@patch("duo_workflow_service.workflows.fix_pipeline.workflow.RunToolNode")
@patch("duo_workflow_service.workflows.fix_pipeline.workflow.create_chat_model")
@patch("duo_workflow_service.workflows.abstract_workflow.UserInterface", autospec=True)
@patch.dict(os.environ, {"DW_INTERNAL_EVENT__ENABLED": "true"})
async def test_workflow_run(
    mock_checkpoint_notifier,
    mock_chat_client,
    mock_run_tool_node_generic_class,
    mock_tools_executor,
    mock_agent,
    mock_tools_registry_cls,
    mock_tools_registry,
    mock_tools_approval_component,
    mock_handover_agent,
    mock_planner_component,
    mock_executor_component,
    mock_fetch_workflow_and_container_data,
    mock_git_lab_workflow_instance,
    checkpoint_tuple,
):
    mock_user_interface_instance = mock_checkpoint_notifier.return_value
    mock_git_lab_workflow_instance.__aenter__.return_value = MemorySaver()

    mock_tools_approval_component.return_value.attach.side_effect = [
        "build_context_tools",
        "execution_tools",
    ]

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

    mock_run_tool_node_class = mock_run_tool_node_generic_class.__getitem__.return_value
    mock_run_tool_node_class.return_value.run.return_value = {
        "command_output": ["test string"],
        "state": WorkflowState(
            status=WorkflowStatusEnum.NOT_STARTED,
            conversation_history={},
            last_human_input=None,
            handover=[],
            ui_chat_log=[],
            plan=Plan(steps=[]),
            project=None,
            goal=None,
            additional_context=None,
        ),
        "ui_chat_log": [],
    }

    workflow = Workflow(
        "123",
        workflow_type=CategoryEnum.WORKFLOW_FIX_PIPELINE,
        workflow_metadata={"git_branch": "test-branch"},
    )

    await workflow.run("https://example.com/project/-/jobs/1")
    assert mock_agent.call_count == 1
    assert mock_agent.return_value.run.call_count == 2

    assert mock_tools_executor.call_count == 1
    assert mock_tools_executor.return_value.run.call_count >= 1

    assert mock_handover_agent.call_count == 3
    assert mock_handover_agent.return_value.run.call_count == 3

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
@patch("duo_workflow_service.workflows.fix_pipeline.workflow.Agent")
@patch("duo_workflow_service.workflows.fix_pipeline.workflow.ToolsExecutor")
@patch("duo_workflow_service.workflows.fix_pipeline.workflow.RunToolNode")
@patch("duo_workflow_service.workflows.fix_pipeline.workflow.create_chat_model")
@patch.dict(os.environ, {"DW_INTERNAL_EVENT__ENABLED": "true"})
async def test_workflow_run_when_exception(
    mock_chat_client,
    mock_run_tool_node_generic_class,
    mock_tools_executor,
    mock_agent,
    mock_executor_component,
    mock_planner_component,
    mock_handover_agent,
    mock_tools_registry_cls,
    mock_fetch_workflow_and_container_data,
    mock_git_lab_workflow_instance,
):
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
        workflow_type=CategoryEnum.WORKFLOW_FIX_PIPELINE,
    )
    with patch(
        "duo_workflow_service.workflows.fix_pipeline.workflow.StateGraph"
    ) as graph:
        compiled_graph = MagicMock()
        compiled_graph.aget_state = AsyncMock(return_value=None)
        compiled_graph.astream.return_value = AsyncIterator()
        instance = graph.return_value
        instance.compile.return_value = compiled_graph
        await workflow.run("https://example.com/project/-/jobs/1")

    assert workflow.is_done
