import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.graph import END, StateGraph
from langgraph.utils.runnable import Runnable

from duo_workflow_service.agents import HumanApprovalCheckExecutor
from duo_workflow_service.components.human_approval.plan_approval import (
    PlanApprovalComponent,
)
from duo_workflow_service.entities.event import WorkflowEventType
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    Plan,
    Task,
    TaskStatus,
    ToolStatus,
    WorkflowState,
    WorkflowStatusEnum,
)


def set_up_graph(
    node_return_value, component: PlanApprovalComponent
) -> tuple[Runnable, AsyncMock, AsyncMock, AsyncMock]:
    graph = StateGraph(WorkflowState)
    graph.set_entry_point("first_node")
    mock_entry_node = AsyncMock(return_value=node_return_value)
    graph.add_node("first_node", mock_entry_node)

    mock_termination_node = AsyncMock(return_value=node_return_value)
    graph.add_node("termination", mock_termination_node)
    graph.add_edge("termination", END)

    mock_continuation_node = AsyncMock(return_value=node_return_value)
    graph.add_node("continuation", mock_continuation_node)
    graph.add_edge("continuation", END)
    entry_point = component.attach(
        graph=graph,
        exit_node="termination",
        back_node="first_node",
        next_node="continuation",
    )

    graph.add_edge("first_node", entry_point)
    return (
        graph.compile(),
        mock_entry_node,
        mock_continuation_node,
        mock_termination_node,
    )


class TestPlanApprovalComponent:
    @pytest.fixture
    def node_return_value(self):
        return {
            "status": WorkflowStatusEnum.PLANNING,
            "conversation_history": {},
            "last_human_input": None,
            "handover": [],
            "ui_chat_log": [],
            "plan": Plan(
                steps=[
                    Task(id="1", description="Test step", status=TaskStatus.NOT_STARTED)
                ]
            ),
        }

    @pytest.fixture
    def mock_check_executor(self, node_return_value):
        mock = MagicMock(spec=HumanApprovalCheckExecutor)
        mock.run.return_value = node_return_value
        return mock

    @pytest.fixture
    def component(self, graph_config):
        return PlanApprovalComponent(
            workflow_id=graph_config["configurable"]["thread_id"],
            approved_agent_name="test-agent",
        )

    @pytest.mark.asyncio
    async def test_attach_with_plan_approval(
        self,
        component: PlanApprovalComponent,
        graph_config,
        graph_input: WorkflowState,
        mock_check_executor,
        node_return_value,
    ):
        with patch(
            "duo_workflow_service.components.human_approval.component.HumanApprovalCheckExecutor",
            return_value=mock_check_executor,
        ), patch.dict(os.environ, {"WORKFLOW_INTERRUPT": "True"}):

            graph, mock_entry_node, mock_continuation_node, mock_termination_node = (
                set_up_graph(node_return_value, component)
            )

            mock_check_executor.run.return_value = {
                "last_human_input": {
                    "event_type": WorkflowEventType.RESUME,
                }
            }

            response = await graph.ainvoke(input=graph_input, config=graph_config)

            assert "ui_chat_log" in response
            assert len(response["ui_chat_log"]) == 1
            chat_log = response["ui_chat_log"][0]
            assert chat_log["correlation_id"] is None
            assert chat_log["message_type"] == MessageTypeEnum.REQUEST
            assert "Review the proposed plan" in chat_log["content"]
            assert "select Approve plan" in chat_log["content"]
            assert chat_log["timestamp"] is not None
            assert chat_log["status"] == ToolStatus.SUCCESS
            assert chat_log["tool_info"] is None

            mock_check_executor.run.assert_called_once()
            mock_entry_node.assert_called_once()
            mock_continuation_node.assert_called_once()
            mock_termination_node.assert_not_called()
