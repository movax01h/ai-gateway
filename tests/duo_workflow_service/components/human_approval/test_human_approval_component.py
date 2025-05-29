import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.graph import END, StateGraph
from langgraph.utils.runnable import Runnable

from duo_workflow_service.agents import HumanApprovalCheckExecutor
from duo_workflow_service.components.human_approval.component import (
    HumanApprovalComponent,
)
from duo_workflow_service.entities.event import WorkflowEventType
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    ToolStatus,
    WorkflowState,
    WorkflowStatusEnum,
)
from lib import Result, result


class HumanApprovalComponentTestProxy(HumanApprovalComponent):
    """A concrete test proxy class for HumanApprovalComponent testing."""

    _approval_req_workflow_state: WorkflowStatusEnum = (
        WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED
    )
    _node_prefix: str = "test_approval"

    def _build_approval_request(self, _state) -> Result[str, RuntimeError]:
        return result.Ok("Test approval message")


class HumanApprovalComponentReturnToTheAgentTestProxy(HumanApprovalComponentTestProxy):
    def _build_approval_request(self, state):
        return result.Error(RuntimeError("Error building approval request"))


def set_up_graph(
    node_return_values, component: HumanApprovalComponent
) -> tuple[Runnable, AsyncMock, AsyncMock, AsyncMock]:
    graph = StateGraph(WorkflowState)
    graph.set_entry_point("first_node")
    mock_entry_node = AsyncMock(side_effect=node_return_values)
    graph.add_node("first_node", mock_entry_node)

    mock_termination_node = AsyncMock(side_effect=node_return_values)
    graph.add_node("termination", mock_termination_node)
    graph.add_edge("termination", END)

    mock_continuation_node = AsyncMock(side_effect=node_return_values)
    graph.add_node("continuation", mock_continuation_node)
    graph.add_edge("continuation", END)
    entry_point = component.attach(
        graph=graph,
        exit_node="termination",
        back_node="first_node",
        next_node="continuation",
    )

    graph.add_conditional_edges(
        "first_node",
        lambda s: (
            "termination"
            if s["status"] == WorkflowStatusEnum.CANCELLED
            else entry_point
        ),
    )
    return (
        graph.compile(),
        mock_entry_node,
        mock_continuation_node,
        mock_termination_node,
    )


def node_return_value(conversation_history={}, status=WorkflowStatusEnum.PLANNING):
    return {
        "status": status,
        "conversation_history": conversation_history,
        "last_human_input": None,
        "handover": [],
        "ui_chat_log": [],
    }


class TestHumanApprovalComponent:
    @pytest.fixture
    def mock_check_executor(self):
        mock = MagicMock(spec=HumanApprovalCheckExecutor)
        return mock

    @pytest.fixture
    def component(self, graph_config):
        return HumanApprovalComponentTestProxy(
            workflow_id=graph_config["configurable"]["thread_id"],
            approved_agent_name="test-agent",
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "mock_env", [{"WORKFLOW_INTERRUPT": "false"}, {"USE_MEMSAVER": "true"}]
    )
    async def test_without_env_vars(
        self,
        mock_env,
        component: HumanApprovalComponent,
        graph_config,
        graph_input: WorkflowState,
        mock_check_executor,
    ):
        with patch.dict(os.environ, mock_env):
            graph = StateGraph(WorkflowState)
            graph.set_entry_point("first_node")
            mock_entry_node = AsyncMock(return_value=node_return_value())
            graph.add_node("first_node", mock_entry_node)

            mock_continuation_node = AsyncMock(return_value=node_return_value())
            graph.add_node("continuation", mock_continuation_node)
            graph.add_edge("continuation", END)
            entry_point = component.attach(
                graph=graph,
                exit_node="termination",
                back_node="first_node",
                next_node="continuation",
            )

            graph.add_edge("first_node", entry_point)

            await graph.compile().ainvoke(input=graph_input, config=graph_config)

            assert entry_point == "continuation"
            mock_check_executor.run.assert_not_called()
            mock_entry_node.assert_called_once()
            mock_continuation_node.assert_called_once()

    @pytest.mark.asyncio
    async def test_attach_with_human_approval(
        self,
        component: HumanApprovalComponent,
        graph_config,
        graph_input: WorkflowState,
        mock_check_executor,
    ):
        with patch(
            "duo_workflow_service.components.human_approval.component.HumanApprovalCheckExecutor",
            return_value=mock_check_executor,
        ), patch.dict(os.environ, {"WORKFLOW_INTERRUPT": "True"}):

            graph, mock_entry_node, mock_continuation_node, mock_termination_node = (
                set_up_graph([node_return_value()], component)
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
            assert chat_log["content"] == "Test approval message"
            assert chat_log["timestamp"] is not None
            assert chat_log["status"] == ToolStatus.SUCCESS
            assert chat_log["tool_info"] is None

            mock_check_executor.run.assert_called_once()
            mock_entry_node.assert_called_once()
            mock_continuation_node.assert_called_once()
            mock_termination_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_attach_with_human_stop(
        self,
        component: HumanApprovalComponent,
        graph_config,
        graph_input: WorkflowState,
        mock_check_executor,
    ):

        with patch(
            "duo_workflow_service.components.human_approval.component.HumanApprovalCheckExecutor",
            return_value=mock_check_executor,
        ), patch.dict(os.environ, {"WORKFLOW_INTERRUPT": "True"}):

            graph, mock_entry_node, mock_continuation_node, mock_termination_node = (
                set_up_graph([node_return_value()], component)
            )

            mock_check_executor.run.return_value = {
                "last_human_input": {
                    "event_type": WorkflowEventType.STOP,
                }
            }

            response = await graph.ainvoke(input=graph_input, config=graph_config)

            assert "ui_chat_log" in response
            assert len(response["ui_chat_log"]) == 1
            chat_log = response["ui_chat_log"][0]
            assert chat_log["correlation_id"] is None
            assert chat_log["message_type"] == MessageTypeEnum.REQUEST
            assert chat_log["content"] == "Test approval message"
            assert chat_log["timestamp"] is not None
            assert chat_log["status"] == ToolStatus.SUCCESS
            assert chat_log["tool_info"] is None

            mock_entry_node.assert_called_once()
            mock_check_executor.run.assert_called_once()
            mock_termination_node.assert_called_once()
            mock_continuation_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_attach_with_human_feedback(
        self,
        component: HumanApprovalComponent,
        graph_config,
        graph_input: WorkflowState,
        mock_check_executor,
    ):

        with patch(
            "duo_workflow_service.components.human_approval.component.HumanApprovalCheckExecutor",
            return_value=mock_check_executor,
        ), patch.dict(os.environ, {"WORKFLOW_INTERRUPT": "True"}):

            graph, mock_entry_node, mock_continuation_node, mock_termination_node = (
                set_up_graph([node_return_value(), node_return_value()], component)
            )

            mock_check_executor.run.side_effect = [
                {
                    "last_human_input": {
                        "event_type": WorkflowEventType.MESSAGE,
                        "message": "Please check these conditions before proceeding",
                    }
                },
                {
                    "last_human_input": {
                        "event_type": WorkflowEventType.RESUME,
                    }
                },
            ]

            # Run the graph
            response = await graph.ainvoke(input=graph_input, config=graph_config)

            assert "ui_chat_log" in response
            assert len(response["ui_chat_log"]) == 2  # Two entries for two iterations

            chat_log1 = response["ui_chat_log"][0]
            assert chat_log1["correlation_id"] is None
            assert chat_log1["message_type"] == MessageTypeEnum.REQUEST
            assert chat_log1["content"] == "Test approval message"
            assert chat_log1["timestamp"] is not None
            assert chat_log1["status"] == ToolStatus.SUCCESS
            assert chat_log1["tool_info"] is None

            chat_log2 = response["ui_chat_log"][1]
            assert chat_log2["correlation_id"] is None
            assert chat_log2["message_type"] == MessageTypeEnum.REQUEST
            assert chat_log2["content"] == "Test approval message"
            assert chat_log2["timestamp"] is not None
            assert chat_log2["status"] == ToolStatus.SUCCESS
            assert chat_log2["tool_info"] is None

            assert mock_entry_node.call_count == 2
            assert mock_check_executor.run.call_count == 2
            mock_continuation_node.assert_called_once()
            mock_termination_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_attach_with_return_to_the_agent(
        self,
        graph_config,
        graph_input: WorkflowState,
        mock_check_executor,
    ):

        with patch(
            "duo_workflow_service.components.human_approval.component.HumanApprovalCheckExecutor",
            return_value=mock_check_executor,
        ), patch.dict(os.environ, {"WORKFLOW_INTERRUPT": "True"}):
            component = HumanApprovalComponentReturnToTheAgentTestProxy(
                workflow_id=graph_config["configurable"]["thread_id"],
                approved_agent_name="test-agent",
            )

            graph, mock_entry_node, mock_continuation_node, mock_termination_node = (
                set_up_graph(
                    [
                        node_return_value(),
                        node_return_value(status=WorkflowStatusEnum.CANCELLED),
                    ],
                    component,
                )
            )
            # Run the graph
            response = await graph.ainvoke(input=graph_input, config=graph_config)

            assert "ui_chat_log" in response
            assert len(response["ui_chat_log"]) == 0

            assert mock_entry_node.call_count == 2
            mock_check_executor.run.assert_not_called()
            mock_continuation_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_attach_node_creation(
        self,
        component: HumanApprovalComponent,
        mock_check_executor,
    ):

        with patch(
            "duo_workflow_service.components.human_approval.component.HumanApprovalCheckExecutor",
            return_value=mock_check_executor,
        ) as mock_check_exec_cls, patch.dict(
            os.environ, {"WORKFLOW_INTERRUPT": "True"}
        ):

            set_up_graph(node_return_value(), component)

            mock_check_exec_cls.assert_called_once_with(
                agent_name="test-agent", workflow_id="test-workflow"
            )
