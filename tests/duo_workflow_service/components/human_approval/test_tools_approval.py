"""Tests for ToolsApprovalComponent."""

import os
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from langgraph.utils.runnable import Runnable

from duo_workflow_service.agents import HumanApprovalCheckExecutor
from duo_workflow_service.components.human_approval.tools_approval import (
    ToolsApprovalComponent,
)
from duo_workflow_service.components.tools_registry import ToolsRegistry
from duo_workflow_service.entities.event import WorkflowEventType
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    Plan,
    ToolStatus,
    WorkflowState,
    WorkflowStatusEnum,
)
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool


def set_up_graph(
    node_return_value, component: ToolsApprovalComponent
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


def node_return_value(messages):
    return {
        "status": WorkflowStatusEnum.PLANNING,
        "conversation_history": {"test-agent": messages},
        "last_human_input": None,
        "handover": [],
        "ui_chat_log": [],
        "plan": Plan(steps=[]),
    }


class TestToolsApprovalComponent:
    @pytest.fixture
    def mock_check_executor(self):
        mock = MagicMock(spec=HumanApprovalCheckExecutor)
        return mock

    @pytest.fixture
    def mock_tool(self):
        mock = MagicMock(DuoBaseTool)
        mock.args_schema = None
        return mock

    @pytest.fixture
    def mock_tools_registry(self, mock_tool):
        mock = MagicMock(ToolsRegistry)
        mock.get_batch.return_value = [mock_tool]
        mock.get.return_value = mock_tool
        mock.approval_required.return_value = False
        return mock

    @pytest.fixture
    def component(self, graph_config, mock_tools_registry):
        return ToolsApprovalComponent(
            workflow_id=graph_config["configurable"]["thread_id"],
            approved_agent_name="test-agent",
            tools_registry=mock_tools_registry,
        )

    @pytest.mark.asyncio
    async def test_tools_approval_with_multiple_tools(
        self,
        component: ToolsApprovalComponent,
        graph_config,
        graph_input: WorkflowState,
        mock_tool,
        mock_tools_registry,
        mock_check_executor,
    ):

        with patch(
            "duo_workflow_service.components.human_approval.component.HumanApprovalCheckExecutor",
            return_value=mock_check_executor,
        ), patch(
            "duo_workflow_service.components.human_approval.tools_approval.format_tool_display_message",
            side_effect=[
                "Using mock tool1: {'arg1': 'value1'}",
                "Using mock tool2: {'arg2': 'value2'}",
            ],
        ) as mock_format_tool_msg, patch.dict(
            os.environ, {"WORKFLOW_INTERRUPT": "True"}
        ):
            tool1_call = {"id": "1", "name": "tool1", "args": {"arg1": "value1"}}
            tool2_call = {"id": "2", "name": "tool2", "args": {"arg2": "value2"}}
            pre_aprroved_tool_call = {
                "id": "3",
                "name": "pre_approved_tool",
                "args": {"arg3": "value3"},
            }
            mock_tools_registry.approval_required.side_effect = [True, True, False]

            node_resp = node_return_value(
                messages=[
                    AIMessage(
                        content="Testing tools",
                        tool_calls=[
                            tool1_call,
                            tool2_call,
                            pre_aprroved_tool_call,
                        ],
                    )
                ]
            )

            graph, mock_entry_node, mock_continuation_node, mock_termination_node = (
                set_up_graph(node_resp, component)
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
            assert "Using mock tool1: {'arg1': 'value1'}" in chat_log["content"]
            assert "Using mock tool2: {'arg2': 'value2'}" in chat_log["content"]
            assert "In order to complete the current task" in chat_log["content"]
            assert chat_log["timestamp"] is not None
            assert chat_log["status"] == ToolStatus.SUCCESS
            assert chat_log["tool_info"] is None

            mock_tools_registry.get.assert_has_calls([call("tool1"), call("tool2")])
            assert len(mock_format_tool_msg.mock_calls) == 2
            mock_format_tool_msg.assert_has_calls(
                [
                    call(mock_tool, tool1_call["args"]),
                    call(mock_tool, tool2_call["args"]),
                ]
            )
            mock_check_executor.run.assert_called_once()
            mock_entry_node.assert_called_once()
            mock_continuation_node.assert_called_once()
            mock_termination_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_tools_approval_with_no_tools(
        self,
        component: ToolsApprovalComponent,
        graph_config,
        graph_input: WorkflowState,
        mock_tools_registry,
        mock_check_executor,
    ):
        with patch(
            "duo_workflow_service.components.human_approval.component.HumanApprovalCheckExecutor",
            return_value=mock_check_executor,
        ), patch.dict(os.environ, {"WORKFLOW_INTERRUPT": "True"}):
            node_resp = node_return_value(
                messages=[
                    AIMessage(
                        content="Testing tools",
                        tool_calls=[],
                    )
                ]
            )

            graph, mock_entry_node, mock_continuation_node, mock_termination_node = (
                set_up_graph(node_resp, component)
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
            assert (
                "Found no tool call requests to approve. If this situation persists, please file a bug report"
                == chat_log["content"]
            )

            mock_tools_registry.get.assert_not_called()
            mock_check_executor.run.assert_called_once()
            mock_entry_node.assert_called_once()
            mock_continuation_node.assert_called_once()
            mock_termination_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_tools_approval_with_no_messages(
        self,
        component: ToolsApprovalComponent,
        graph_config,
        graph_input: WorkflowState,
        mock_tools_registry,
        mock_check_executor,
    ):
        with patch(
            "duo_workflow_service.components.human_approval.component.HumanApprovalCheckExecutor",
            return_value=mock_check_executor,
        ), patch(
            "duo_workflow_service.components.human_approval.tools_approval.format_tool_display_message",
            return_value=None,
        ) as mock_format_tool_msg, patch.dict(
            os.environ, {"WORKFLOW_INTERRUPT": "True"}
        ):
            node_resp = node_return_value(
                messages=[
                    AIMessage(
                        content="Testing tools",
                        tool_calls=[
                            {"id": "1", "name": "tool1", "args": {"arg1": "value1"}}
                        ],
                    )
                ]
            )

            graph, mock_entry_node, mock_continuation_node, mock_termination_node = (
                set_up_graph(node_resp, component)
            )

            mock_check_executor.run.return_value = {
                "last_human_input": {
                    "event_type": WorkflowEventType.RESUME,
                }
            }

            with pytest.raises(
                RuntimeError, match="No valid tool calls were found to display."
            ):
                await graph.ainvoke(input=graph_input, config=graph_config)

                mock_check_executor.run.assert_not_called()
                mock_entry_node.assert_called_once()
                mock_continuation_node.assert_not_called()
                mock_termination_node.assert_not_called()
