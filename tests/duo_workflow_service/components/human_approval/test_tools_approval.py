"""Tests for ToolsApprovalComponent."""

import os
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
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
from duo_workflow_service.tools.toolset import MalformedToolCallError


def set_up_graph(
    node_return_value, component: ToolsApprovalComponent
) -> tuple[Runnable, AsyncMock, AsyncMock, AsyncMock]:
    graph = StateGraph(WorkflowState)
    graph.set_entry_point("first_node")
    mock_entry_node = AsyncMock(side_effect=node_return_value)
    graph.add_node("first_node", mock_entry_node)

    mock_termination_node = AsyncMock(side_effect=node_return_value)
    graph.add_node("termination", mock_termination_node)
    graph.add_edge("termination", END)

    mock_continuation_node = AsyncMock(side_effect=node_return_value)
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


def node_return_value(messages, status=WorkflowStatusEnum.PLANNING):
    return {
        "status": status,
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
    def mock_toolset(self, mock_tool):
        mock = MagicMock()
        mock.bindable = []
        mock.__getitem__.return_value = mock_tool
        mock.approved.return_value = True
        return mock

    @pytest.fixture
    def component(self, graph_config, mock_toolset):
        return ToolsApprovalComponent(
            workflow_id=graph_config["configurable"]["thread_id"],
            approved_agent_name="test-agent",
            toolset=mock_toolset,
        )

    @pytest.mark.asyncio
    async def test_tools_approval_with_multiple_tools(
        self,
        component: ToolsApprovalComponent,
        mock_toolset,
        mock_tool,
        graph_config,
        graph_input: WorkflowState,
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
            tool1_call = {
                "id": "1",
                "name": "tool1",
                "args": {"arg1": "value1"},
                "type": "tool_call",
            }
            tool2_call = {
                "id": "2",
                "name": "tool2",
                "args": {"arg2": "value2"},
                "type": "tool_call",
            }
            pre_approved_tool_call = {
                "id": "3",
                "name": "pre_approved_tool",
                "args": {"arg3": "value3"},
                "type": "tool_call",
            }

            mock_toolset.approved.side_effect = [False, False, True]

            node_resp = [
                node_return_value(
                    messages=[
                        AIMessage(
                            content="Testing tools",
                            tool_calls=[
                                tool1_call,
                                tool2_call,
                                pre_approved_tool_call,
                            ],
                        )
                    ]
                )
            ]

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

            assert mock_toolset.validate_tool_call.call_count == 3
            mock_toolset.validate_tool_call.assert_has_calls(
                [
                    call(tool1_call),
                    call(tool2_call),
                    call(pre_approved_tool_call),
                ]
            )

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
        mock_toolset,
        mock_check_executor,
    ):
        with patch(
            "duo_workflow_service.components.human_approval.component.HumanApprovalCheckExecutor",
            return_value=mock_check_executor,
        ), patch.dict(os.environ, {"WORKFLOW_INTERRUPT": "True"}):
            node_resp = [
                node_return_value(
                    messages=[
                        AIMessage(
                            content="Testing no tools",
                            tool_calls=[],
                        )
                    ]
                ),
                node_return_value(
                    status=WorkflowStatusEnum.CANCELLED,
                    messages=[
                        AIMessage(
                            content="Cancel the execution to allow graph to complete without further errors",
                            tool_calls=[],
                        )
                    ],
                ),
            ]

            graph, mock_entry_node, mock_continuation_node, _ = set_up_graph(
                node_resp, component
            )

            response = await graph.ainvoke(input=graph_input, config=graph_config)

            assert mock_entry_node.call_count == 2
            assert "ui_chat_log" in response
            assert len(response["ui_chat_log"]) == 0

            mock_toolset.__getitem__.assert_not_called()
            mock_check_executor.run.assert_not_called()
            mock_continuation_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_tools_approval_with_no_messages(
        self,
        component: ToolsApprovalComponent,
        graph_config,
        graph_input: WorkflowState,
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
            node_resp = [
                node_return_value(
                    messages=[
                        AIMessage(
                            content="Testing tools",
                            tool_calls=[
                                {"id": "1", "name": "tool1", "args": {"arg1": "value1"}}
                            ],
                        )
                    ]
                )
            ]

            graph, mock_entry_node, mock_continuation_node, mock_termination_node = (
                set_up_graph(node_resp, component)
            )

            with pytest.raises(
                RuntimeError, match="No valid tool calls were found to display."
            ):
                await graph.ainvoke(input=graph_input, config=graph_config)

                mock_check_executor.run.assert_not_called()
                mock_entry_node.assert_called_once()
                mock_continuation_node.assert_not_called()
                mock_termination_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_tools_approval_with_both_valid_and_invalid_tool_calls(
        self,
        component: ToolsApprovalComponent,
        graph_config,
        graph_input: WorkflowState,
        mock_tool,
        mock_toolset,
        mock_check_executor,
    ):
        with patch(
            "duo_workflow_service.components.human_approval.component.HumanApprovalCheckExecutor",
            return_value=mock_check_executor,
        ), patch(
            "duo_workflow_service.components.human_approval.tools_approval.format_tool_display_message",
            return_value="Using mock tool1: {'arg1': 'value1'}",
        ) as mock_format_tool_msg, patch.dict(
            os.environ, {"WORKFLOW_INTERRUPT": "True"}
        ):
            valid_tool_call = ToolCall(
                id="1",
                name="valid_tool",
                args={"arg1": "value1"},
            )
            pre_approved_tool_call = ToolCall(
                id="2",
                name="pre_approved_tool",
                args={"arg2": "value2"},
            )
            non_existent_tool_call = ToolCall(
                id="3",
                name="non_existent_tool",
                args={"arg3": "value3"},
            )

            # Set up the side effects:
            # 1. First call (valid_tool) - returns False (not pre-approved, needs approval)
            # 2. Second call (pre_approved_tool) - returns True (pre-approved, needs no approval)
            # 3. Third call (non_existent_tool) - raises MalformedToolCallError
            mock_toolset.validate_tool_call.side_effect = [
                valid_tool_call,
                pre_approved_tool_call,
                MalformedToolCallError(
                    f"Tool '{non_existent_tool_call['name']}' does not exist",
                    tool_call=non_existent_tool_call,
                ),
            ]

            node_resp = [
                node_return_value(
                    messages=[
                        HumanMessage(
                            content="Testing mixed tools",
                        ),
                        AIMessage(
                            content="Those are mixed tools",
                            tool_calls=[
                                valid_tool_call,
                                pre_approved_tool_call,
                                non_existent_tool_call,
                            ],
                        ),
                    ]
                ),
                node_return_value(
                    status=WorkflowStatusEnum.CANCELLED,
                    messages=[
                        AIMessage(
                            content="Cancel the execution to allow graph to complete without further errors",
                            tool_calls=[],
                        )
                    ],
                ),
            ]

            graph, mock_entry_node, mock_continuation_node, _ = set_up_graph(
                node_resp, component
            )

            response = await graph.ainvoke(input=graph_input, config=graph_config)

            assert "ui_chat_log" in response
            assert len(response["ui_chat_log"]) == 0

            # Verify that graph execution returned back to the Agent
            assert mock_entry_node.call_count == 2

            # Verify that the 2nd call to mock_entry_node includes conversation_history
            second_call_args = mock_entry_node.call_args_list[1]
            second_call_state = second_call_args[0][
                0
            ]  # First positional argument (state)
            assert "conversation_history" in second_call_state

            conversation_history = second_call_state["conversation_history"]
            assert "test-agent" in conversation_history
            messages = conversation_history["test-agent"]

            # Should have 5 messages total: 1 Human message, 1 AI message,
            # 2 for valid tools (rejected) + 1 for invalid tool (error)
            assert len(messages) == 5

            # First two messages should be ToolMessages for valid tools with rejection content
            first_tool_msg = messages[2]
            assert isinstance(first_tool_msg, ToolMessage)
            assert first_tool_msg.tool_call_id == "1"  # valid_tool
            assert (
                first_tool_msg.content
                == "Tool call has been rejected due to other tool call in the last AIMessage being malformed"
            )

            second_tool_msg = messages[3]
            assert isinstance(second_tool_msg, ToolMessage)
            assert second_tool_msg.tool_call_id == "2"  # pre_approved_tool
            assert (
                second_tool_msg.content
                == "Tool call has been rejected due to other tool call in the last AIMessage being malformed"
            )

            # Third message should be ToolMessage for invalid tool with error content
            third_tool_msg = messages[4]
            assert isinstance(third_tool_msg, ToolMessage)
            assert third_tool_msg.tool_call_id == "3"  # non_existent_tool
            assert "Tool 'non_existent_tool' does not exist" in third_tool_msg.content

            # Verify the execution terminated without calling check executor or other nodes
            mock_format_tool_msg.assert_not_called()
            mock_check_executor.run.assert_not_called()
            mock_continuation_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_tools_approval_with_mallformed_tool_call_arguments(
        self,
        component: ToolsApprovalComponent,
        graph_config,
        graph_input: WorkflowState,
        mock_toolset,
        mock_check_executor,
    ):
        with patch(
            "duo_workflow_service.components.human_approval.component.HumanApprovalCheckExecutor",
            return_value=mock_check_executor,
        ), patch(
            "duo_workflow_service.components.human_approval.tools_approval.format_tool_display_message",
            return_value="Using mock tool1: {'arg1': 'value1'}",
        ) as mock_format_tool_msg, patch.dict(
            os.environ, {"WORKFLOW_INTERRUPT": "True"}
        ):
            invalid_tool_call = ToolCall(
                id="1",
                name="valid_tool",
                args={"list_arg1": "[value1]"},
            )

            mock_toolset.validate_tool_call.side_effect = [
                MalformedToolCallError(
                    "Tool call is invalid due to invalid arguments",
                    tool_call=invalid_tool_call,
                )
            ]

            node_resp = [
                node_return_value(
                    messages=[
                        AIMessage(
                            content="Testing invalid tools",
                            tool_calls=[
                                invalid_tool_call,
                            ],
                        )
                    ]
                ),
                node_return_value(
                    status=WorkflowStatusEnum.CANCELLED,
                    messages=[
                        AIMessage(
                            content="Cancel the execution to allow graph to complete without further errors",
                            tool_calls=[],
                        )
                    ],
                ),
            ]

            graph, mock_entry_node, mock_continuation_node, _ = set_up_graph(
                node_resp, component
            )

            response = await graph.ainvoke(input=graph_input, config=graph_config)

            assert "ui_chat_log" in response
            assert len(response["ui_chat_log"]) == 0

            # Verify that graph execution returned back to the Agent
            assert mock_entry_node.call_count == 2
            mock_toolset.validate_tool_call.assert_called_once()

            # Verify the execution terminated without calling check executor or other nodes
            mock_format_tool_msg.assert_not_called()
            mock_check_executor.run.assert_not_called()
            mock_continuation_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_tools_approval_with_no_op_tool(
        self,
        component: ToolsApprovalComponent,
        mock_toolset,
        mock_tool,
        graph_config,
        graph_input: WorkflowState,
        mock_check_executor,
    ):

        with patch(
            "duo_workflow_service.components.human_approval.component.HumanApprovalCheckExecutor",
            return_value=mock_check_executor,
        ), patch(
            "duo_workflow_service.components.human_approval.tools_approval.format_tool_display_message",
            side_effect=[
                "Using mock tool1: {'arg1': 'value1'}",
            ],
        ) as mock_format_tool_msg, patch.dict(
            os.environ, {"WORKFLOW_INTERRUPT": "True"}
        ):
            tool1_call = {"id": "1", "name": "tool1", "args": {"arg1": "value1"}}
            no_op_tool_call = {
                "id": "3",
                "name": "no_op_tool",
                "args": {"arg3": "value3"},
            }

            mock_toolset.approved.side_effect = [False, True]

            node_resp = [
                node_return_value(
                    messages=[
                        AIMessage(
                            content="Testing tools",
                            tool_calls=[
                                tool1_call,
                                no_op_tool_call,
                            ],
                        )
                    ]
                )
            ]

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
            assert "In order to complete the current task" in chat_log["content"]
            assert chat_log["timestamp"] is not None
            assert chat_log["status"] == ToolStatus.SUCCESS
            assert chat_log["tool_info"] is None

            mock_format_tool_msg.assert_has_calls(
                [
                    call(mock_tool, tool1_call["args"]),
                ]
            )
            mock_check_executor.run.assert_called_once()
            mock_entry_node.assert_called_once()
            mock_continuation_node.assert_called_once()
            mock_termination_node.assert_not_called()
