from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from duo_workflow_service.agent_platform.experimental.components.agent.nodes.agent_node import (
    AgentFinalOutput,
)
from duo_workflow_service.agent_platform.experimental.components.agent.nodes.final_response_node import (
    FinalResponseNode,
)
from duo_workflow_service.agent_platform.experimental.components.agent.ui_log import (
    UILogEventsAgent,
)
from duo_workflow_service.agent_platform.experimental.state import (
    FlowState,
    FlowStateKeys,
)
from duo_workflow_service.entities.state import WorkflowStatusEnum


class TestFinalResponseNode:
    """Test suite for FinalResponseNode class."""

    @pytest.mark.asyncio
    async def test_run_success_with_output(
        self,
        component_name,
        simple_output,
        flow_state_with_message,
        tool_call_id,
        final_response_content,
        ui_history,
    ):
        """Test successful run with output IOKey."""
        node = FinalResponseNode(
            component_name=component_name,
            name="test_node",
            output=simple_output,
            ui_history=ui_history,
        )

        # Execute
        result = await node.run(flow_state_with_message)

        # Verify
        assert FlowStateKeys.CONVERSATION_HISTORY in result
        assert component_name in result[FlowStateKeys.CONVERSATION_HISTORY]

        # Check that a ToolMessage was created
        tool_messages = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert len(tool_messages) == 1
        assert isinstance(tool_messages[0], ToolMessage)
        assert tool_messages[0].content == ""
        assert tool_messages[0].tool_call_id == tool_call_id

        # Check that output was set in context
        assert "context" in result
        assert result["context"]["result"] == final_response_content

        # Verify ui_history.log.success was called with the correct parameters
        ui_history.log.success.assert_called_once_with(
            final_response_content,
            event=UILogEventsAgent.ON_AGENT_FINAL_ANSWER,
        )

        # Verify ui_history.pop_state_updates was called
        ui_history.pop_state_updates.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_success_without_output(
        self, component_name, flow_state_with_message, tool_call_id, ui_history
    ):
        """Test successful run without output IOKey."""
        node = FinalResponseNode(
            component_name=component_name,
            name="test_node",
            output=None,
            ui_history=ui_history,
        )

        # Execute
        result = await node.run(flow_state_with_message)

        # Verify
        assert FlowStateKeys.CONVERSATION_HISTORY in result
        assert component_name in result[FlowStateKeys.CONVERSATION_HISTORY]

        # Check that a ToolMessage was created
        tool_messages = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert len(tool_messages) == 1
        assert isinstance(tool_messages[0], ToolMessage)
        assert tool_messages[0].content == ""
        assert tool_messages[0].tool_call_id == tool_call_id

        # Check that no output was set in context (since output is None)
        assert "context" not in result

    @pytest.mark.asyncio
    async def test_run_success_with_nested_output(
        self, component_name, nested_output, final_response_content, ui_history
    ):
        """Test successful run with nested output IOKey."""
        node = FinalResponseNode(
            component_name=component_name,
            name="test_node",
            output=nested_output,
            ui_history=ui_history,
        )

        # Create mock tool call with different content
        nested_response_content = "Nested task completed!"
        mock_tool_call = {
            "id": "test_tool_call_id",
            "name": AgentFinalOutput.tool_title,
            "args": {"final_response": nested_response_content},
        }

        # Create mock AI message with tool calls
        mock_ai_message = Mock(spec=AIMessage)
        mock_ai_message.tool_calls = [mock_tool_call]

        # Create state
        state: FlowState = {
            "status": WorkflowStatusEnum.EXECUTION,
            "conversation_history": {component_name: [mock_ai_message]},
            "ui_chat_log": [],
            "context": {},
        }

        # Execute
        result = await node.run(state)

        # Verify nested structure was created
        assert "context" in result
        assert "workflow" in result["context"]
        assert "final" in result["context"]["workflow"]
        assert "response" in result["context"]["workflow"]["final"]
        assert (
            result["context"]["workflow"]["final"]["response"]
            == nested_response_content
        )

    @pytest.mark.asyncio
    async def test_run_last_messages_is_not_ai_message_raises_error(
        self, component_name, simple_output, base_flow_state, ui_history
    ):
        """Test run with multiple tool calls raises ValueError."""
        node = FinalResponseNode(
            component_name=component_name,
            name="test_node",
            output=simple_output,
            ui_history=ui_history,
        )

        # Create state with multiple tool calls
        state = base_flow_state.copy()
        state["conversation_history"] = {component_name: [Mock(spec=BaseMessage)]}

        # Execute and verify error
        with pytest.raises(ValueError) as exc_info:
            await node.run(state)

        error_message = str(exc_info.value)
        assert (
            f"The last message of {component_name} is not of type AIMessage"
            == error_message
        )

    @pytest.mark.asyncio
    async def test_run_multiple_tool_calls_raises_error(
        self,
        component_name,
        simple_output,
        base_flow_state,
        mock_ai_message_with_multiple_tools,
        ui_history,
    ):
        """Test run with multiple tool calls raises ValueError."""
        node = FinalResponseNode(
            component_name=component_name,
            name="test_node",
            output=simple_output,
            ui_history=ui_history,
        )

        # Create state with multiple tool calls
        state = base_flow_state.copy()
        state["conversation_history"] = {
            component_name: [mock_ai_message_with_multiple_tools]
        }

        # Execute and verify error
        with pytest.raises(ValueError) as exc_info:
            await node.run(state)

        error_message = str(exc_info.value)
        assert "Too many tool calls found in the last message" in error_message
        assert component_name in error_message

    @pytest.mark.asyncio
    async def test_run_no_final_response_tool_call_raises_error(
        self,
        component_name,
        simple_output,
        base_flow_state,
        mock_ai_message_without_final_tool,
        ui_history,
    ):
        """Test run raises ValueError when no final response tool call is found."""
        node = FinalResponseNode(
            component_name=component_name,
            name="test_node",
            output=simple_output,
            ui_history=ui_history,
        )

        # Create state with wrong tool call
        state = base_flow_state.copy()
        state["conversation_history"] = {
            component_name: [mock_ai_message_without_final_tool]
        }

        # Execute and verify error
        with pytest.raises(ValueError) as exc_info:
            await node.run(state)

        error_message = str(exc_info.value)
        assert (
            "Final response tool call not found in the conversation history"
            in error_message
        )
        assert component_name in error_message

    @pytest.mark.asyncio
    async def test_run_empty_tool_calls_raises_error(
        self,
        component_name,
        simple_output,
        base_flow_state,
        mock_ai_message_empty_tools,
        ui_history,
    ):
        """Test run raises ValueError when tool_calls is empty."""
        node = FinalResponseNode(
            component_name=component_name,
            name="test_node",
            output=simple_output,
            ui_history=ui_history,
        )

        # Create state with empty tool calls
        state = base_flow_state.copy()
        state["conversation_history"] = {component_name: [mock_ai_message_empty_tools]}

        # Execute and verify error
        with pytest.raises(ValueError) as exc_info:
            await node.run(state)

        error_message = str(exc_info.value)
        assert "No tool calls found in the last message" in error_message
        assert component_name in error_message

    @pytest.mark.asyncio
    async def test_run_no_conversation_history_raises_error(
        self, component_name, simple_output, flow_state_no_history, ui_history
    ):
        """Test run raises ValueError when no conversation history exists for component."""
        node = FinalResponseNode(
            component_name=component_name,
            name="test_node",
            output=simple_output,
            ui_history=ui_history,
        )

        # Execute and verify error
        with pytest.raises(ValueError) as exc_info:
            await node.run(flow_state_no_history)

        error_message = str(exc_info.value)
        assert f"No messages found for {component_name}" == error_message

    @pytest.mark.asyncio
    async def test_run_empty_conversation_history_raises_error(
        self, component_name, simple_output, flow_state_empty_history, ui_history
    ):
        """Test run raises ValueError when conversation history is empty for component."""
        node = FinalResponseNode(
            component_name=component_name,
            name="test_node",
            output=simple_output,
            ui_history=ui_history,
        )

        # Execute and verify error
        with pytest.raises(ValueError) as exc_info:
            await node.run(flow_state_empty_history)

        error_message = str(exc_info.value)
        assert f"No messages found for {component_name}" == error_message

    @pytest.mark.asyncio
    async def test_run_with_multiple_messages_uses_last(
        self, component_name, simple_output, base_flow_state, ui_history
    ):
        """Test run uses the last message in conversation history."""
        node = FinalResponseNode(
            component_name=component_name,
            name="test_node",
            output=simple_output,
            ui_history=ui_history,
        )

        # Create first message (should be ignored)
        first_tool_call = {
            "id": "first_tool_id",
            "name": AgentFinalOutput.tool_title,
            "args": {"final_response": "First response"},
        }
        first_message = Mock(spec=AIMessage)
        first_message.tool_calls = [first_tool_call]

        # Create last message (should be used)
        last_tool_call = {
            "id": "last_tool_id",
            "name": AgentFinalOutput.tool_title,
            "args": {"final_response": "Last response"},
        }
        last_message = Mock(spec=AIMessage)
        last_message.tool_calls = [last_tool_call]

        # Create state with multiple messages
        state = base_flow_state.copy()
        state["conversation_history"] = {component_name: [first_message, last_message]}

        # Execute
        result = await node.run(state)

        # Verify last message was used
        tool_messages = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert tool_messages[0].tool_call_id == "last_tool_id"
        assert result["context"]["result"] == "Last response"
