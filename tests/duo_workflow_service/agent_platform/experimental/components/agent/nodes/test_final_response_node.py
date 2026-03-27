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
        conversation_history_key,
        simple_output,
        flow_state_with_message,
        tool_call_id,
        final_response_content,
        ui_history,
        component_name,
    ):
        """Test successful run with output IOKey."""
        node = FinalResponseNode(
            name="test_node",
            conversation_history_key_factory=lambda _: conversation_history_key,
            output_key_factory=lambda _: simple_output,
            ui_history=ui_history,
            response_schema=AgentFinalOutput,
        )

        # Get existing history
        existing_history = flow_state_with_message[FlowStateKeys.CONVERSATION_HISTORY][
            component_name
        ]

        # Execute
        result = await node.run(flow_state_with_message)

        # Verify
        assert FlowStateKeys.CONVERSATION_HISTORY in result
        assert component_name in result[FlowStateKeys.CONVERSATION_HISTORY]

        # Verify component appends completion marker to existing conversation history
        result_messages = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert len(result_messages) == len(existing_history) + 1
        assert result_messages[:-1] == existing_history
        assert isinstance(result_messages[-1], ToolMessage)
        assert result_messages[-1].content == ""
        assert result_messages[-1].tool_call_id == tool_call_id

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
    async def test_run_success_with_nested_output(
        self,
        conversation_history_key,
        nested_output,
        ui_history,
        component_name,
    ):
        """Test successful run with nested output IOKey."""
        node = FinalResponseNode(
            name="test_node",
            conversation_history_key_factory=lambda _: conversation_history_key,
            output_key_factory=lambda _: nested_output,
            ui_history=ui_history,
            response_schema=AgentFinalOutput,
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
        self,
        conversation_history_key,
        simple_output,
        base_flow_state,
        ui_history,
        component_name,
    ):
        """Test run with multiple tool calls raises ValueError."""
        node = FinalResponseNode(
            name="test_node",
            conversation_history_key_factory=lambda _: conversation_history_key,
            output_key_factory=lambda _: simple_output,
            ui_history=ui_history,
        )

        # Create state with multiple tool calls
        state = base_flow_state.copy()
        state["conversation_history"] = {component_name: [Mock(spec=BaseMessage)]}

        # Execute and verify error
        with pytest.raises(ValueError) as exc_info:
            await node.run(state)

        error_message = str(exc_info.value)
        assert "is not of type AIMessage" in error_message
        assert component_name in error_message

    @pytest.mark.asyncio
    async def test_run_multiple_tool_calls_raises_error(
        self,
        conversation_history_key,
        simple_output,
        base_flow_state,
        mock_ai_message_with_multiple_tools,
        ui_history,
        component_name,
    ):
        """Test run with multiple tool calls raises ValueError."""
        node = FinalResponseNode(
            name="test_node",
            conversation_history_key_factory=lambda _: conversation_history_key,
            output_key_factory=lambda _: simple_output,
            ui_history=ui_history,
            response_schema=AgentFinalOutput,
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
        conversation_history_key,
        simple_output,
        base_flow_state,
        mock_ai_message_without_final_tool,
        ui_history,
        component_name,
    ):
        """Test run raises ValueError when no final response tool call is found."""
        node = FinalResponseNode(
            name="test_node",
            conversation_history_key_factory=lambda _: conversation_history_key,
            output_key_factory=lambda _: simple_output,
            ui_history=ui_history,
            response_schema=AgentFinalOutput,
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
    async def test_run_schema_mode_no_tool_calls_raises_error(
        self,
        conversation_history_key,
        simple_output,
        base_flow_state,
        ui_history,
        component_name,
    ):
        """Test run raises ValueError when schema is set but model returns no tool calls."""
        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = []
        mock_message.text = "Text-only response"

        node = FinalResponseNode(
            name="test_node",
            conversation_history_key_factory=lambda _: conversation_history_key,
            output_key_factory=lambda _: simple_output,
            ui_history=ui_history,
            response_schema=AgentFinalOutput,
        )

        state = base_flow_state.copy()
        state["conversation_history"] = {component_name: [mock_message]}

        with pytest.raises(ValueError) as exc_info:
            await node.run(state)

        error_message = str(exc_info.value)
        assert "Response schema requires a tool call" in error_message
        assert component_name in error_message

    @pytest.mark.asyncio
    async def test_run_empty_tool_calls_uses_text_path(
        self,
        conversation_history_key,
        simple_output,
        base_flow_state,
        ui_history,
        component_name,
    ):
        """Test run with empty tool calls uses text-only response path."""
        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = []
        mock_message.text = "Text-only response"

        node = FinalResponseNode(
            name="test_node",
            conversation_history_key_factory=lambda _: conversation_history_key,
            output_key_factory=lambda _: simple_output,
            ui_history=ui_history,
        )

        state = base_flow_state.copy()
        state["conversation_history"] = {component_name: [mock_message]}

        result = await node.run(state)

        # Verify text-only path: output set
        assert "context" in result
        assert result["context"]["result"] == "Text-only response"

        ui_history.log.success.assert_called_once_with(
            "Text-only response",
            event=UILogEventsAgent.ON_AGENT_FINAL_ANSWER,
        )

    @pytest.mark.asyncio
    async def test_run_no_conversation_history_raises_error(
        self,
        conversation_history_key,
        simple_output,
        flow_state_no_history,
        ui_history,
        component_name,
    ):
        """Test run raises ValueError when no conversation history exists for component."""
        node = FinalResponseNode(
            name="test_node",
            conversation_history_key_factory=lambda _: conversation_history_key,
            output_key_factory=lambda _: simple_output,
            ui_history=ui_history,
        )

        # Execute and verify error
        with pytest.raises(ValueError) as exc_info:
            await node.run(flow_state_no_history)

        error_message = str(exc_info.value)
        assert "No messages found" in error_message
        assert component_name in error_message

    @pytest.mark.asyncio
    async def test_run_empty_conversation_history_raises_error(
        self,
        conversation_history_key,
        simple_output,
        flow_state_empty_history,
        ui_history,
        component_name,
    ):
        """Test run raises ValueError when conversation history is empty for component."""
        node = FinalResponseNode(
            name="test_node",
            conversation_history_key_factory=lambda _: conversation_history_key,
            output_key_factory=lambda _: simple_output,
            ui_history=ui_history,
        )

        # Execute and verify error
        with pytest.raises(ValueError) as exc_info:
            await node.run(flow_state_empty_history)

        error_message = str(exc_info.value)
        assert "No messages found" in error_message
        assert component_name in error_message

    @pytest.mark.asyncio
    async def test_run_with_multiple_messages_uses_last(
        self,
        conversation_history_key,
        simple_output,
        base_flow_state,
        ui_history,
        component_name,
    ):
        """Test run uses the last message in conversation history."""
        node = FinalResponseNode(
            name="test_node",
            conversation_history_key_factory=lambda _: conversation_history_key,
            output_key_factory=lambda _: simple_output,
            ui_history=ui_history,
            response_schema=AgentFinalOutput,
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

        # Verify all prior messages preserved and completion marker appended
        result_messages = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert (
            len(result_messages) == 3
        ), "Expected 2 AI messages plus completion marker"
        assert result_messages[0] == first_message
        assert result_messages[1] == last_message
        assert isinstance(result_messages[2], ToolMessage)
        assert result_messages[2].tool_call_id == "last_tool_id"
        assert result["context"]["result"] == "Last response"
