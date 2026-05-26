"""Test suite for v1 ToolApprovalRequestNode class."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_approval_request_node import (
    ToolApprovalRequestNode,
)
from duo_workflow_service.agent_platform.v1.state import FlowStateKeys
from duo_workflow_service.agent_platform.v1.state.base import (
    IOKey,
    RuntimeIOKey,
)
from duo_workflow_service.agent_platform.v1.ui_log import UIHistory
from duo_workflow_service.entities import (
    MessageTypeEnum,
    ToolStatus,
    WorkflowStatusEnum,
)
from duo_workflow_service.tools import MalformedToolCallError


@pytest.fixture(name="conversation_history_key")
def conversation_history_key_fixture(component_name):
    """Fixture for conversation history key."""
    return IOKey(target="conversation_history", subkeys=[component_name])


@pytest.fixture(name="status_key")
def status_key_fixture():
    """Fixture for status key."""
    return IOKey(target="status")


@pytest.fixture(name="mock_toolset")
def mock_toolset_fixture():
    """Fixture for mock toolset."""
    mock_toolset = Mock()

    # Mock tool
    mock_tool = Mock()
    mock_tool.name = "test_tool"

    # Setup toolset to return tool by name
    mock_toolset.__getitem__ = Mock(return_value=mock_tool)

    # validate_tool_call succeeds by default
    mock_toolset.validate_tool_call = Mock()

    # Tools are NOT pre-approved by default (require approval)
    mock_toolset.approved = Mock(return_value=False)

    return mock_toolset


@pytest.fixture(name="mock_ui_history")
def mock_ui_history_fixture():
    """Fixture for mock UI history."""
    return Mock(spec=UIHistory)


@pytest.fixture(name="tool_approval_request_node")
def tool_approval_request_node_fixture(
    conversation_history_key, status_key, mock_toolset, mock_ui_history
):
    """Fixture for ToolApprovalRequestNode instance."""
    return ToolApprovalRequestNode(
        name="test_agent#tool_approval_request",
        conversation_history_key=RuntimeIOKey(
            alias="conversation_history", factory=lambda _: conversation_history_key
        ),
        toolset=mock_toolset,
        pre_approved_tools=[],
        status_key=RuntimeIOKey(alias="status", factory=lambda _: status_key),
        ui_history=mock_ui_history,
    )


@pytest.fixture(name="mock_ai_message_with_tool_calls")
def mock_ai_message_with_tool_calls_fixture():
    """Fixture for AIMessage with tool calls."""
    mock_message = Mock(spec=AIMessage)
    mock_message.tool_calls = [
        {"id": "call_123", "name": "test_tool", "args": {"param": "value"}},
    ]
    return mock_message


class TestToolApprovalRequestNodeValidCalls:
    """Test suite for valid tool call handling."""

    @pytest.mark.asyncio
    async def test_valid_tool_calls_creates_ui_logs_and_sets_status(
        self,
        tool_approval_request_node,
        base_flow_state,
        component_name,
        mock_ai_message_with_tool_calls,
    ):
        """Test that valid tool calls create UI logs and set TOOL_CALL_APPROVAL_REQUIRED status."""
        # Setup state with conversation history
        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {
            component_name: [mock_ai_message_with_tool_calls]
        }

        # Mock format_tool_display_message to return display text
        with patch(
            "duo_workflow_service.agent_platform.v1.components."
            "agent.nodes.tool_approval_request_node.format_tool_display_message"
        ) as mock_format:
            mock_format.return_value = "Execute test_tool with param=value"

            result = await tool_approval_request_node.run(state)

            # Should include UI chat logs
            assert "ui_chat_log" in result
            assert len(result["ui_chat_log"]) == 1

            ui_log = result["ui_chat_log"][0]
            # UiChatLog is TypedDict, so it's a dict
            assert ui_log["message_type"] == MessageTypeEnum.REQUEST
            assert ui_log["content"] == "Execute test_tool with param=value"
            assert ui_log["status"] == ToolStatus.SUCCESS
            assert ui_log["tool_info"]["name"] == "test_tool"
            assert ui_log["message_id"] == "request-call_123"

            # Should set status to TOOL_CALL_APPROVAL_REQUIRED (nested dict)
            assert "status" in result
            assert (
                result["status"] == WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED.value
            )

    @pytest.mark.asyncio
    async def test_multiple_valid_tool_calls_creates_multiple_ui_logs(
        self,
        tool_approval_request_node,
        base_flow_state,
        component_name,
    ):
        """Test that multiple valid tool calls create multiple UI logs."""
        # Create message with multiple tool calls
        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = [
            {"id": "call_1", "name": "tool_a", "args": {"x": 1}},
            {"id": "call_2", "name": "tool_b", "args": {"y": 2}},
        ]

        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {component_name: [mock_message]}

        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent."
            "nodes.tool_approval_request_node.format_tool_display_message"
        ) as mock_format:
            mock_format.side_effect = ["Display A", "Display B"]

            result = await tool_approval_request_node.run(state)

            # Should have 2 UI logs
            assert len(result["ui_chat_log"]) == 2
            assert result["ui_chat_log"][0]["message_id"] == "request-call_1"
            assert result["ui_chat_log"][1]["message_id"] == "request-call_2"

    @pytest.mark.asyncio
    async def test_tool_with_none_display_message_skipped(
        self,
        tool_approval_request_node,
        base_flow_state,
        component_name,
    ):
        """Test that tools with None display message are skipped from UI logs."""
        # Create message with multiple tool calls
        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = [
            {"id": "call_1", "name": "tool_a", "args": {"x": 1}},
            {"id": "call_2", "name": "tool_b", "args": {"y": 2}},
        ]

        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {component_name: [mock_message]}

        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent."
            "nodes.tool_approval_request_node.format_tool_display_message"
        ) as mock_format:
            # First returns None (skip), second returns display text
            mock_format.side_effect = [None, "Display B"]

            result = await tool_approval_request_node.run(state)

            # Should only have 1 UI log (second one)
            assert len(result["ui_chat_log"]) == 1
            assert result["ui_chat_log"][0]["message_id"] == "request-call_2"


class TestToolApprovalRequestNodeInvalidCalls:
    """Test suite for invalid tool call handling."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("conversation_history_key")
    async def test_invalid_tool_calls_returns_error_messages(
        self,
        tool_approval_request_node,
        base_flow_state,
        component_name,
        mock_toolset,
    ):
        """Test that invalid tool calls return error ToolMessages."""
        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = [
            {"id": "call_123", "name": "bad_tool", "args": {"invalid": "args"}},
        ]

        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {component_name: [mock_message]}

        # Make validate_tool_call raise MalformedToolCallError
        error = MalformedToolCallError(
            "Invalid arguments",
            tool_call={"id": "call_123", "name": "bad_tool"},
        )
        mock_toolset.validate_tool_call.side_effect = error

        result = await tool_approval_request_node.run(state)

        # Should include conversation history with error messages
        assert "conversation_history" in result
        assert component_name in result["conversation_history"]

        new_messages = result["conversation_history"][component_name]
        # Should have original message + 1 error ToolMessage
        assert len(new_messages) == 2
        assert isinstance(new_messages[0], AIMessage)
        assert isinstance(new_messages[1], ToolMessage)

        # Verify error message
        assert new_messages[1].tool_call_id == "call_123"
        assert "Invalid arguments" in str(new_messages[1].content)

        # Should set status to EXECUTION (not approval required)
        assert "status" in result
        assert result["status"] == WorkflowStatusEnum.EXECUTION.value

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("conversation_history_key")
    async def test_mixed_valid_invalid_rejects_entire_batch(
        self,
        tool_approval_request_node,
        base_flow_state,
        component_name,
        mock_toolset,
    ):
        """Test that when any call is invalid, the entire batch is rejected.

        Every tool_call_id in the AIMessage must have a corresponding ToolMessage or Anthropic returns a 400 error.
        Valid calls are cancelled so the LLM can replan from scratch.
        """
        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = [
            {"id": "call_good", "name": "good_tool", "args": {}},
            {"id": "call_bad", "name": "bad_tool", "args": {}},
        ]

        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {component_name: [mock_message]}

        # First call valid, second call invalid
        def validate_side_effect(tool_call):
            if tool_call["id"] == "call_bad":
                raise MalformedToolCallError("Bad tool error", tool_call=tool_call)

        mock_toolset.validate_tool_call.side_effect = validate_side_effect

        result = await tool_approval_request_node.run(state)

        # Should return a ToolMessage for every call in the batch (original + 2 errors)
        new_messages = result["conversation_history"][component_name]
        assert len(new_messages) == 3
        assert isinstance(new_messages[1], ToolMessage)
        assert isinstance(new_messages[2], ToolMessage)

        # Valid call gets a cancellation message
        assert new_messages[1].tool_call_id == "call_good"
        assert "cancelled" in new_messages[1].content

        # Invalid call gets the actual error
        assert new_messages[2].tool_call_id == "call_bad"
        assert "Bad tool error" in new_messages[2].content


class TestToolApprovalRequestNodeNoToolCalls:
    """Test suite for no tool calls handling."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("conversation_history_key")
    async def test_no_tool_calls_returns_error_human_message(
        self,
        tool_approval_request_node,
        base_flow_state,
        component_name,
    ):
        """Test that AIMessage without tool calls returns error HumanMessage."""
        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = []

        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {component_name: [mock_message]}

        result = await tool_approval_request_node.run(state)

        # Should include conversation history with error message
        assert "conversation_history" in result
        assert component_name in result["conversation_history"]

        new_messages = result["conversation_history"][component_name]
        # Should have original message + 1 error HumanMessage
        assert len(new_messages) == 2
        assert isinstance(new_messages[1], HumanMessage)
        assert "No tool calls found" in new_messages[1].content

        # Should set status to EXECUTION
        assert result["status"] == WorkflowStatusEnum.EXECUTION.value

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("conversation_history_key")
    async def test_non_ai_message_returns_error(
        self,
        tool_approval_request_node,
        base_flow_state,
        component_name,
    ):
        """Test that non-AIMessage returns error HumanMessage."""
        mock_message = Mock(spec=HumanMessage)
        mock_message.content = "user message"

        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {component_name: [mock_message]}

        result = await tool_approval_request_node.run(state)

        # Should return error message
        new_messages = result["conversation_history"][component_name]
        assert len(new_messages) == 2
        assert isinstance(new_messages[1], HumanMessage)
        assert "No tool calls found" in new_messages[1].content


class TestToolApprovalRequestNodePreApproved:
    """Test suite for pre-approved tool handling."""

    @pytest.mark.asyncio
    async def test_all_pre_approved_skips_approval(
        self,
        conversation_history_key,
        status_key,
        mock_toolset,
        mock_ui_history,
        base_flow_state,
        component_name,
    ):
        """Test that when all tools are pre-approved, approval is skipped."""
        # Create node with approved tools
        node = ToolApprovalRequestNode(
            name="test_agent#tool_approval_request",
            conversation_history_key=RuntimeIOKey(
                alias="conversation_history", factory=lambda _: conversation_history_key
            ),
            toolset=mock_toolset,
            pre_approved_tools=["approved_tool"],
            status_key=RuntimeIOKey(alias="status", factory=lambda _: status_key),
            ui_history=mock_ui_history,
        )

        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = [
            {"id": "call_1", "name": "approved_tool", "args": {}},
        ]

        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {component_name: [mock_message]}

        # Mock _should_skip_approval to return True for approved_tool
        with patch.object(node, "_should_skip_approval", return_value=True):
            result = await node.run(state)

            # Should set status to EXECUTION for explicit routing
            assert "status" in result
            assert result["status"] == WorkflowStatusEnum.EXECUTION.value

            # Should NOT include ui_chat_log
            assert "ui_chat_log" not in result


class TestToolApprovalRequestNodeEdgeCases:
    """Test suite for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_history_raises_error(
        self, tool_approval_request_node, base_flow_state, component_name
    ):
        """Test that empty conversation history raises RuntimeError."""
        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {component_name: []}

        with pytest.raises(RuntimeError, match="No conversation history found"):
            await tool_approval_request_node.run(state)

    @pytest.mark.asyncio
    async def test_all_tools_return_none_display_raises_error(
        self,
        tool_approval_request_node,
        base_flow_state,
        component_name,
    ):
        """Test that when all tools return None for display message, raises RuntimeError."""
        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = [
            {"id": "call_1", "name": "tool_a", "args": {}},
        ]

        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {component_name: [mock_message]}

        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent."
            "nodes.tool_approval_request_node.format_tool_display_message"
        ) as mock_format:
            # All return None
            mock_format.return_value = None

            with pytest.raises(
                RuntimeError, match="No valid tool calls found to display for approval"
            ):
                await tool_approval_request_node.run(state)
