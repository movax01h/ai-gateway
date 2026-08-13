"""Test suite for v1 ToolApprovalFetchNode class."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from duo_workflow_service.agent_platform.utils.tool_event_tracker import (
    ToolEventTracker,
)
from duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_approval_fetch_node import (
    ToolApprovalFetchNode,
)
from duo_workflow_service.agent_platform.v1.state import FlowStateKeys
from duo_workflow_service.agent_platform.v1.state.base import (
    FlowEvent,
    FlowEventType,
    IOKey,
    RuntimeIOKey,
)
from duo_workflow_service.entities import WorkflowStatusEnum
from lib.internal_events import InternalEventAdditionalProperties
from lib.internal_events.event_enum import EventEnum


@pytest.fixture(name="conversation_history_key")
def conversation_history_key_fixture(component_name):
    """Fixture for conversation history key."""
    return IOKey(target="conversation_history", subkeys=[component_name])


@pytest.fixture(name="status_key")
def status_key_fixture():
    """Fixture for status key."""
    return IOKey(target="status")


@pytest.fixture(name="tool_approval_fetch_node")
def tool_approval_fetch_node_fixture(conversation_history_key, status_key):
    """Fixture for ToolApprovalFetchNode instance."""

    return ToolApprovalFetchNode(
        name="test_agent#tool_approval_fetch",
        conversation_history_key=RuntimeIOKey(
            alias="conversation_history", factory=lambda _: conversation_history_key
        ),
        status_key=RuntimeIOKey(alias="status", factory=lambda _: status_key),
        approval_decision_key=RuntimeIOKey(
            alias="tool_approval_decision",
            factory=lambda _: IOKey(
                target="context", subkeys=["test_agent", "tool_approval_decision"]
            ),
        ),
    )


@pytest.fixture(name="mock_ai_message_with_tool_calls")
def mock_ai_message_with_tool_calls_fixture():
    """Fixture for AIMessage with tool calls."""
    mock_message = Mock(spec=AIMessage)
    mock_message.tool_calls = [
        {"id": "call_123", "name": "test_tool", "args": {"param": "value"}},
        {"id": "call_456", "name": "another_tool", "args": {"foo": "bar"}},
    ]
    return mock_message


class TestToolApprovalFetchNodeApprove:
    """Test suite for APPROVE event handling."""

    @pytest.mark.asyncio
    async def test_approve_returns_status_update_only(
        self,
        tool_approval_fetch_node,
        base_flow_state,
        component_name,
        mock_ai_message_with_tool_calls,
    ):
        """Test that APPROVE event returns only status update, no history changes."""
        # Setup state with conversation history
        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {
            component_name: [mock_ai_message_with_tool_calls]
        }

        # Mock the interrupt to return APPROVE event
        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_approval_fetch_node.interrupt"
        ) as mock_interrupt:
            mock_interrupt.return_value = FlowEvent(event_type=FlowEventType.APPROVE)

            result = await tool_approval_fetch_node.run(state)

            # Should return status update to EXECUTION
            assert "status" in result
            assert result["status"] == WorkflowStatusEnum.EXECUTION.value

            # Should NOT include conversation history updates
            assert "conversation_history" not in result


class TestToolApprovalFetchNodeReject:
    """Test suite for REJECT event handling."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("conversation_history_key")
    async def test_reject_adds_rejection_tool_messages(
        self,
        tool_approval_fetch_node,
        base_flow_state,
        component_name,
        mock_ai_message_with_tool_calls,
    ):
        """Test that REJECT event adds rejection ToolMessages to conversation history."""
        # Setup state with conversation history
        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {
            component_name: [mock_ai_message_with_tool_calls]
        }

        # Mock the interrupt to return REJECT event
        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_approval_fetch_node.interrupt"
        ) as mock_interrupt:
            mock_interrupt.return_value = FlowEvent(event_type=FlowEventType.REJECT)

            result = await tool_approval_fetch_node.run(state)

            # Should include conversation history with rejection messages
            assert "conversation_history" in result
            assert component_name in result["conversation_history"]

            new_messages = result["conversation_history"][component_name]
            # Should have original message + 2 rejection ToolMessages (one per tool call)
            assert len(new_messages) == 3
            assert isinstance(new_messages[0], AIMessage)
            assert isinstance(new_messages[1], ToolMessage)
            assert isinstance(new_messages[2], ToolMessage)

            # Verify rejection message content
            assert new_messages[1].tool_call_id == "call_123"
            assert "rejected by user" in new_messages[1].content

            # Should also update status to EXECUTION
            assert "status" in result
            assert result["status"] == WorkflowStatusEnum.EXECUTION.value

    @pytest.mark.asyncio
    async def test_reject_with_no_tool_calls_raises_error(
        self, tool_approval_fetch_node, base_flow_state, component_name
    ):
        """Test that REJECT event with no tool calls raises RuntimeError."""
        # Setup state with conversation history without tool calls
        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = []

        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {component_name: [mock_message]}

        # Mock the interrupt to return REJECT event
        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_approval_fetch_node.interrupt"
        ) as mock_interrupt:
            mock_interrupt.return_value = FlowEvent(event_type=FlowEventType.REJECT)

            with pytest.raises(RuntimeError, match="No tool calls found to reject"):
                await tool_approval_fetch_node.run(state)


class TestToolApprovalFetchNodeModify:
    """Test suite for MODIFY event handling."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("conversation_history_key")
    async def test_modify_adds_rejection_messages_and_user_feedback(
        self,
        tool_approval_fetch_node,
        base_flow_state,
        component_name,
        mock_ai_message_with_tool_calls,
    ):
        """Test that MODIFY event adds rejection ToolMessages + HumanMessage with feedback."""
        # Setup state with conversation history
        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {
            component_name: [mock_ai_message_with_tool_calls]
        }

        user_feedback = "Please use a different approach"

        # Mock the interrupt to return MODIFY event with message
        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_approval_fetch_node.interrupt"
        ) as mock_interrupt:
            mock_interrupt.return_value = FlowEvent(
                event_type=FlowEventType.MODIFY, message=user_feedback
            )

            result = await tool_approval_fetch_node.run(state)

            # Should include conversation history with rejection messages + user feedback
            assert "conversation_history" in result
            assert component_name in result["conversation_history"]

            new_messages = result["conversation_history"][component_name]
            # Should have original message + 2 rejection ToolMessages + 1 HumanMessage
            assert len(new_messages) == 4
            assert isinstance(new_messages[0], AIMessage)
            assert isinstance(new_messages[1], ToolMessage)
            assert isinstance(new_messages[2], ToolMessage)
            assert isinstance(new_messages[3], HumanMessage)

            # Verify rejection messages
            assert new_messages[1].tool_call_id == "call_123"
            assert "rejected by user" in new_messages[1].content
            assert new_messages[2].tool_call_id == "call_456"
            assert "rejected by user" in new_messages[2].content

            # Verify user feedback message
            assert new_messages[3].content == user_feedback

            # Should also update status to EXECUTION
            assert "status" in result
            assert result["status"] == WorkflowStatusEnum.EXECUTION.value

            # UI chat log for user feedback is emitted by the flow base via
            # Command(update=...) — not by this node — so no ui_chat_log in result
            assert "ui_chat_log" not in result

    @pytest.mark.asyncio
    async def test_modify_without_message_raises_error(
        self,
        tool_approval_fetch_node,
        base_flow_state,
        component_name,
        mock_ai_message_with_tool_calls,
    ):
        """Test that MODIFY event without message raises ValueError."""
        # Setup state with conversation history
        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {
            component_name: [mock_ai_message_with_tool_calls]
        }

        # Mock the interrupt to return MODIFY event without message
        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_approval_fetch_node.interrupt"
        ) as mock_interrupt:
            mock_interrupt.return_value = FlowEvent(event_type=FlowEventType.MODIFY)

            with pytest.raises(
                ValueError,
                match="MODIFY event must include a message with user feedback",
            ):
                await tool_approval_fetch_node.run(state)

    @pytest.mark.asyncio
    async def test_modify_with_empty_message_raises_error(
        self,
        tool_approval_fetch_node,
        base_flow_state,
        component_name,
        mock_ai_message_with_tool_calls,
    ):
        """Test that MODIFY event with empty message raises ValueError."""
        # Setup state with conversation history
        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {
            component_name: [mock_ai_message_with_tool_calls]
        }

        # Mock the interrupt to return MODIFY event with empty message
        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_approval_fetch_node.interrupt"
        ) as mock_interrupt:
            mock_interrupt.return_value = FlowEvent(
                event_type=FlowEventType.MODIFY, message=""
            )

            with pytest.raises(
                ValueError,
                match="MODIFY event must include a message with user feedback",
            ):
                await tool_approval_fetch_node.run(state)

    @pytest.mark.asyncio
    async def test_modify_with_no_tool_calls_raises_error(
        self, tool_approval_fetch_node, base_flow_state, component_name
    ):
        """Test that MODIFY event with no tool calls raises RuntimeError."""
        # Setup state with conversation history without tool calls
        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = []

        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {component_name: [mock_message]}

        # Mock the interrupt to return MODIFY event with message
        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_approval_fetch_node.interrupt"
        ) as mock_interrupt:
            mock_interrupt.return_value = FlowEvent(
                event_type=FlowEventType.MODIFY, message="Try something else"
            )

            with pytest.raises(RuntimeError, match="No tool calls found to reject"):
                await tool_approval_fetch_node.run(state)


class TestToolApprovalFetchNodeUnknownEvent:
    """Test suite for unknown event type handling."""

    @pytest.mark.asyncio
    async def test_unknown_event_type_raises_error(
        self,
        tool_approval_fetch_node,
        base_flow_state,
        component_name,
        mock_ai_message_with_tool_calls,
    ):
        """Test that unknown event type raises ValueError."""
        # Setup state with conversation history
        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {
            component_name: [mock_ai_message_with_tool_calls]
        }

        # Mock the interrupt to return unknown event type
        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_approval_fetch_node.interrupt"
        ) as mock_interrupt:
            mock_interrupt.return_value = {"event_type": "unknown_type"}

            with pytest.raises(
                ValueError,
                match="Unexpected event type for tool approval: unknown_type. Expected APPROVE, REJECT, or MODIFY.",
            ):
                await tool_approval_fetch_node.run(state)


class TestToolApprovalFetchNodeTracking:
    """Test suite for approval-resolution internal event tracking."""

    @pytest.fixture(name="tracking_fetch_node")
    def tracking_fetch_node_fixture(
        self,
        conversation_history_key,
        status_key,
        approval_requests_key,
        flow_id,
        flow_type,
        mock_internal_event_client,
    ):
        """Fixture for ToolApprovalFetchNode with a real tracker and mock event client."""
        tracker = ToolEventTracker(
            flow_id=flow_id,
            flow_type=flow_type,
            internal_event_client=mock_internal_event_client,
        )
        return ToolApprovalFetchNode(
            name="test_agent#tool_approval_fetch",
            conversation_history_key=RuntimeIOKey(
                alias="conversation_history", factory=lambda _: conversation_history_key
            ),
            status_key=RuntimeIOKey(alias="status", factory=lambda _: status_key),
            approval_decision_key=RuntimeIOKey(
                alias="tool_approval_decision",
                factory=lambda _: IOKey(
                    target="context", subkeys=["test_agent", "tool_approval_decision"]
                ),
            ),
            tracker=tracker,
            approval_requests_key=approval_requests_key,
        )

    @pytest.fixture(name="state_with_tool_calls")
    def state_with_tool_calls_fixture(
        self, base_flow_state, component_name, mock_ai_message_with_tool_calls
    ):
        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {
            component_name: [mock_ai_message_with_tool_calls]
        }
        # Both tool calls were persisted by the request node as needing approval
        state[FlowStateKeys.CONTEXT] = {
            **state.get(FlowStateKeys.CONTEXT, {}),
            component_name: {
                "tool_approval_requests": [
                    {"id": "call_123", "name": "test_tool"},
                    {"id": "call_456", "name": "another_tool"},
                ]
            },
        }
        return state

    def _expected_properties(self, flow_type, flow_id, tool_name, outcome):
        return InternalEventAdditionalProperties(
            label=flow_type.value,
            property=outcome,
            value=flow_id,
            tool_name=tool_name,
        )

    @pytest.mark.asyncio
    async def test_approve_tracks_resolution_with_approval_outcome(
        self,
        tracking_fetch_node,
        state_with_tool_calls,
        mock_internal_event_client,
        flow_id,
        flow_type,
    ):
        """APPROVE tracks one resolution event per pending tool call with outcome approval."""
        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_approval_fetch_node.interrupt"
        ) as mock_interrupt:
            mock_interrupt.return_value = FlowEvent(event_type=FlowEventType.APPROVE)

            await tracking_fetch_node.run(state_with_tool_calls)

        assert mock_internal_event_client.track_event.call_count == 2
        for tool_name in ("test_tool", "another_tool"):
            mock_internal_event_client.track_event.assert_any_call(
                event_name=EventEnum.WORKFLOW_TOOL_APPROVAL_RESOLVED.value,
                additional_properties=self._expected_properties(
                    flow_type, flow_id, tool_name, "approval"
                ),
                category=flow_type.value,
            )

    @pytest.mark.asyncio
    async def test_reject_tracks_resolution_with_rejection_outcome(
        self,
        tracking_fetch_node,
        state_with_tool_calls,
        mock_internal_event_client,
        flow_id,
        flow_type,
    ):
        """REJECT tracks resolution events with outcome rejection."""
        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_approval_fetch_node.interrupt"
        ) as mock_interrupt:
            mock_interrupt.return_value = FlowEvent(event_type=FlowEventType.REJECT)

            await tracking_fetch_node.run(state_with_tool_calls)

        assert mock_internal_event_client.track_event.call_count == 2
        mock_internal_event_client.track_event.assert_any_call(
            event_name=EventEnum.WORKFLOW_TOOL_APPROVAL_RESOLVED.value,
            additional_properties=self._expected_properties(
                flow_type, flow_id, "test_tool", "rejection"
            ),
            category=flow_type.value,
        )

    @pytest.mark.asyncio
    async def test_modify_tracks_modification_and_omits_user_feedback(
        self,
        tracking_fetch_node,
        state_with_tool_calls,
        mock_internal_event_client,
        flow_id,
        flow_type,
    ):
        """MODIFY tracks outcome modification; user feedback text never appears in the payload."""
        feedback = "please use --dry-run instead of --force"
        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_approval_fetch_node.interrupt"
        ) as mock_interrupt:
            mock_interrupt.return_value = FlowEvent(
                event_type=FlowEventType.MODIFY, message=feedback
            )

            await tracking_fetch_node.run(state_with_tool_calls)

        assert mock_internal_event_client.track_event.call_count == 2
        mock_internal_event_client.track_event.assert_any_call(
            event_name=EventEnum.WORKFLOW_TOOL_APPROVAL_RESOLVED.value,
            additional_properties=self._expected_properties(
                flow_type, flow_id, "test_tool", "modification"
            ),
            category=flow_type.value,
        )
        # Privacy: the user feedback and tool args must never appear in payloads
        all_calls = str(mock_internal_event_client.track_event.call_args_list)
        assert feedback not in all_calls
        assert "--dry-run" not in all_calls
        assert "param" not in all_calls  # tool call args
        assert "'bar'" not in all_calls  # tool call args

    @pytest.mark.asyncio
    async def test_mixed_batch_tracks_resolution_only_for_requested_calls(
        self,
        tracking_fetch_node,
        state_with_tool_calls,
        component_name,
        mock_internal_event_client,
        flow_id,
        flow_type,
    ):
        """Only the persisted requested subset is resolved: a pre-approved call in the same batch (present in the
        AIMessage but never requested) tracks no resolve event."""
        state_with_tool_calls[FlowStateKeys.CONTEXT][component_name] = {
            "tool_approval_requests": [{"id": "call_456", "name": "another_tool"}]
        }

        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_approval_fetch_node.interrupt"
        ) as mock_interrupt:
            mock_interrupt.return_value = FlowEvent(event_type=FlowEventType.APPROVE)

            await tracking_fetch_node.run(state_with_tool_calls)

        mock_internal_event_client.track_event.assert_called_once_with(
            event_name=EventEnum.WORKFLOW_TOOL_APPROVAL_RESOLVED.value,
            additional_properties=self._expected_properties(
                flow_type, flow_id, "another_tool", "approval"
            ),
            category=flow_type.value,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "requests_context",
        [
            {},
            {"tool_approval_requests": []},
        ],
        ids=["missing", "empty"],
    )
    async def test_missing_or_empty_requested_calls_tracks_nothing(
        self,
        tracking_fetch_node,
        state_with_tool_calls,
        component_name,
        mock_internal_event_client,
        requests_context,
    ):
        """Missing or empty persisted approval-requests state tracks no events and does not crash."""
        state_with_tool_calls[FlowStateKeys.CONTEXT][component_name] = requests_context

        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_approval_fetch_node.interrupt"
        ) as mock_interrupt:
            mock_interrupt.return_value = FlowEvent(event_type=FlowEventType.APPROVE)

            result = await tracking_fetch_node.run(state_with_tool_calls)

        assert result["status"] == WorkflowStatusEnum.EXECUTION.value
        mock_internal_event_client.track_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_tracking_failure_does_not_break_approval_flow(
        self,
        tracking_fetch_node,
        state_with_tool_calls,
        mock_internal_event_client,
    ):
        """A failing event client is swallowed by the emitter; the approval flow completes normally."""
        mock_internal_event_client.track_event.side_effect = RuntimeError("boom")

        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_approval_fetch_node.interrupt"
        ) as mock_interrupt:
            mock_interrupt.return_value = FlowEvent(event_type=FlowEventType.APPROVE)

            result = await tracking_fetch_node.run(state_with_tool_calls)

        assert result["status"] == WorkflowStatusEnum.EXECUTION.value
        assert mock_internal_event_client.track_event.call_count == 2

    @pytest.mark.asyncio
    async def test_malformed_persisted_entries_are_skipped(
        self,
        tracking_fetch_node,
        state_with_tool_calls,
        component_name,
        mock_internal_event_client,
        flow_id,
        flow_type,
    ):
        """Malformed persisted entries are skipped; only the valid entry tracks a resolve event."""
        state_with_tool_calls[FlowStateKeys.CONTEXT][component_name] = {
            "tool_approval_requests": [
                {"id": "call_123"},  # dict without a name
                "call_456",  # not a dict at all
                {"id": "call_456", "name": "another_tool"},
            ]
        }

        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_approval_fetch_node.interrupt"
        ) as mock_interrupt:
            mock_interrupt.return_value = FlowEvent(event_type=FlowEventType.APPROVE)

            result = await tracking_fetch_node.run(state_with_tool_calls)

        assert result["status"] == WorkflowStatusEnum.EXECUTION.value
        mock_internal_event_client.track_event.assert_called_once_with(
            event_name=EventEnum.WORKFLOW_TOOL_APPROVAL_RESOLVED.value,
            additional_properties=self._expected_properties(
                flow_type, flow_id, "another_tool", "approval"
            ),
            category=flow_type.value,
        )
