from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

from duo_workflow_service.agent_platform.experimental.components.human_input.nodes.fetch_node import (
    FetchNode,
)
from duo_workflow_service.agent_platform.experimental.components.human_input.ui_log import (
    UILogEventsHumanInput,
    UserLogWriter,
)
from duo_workflow_service.agent_platform.experimental.state import FlowStateKeys, IOKey
from duo_workflow_service.agent_platform.experimental.state.base import FlowEventType
from duo_workflow_service.agent_platform.experimental.ui_log import UIHistory
from duo_workflow_service.entities.state import WorkflowStatusEnum


class TestFetchNode:
    """Test suite for FetchNode."""

    @pytest.fixture
    def fetch_node(self):
        """Create FetchNode instance for testing."""
        ui_history = UIHistory(
            events=[UILogEventsHumanInput.ON_USER_RESPONSE],
            writer_class=UserLogWriter,
        )
        return FetchNode(
            name="test_component#fetch",
            component_name="test_component",
            sends_response_to="target_agent",
            output=IOKey(target="context", subkeys=["test_component", "approval"]),
            ui_history=ui_history,
        )

    @pytest.fixture
    def sample_state(self):
        """Create sample FlowState for testing."""
        return {
            "status": WorkflowStatusEnum.INPUT_REQUIRED,
            "conversation_history": {},
            "ui_chat_log": [],
            "context": {},
        }

    @pytest.mark.asyncio
    async def test_interrupt_handling_response_event(self, fetch_node, sample_state):
        """Test successful interrupt handling with RESPONSE event."""
        mock_event = {
            "event_type": FlowEventType.RESPONSE,
            "message": "User input response",
        }

        with patch(
            "duo_workflow_service.agent_platform.experimental.components.human_input.nodes.fetch_node.interrupt",
            return_value=mock_event,
        ):
            result = await fetch_node.run(sample_state)

            # Verify status transition to EXECUTION
            assert result[FlowStateKeys.STATUS] == WorkflowStatusEnum.EXECUTION.value

            # Verify conversation history contains HumanMessage
            assert FlowStateKeys.CONVERSATION_HISTORY in result
            conversation = result[FlowStateKeys.CONVERSATION_HISTORY]
            assert "target_agent" in conversation
            assert len(conversation["target_agent"]) == 1

            message = conversation["target_agent"][0]
            assert isinstance(message, HumanMessage)
            assert message.content == "User input response"

            # Verify UI chat log is present and contains user response
            assert FlowStateKeys.UI_CHAT_LOG in result
            ui_logs = result[FlowStateKeys.UI_CHAT_LOG]
            assert len(ui_logs) == 1
            assert ui_logs[0]["content"] == "User input response"
            assert ui_logs[0]["message_type"] == "user"

    @pytest.mark.asyncio
    async def test_interrupt_handling_approve_event(self, fetch_node, sample_state):
        """Test that APPROVE event stores approval in context."""
        mock_event = {
            "event_type": FlowEventType.APPROVE,
        }

        with patch(
            "duo_workflow_service.agent_platform.experimental.components.human_input.nodes.fetch_node.interrupt",
            return_value=mock_event,
        ):
            result = await fetch_node.run(sample_state)

            # Verify status transition to EXECUTION
            assert result[FlowStateKeys.STATUS] == WorkflowStatusEnum.EXECUTION.value

            # Verify approval is stored in context
            assert "context" in result
            assert "test_component" in result["context"]
            assert result["context"]["test_component"]["approval"] == "approve"

            # Verify UI chat log is present but empty for approve events
            assert FlowStateKeys.UI_CHAT_LOG in result
            assert result[FlowStateKeys.UI_CHAT_LOG] == []

    @pytest.mark.asyncio
    async def test_interrupt_handling_reject_event(self, fetch_node, sample_state):
        """Test that REJECT event stores rejection in context and adds HumanMessage if message present."""
        mock_event = {
            "event_type": FlowEventType.REJECT,
            "message": "User rejected with reason",
        }

        with patch(
            "duo_workflow_service.agent_platform.experimental.components.human_input.nodes.fetch_node.interrupt",
            return_value=mock_event,
        ):
            result = await fetch_node.run(sample_state)

            # Verify status transition to EXECUTION
            assert result[FlowStateKeys.STATUS] == WorkflowStatusEnum.EXECUTION.value

            # Verify rejection is stored in context
            assert "context" in result
            assert "test_component" in result["context"]
            assert result["context"]["test_component"]["approval"] == "reject"

            # Verify HumanMessage is added to conversation history for REJECT with message
            assert FlowStateKeys.CONVERSATION_HISTORY in result
            conversation = result[FlowStateKeys.CONVERSATION_HISTORY]
            assert "target_agent" in conversation
            assert len(conversation["target_agent"]) == 1

            message = conversation["target_agent"][0]
            assert isinstance(message, HumanMessage)
            assert message.content == "User rejected with reason"

            # Verify UI chat log contains user response
            assert FlowStateKeys.UI_CHAT_LOG in result
            ui_logs = result[FlowStateKeys.UI_CHAT_LOG]
            assert len(ui_logs) == 1
            assert ui_logs[0]["content"] == "User rejected with reason"
            assert ui_logs[0]["message_type"] == "user"

    @pytest.mark.asyncio
    async def test_interrupt_handling_reject_event_without_message(
        self, fetch_node, sample_state
    ):
        """Test that REJECT event without message stores rejection in context but no HumanMessage."""
        mock_event = {
            "event_type": FlowEventType.REJECT,
        }

        with patch(
            "duo_workflow_service.agent_platform.experimental.components.human_input.nodes.fetch_node.interrupt",
            return_value=mock_event,
        ):
            result = await fetch_node.run(sample_state)

            # Verify status transition to EXECUTION
            assert result[FlowStateKeys.STATUS] == WorkflowStatusEnum.EXECUTION.value

            # Verify rejection is stored in context
            assert "context" in result
            assert "test_component" in result["context"]
            assert result["context"]["test_component"]["approval"] == "reject"

            # Verify no conversation history is added when no message
            assert FlowStateKeys.CONVERSATION_HISTORY not in result

            # Verify UI chat log is present but empty when no message
            assert FlowStateKeys.UI_CHAT_LOG in result
            assert result[FlowStateKeys.UI_CHAT_LOG] == []

    @pytest.mark.asyncio
    async def test_interrupt_handling_unknown_event(self, fetch_node, sample_state):
        """Test interrupt handling with unknown event type raises ValueError."""
        mock_event = {
            "event_type": "UNKNOWN_EVENT_TYPE",
        }

        with patch(
            "duo_workflow_service.agent_platform.experimental.components.human_input.nodes.fetch_node.interrupt",
            return_value=mock_event,
        ):
            with pytest.raises(
                ValueError, match="Unknown event type: UNKNOWN_EVENT_TYPE"
            ):
                await fetch_node.run(sample_state)

    @pytest.mark.asyncio
    async def test_interrupt_message_format(self, fetch_node, sample_state):
        """Test that interrupt is called with correct message format."""
        mock_event = {
            "event_type": FlowEventType.RESPONSE,
            "message": "User response",
        }

        with patch(
            "duo_workflow_service.agent_platform.experimental.components.human_input.nodes.fetch_node.interrupt",
            return_value=mock_event,
        ) as mock_interrupt:
            await fetch_node.run(sample_state)

            # Verify interrupt was called with expected message
            mock_interrupt.assert_called_once_with(
                "Workflow interrupted; waiting for user input."
            )

    @pytest.mark.asyncio
    async def test_all_flow_event_types_coverage(self, fetch_node, sample_state):
        """Test that FetchNode can handle all FlowEventType enum values dynamically.

        This test ensures that:
        1. All current FlowEventType values are handled by FetchNode
        2. Future additions to FlowEventType will be caught if not handled
        3. Future removals from FlowEventType will be caught if still referenced
        """
        # Get all FlowEventType values dynamically
        all_event_types = list(FlowEventType)

        # Ensure we have at least the expected types (sanity check)
        assert len(all_event_types) >= 3, "Expected at least RESPONSE, APPROVE, REJECT"

        # Test each event type
        for event_type in all_event_types:
            # Create mock event with optional message for comprehensive testing
            mock_event = {
                "event_type": event_type,
            }

            if event_type == FlowEventType.RESPONSE:
                mock_event["message"] = "User input response"
            elif event_type == FlowEventType.REJECT:
                mock_event["message"] = "User rejected with reason"

            with patch(
                "duo_workflow_service.agent_platform.experimental.components.human_input.nodes.fetch_node.interrupt",
                return_value=mock_event,
            ):
                # This should not raise an exception for any valid FlowEventType
                result = await fetch_node.run(sample_state.copy())

                assert FlowStateKeys.STATUS in result
                assert (
                    result[FlowStateKeys.STATUS] == WorkflowStatusEnum.EXECUTION.value
                )
                assert FlowStateKeys.UI_CHAT_LOG in result
