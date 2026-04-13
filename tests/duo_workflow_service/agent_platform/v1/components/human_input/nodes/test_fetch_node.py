from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

from duo_workflow_service.agent_platform.v1.components.human_input.nodes.fetch_node import (
    FetchNode,
)
from duo_workflow_service.agent_platform.v1.components.human_input.ui_log import (
    UILogEventsHumanInput,
    UserLogWriter,
)
from duo_workflow_service.agent_platform.v1.state import FlowStateKeys, IOKey
from duo_workflow_service.agent_platform.v1.state.base import FlowEventType
from duo_workflow_service.agent_platform.v1.ui_log import UIHistory
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
            output=IOKey(target="context", subkeys=["test_component", "approval"]),
            conversation_history_key=IOKey(
                target="conversation_history", subkeys=["target_agent"]
            ),
            ui_history=ui_history,
            status_key=IOKey(target="status"),
        )

    @pytest.fixture
    def sample_state(self):
        """Create sample FlowState for testing."""
        return {
            "status": WorkflowStatusEnum.INPUT_REQUIRED,
            "conversation_history": {"target_agent": []},
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
            "duo_workflow_service.agent_platform.v1.components.human_input.nodes.fetch_node.interrupt",
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
            "duo_workflow_service.agent_platform.v1.components.human_input.nodes.fetch_node.interrupt",
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
    async def test_interrupt_handling_reject_event_without_message(
        self, fetch_node, sample_state
    ):
        """Test that REJECT event sends instruction to agent and user-friendly message to UI."""
        mock_event = {
            "event_type": FlowEventType.REJECT,
        }

        with patch(
            "duo_workflow_service.agent_platform.v1.components.human_input.nodes.fetch_node.interrupt",
            return_value=mock_event,
        ):
            result = await fetch_node.run(sample_state)

            # Verify status transition to EXECUTION
            assert result[FlowStateKeys.STATUS] == WorkflowStatusEnum.EXECUTION.value

            # Verify rejection is stored in context
            assert "context" in result
            assert "test_component" in result["context"]
            assert result["context"]["test_component"]["approval"] == "reject"

            # Verify default rejection message is added to conversation history
            assert FlowStateKeys.CONVERSATION_HISTORY in result
            conversation = result[FlowStateKeys.CONVERSATION_HISTORY]
            assert "target_agent" in conversation
            assert len(conversation["target_agent"]) == 1

            message = conversation["target_agent"][0]
            assert isinstance(message, HumanMessage)
            assert (
                message.content
                == "User rejected this action. Do not proceed and stop any tool execution in progress."
            )

            # Verify UI chat log contains the user-friendly rejection message
            assert FlowStateKeys.UI_CHAT_LOG in result
            ui_logs = result[FlowStateKeys.UI_CHAT_LOG]
            assert len(ui_logs) == 1
            assert ui_logs[0]["content"] == "Action rejected."
            assert ui_logs[0]["message_type"] == "user"

    @pytest.mark.asyncio
    async def test_interrupt_handling_modify_event(self, fetch_node, sample_state):
        """Test that MODIFY event with message adds HumanMessage to conversation history."""
        mock_event = {
            "event_type": FlowEventType.MODIFY,
            "message": "User requested modification with feedback",
        }

        with patch(
            "duo_workflow_service.agent_platform.v1.components.human_input.nodes.fetch_node.interrupt",
            return_value=mock_event,
        ):
            result = await fetch_node.run(sample_state)

            # Verify status transition to EXECUTION
            assert result[FlowStateKeys.STATUS] == WorkflowStatusEnum.EXECUTION.value

            # Verify modify decision is stored in context for routing
            assert "context" in result
            assert "test_component" in result["context"]
            assert result["context"]["test_component"]["approval"] == "modify"

            # Verify HumanMessage is added to conversation history for MODIFY
            assert FlowStateKeys.CONVERSATION_HISTORY in result
            conversation = result[FlowStateKeys.CONVERSATION_HISTORY]
            assert "target_agent" in conversation
            assert len(conversation["target_agent"]) == 1

            message = conversation["target_agent"][0]
            assert isinstance(message, HumanMessage)
            assert message.content == "User requested modification with feedback"

            # Verify UI chat log contains user response
            assert FlowStateKeys.UI_CHAT_LOG in result
            ui_logs = result[FlowStateKeys.UI_CHAT_LOG]
            assert len(ui_logs) == 1
            assert ui_logs[0]["content"] == "User requested modification with feedback"
            assert ui_logs[0]["message_type"] == "user"

    @pytest.mark.asyncio
    async def test_interrupt_handling_modify_event_without_message(
        self, fetch_node, sample_state
    ):
        """Test that MODIFY event without message raises ValueError."""
        mock_event = {
            "event_type": FlowEventType.MODIFY,
        }

        with patch(
            "duo_workflow_service.agent_platform.v1.components.human_input.nodes.fetch_node.interrupt",
            return_value=mock_event,
        ):
            with pytest.raises(
                ValueError,
                match="MODIFY event must include a message with user feedback",
            ):
                await fetch_node.run(sample_state)

    @pytest.mark.asyncio
    async def test_interrupt_handling_unknown_event(self, fetch_node, sample_state):
        """Test interrupt handling with unknown event type raises ValueError."""
        mock_event = {
            "event_type": "UNKNOWN_EVENT_TYPE",
        }

        with patch(
            "duo_workflow_service.agent_platform.v1.components.human_input.nodes.fetch_node.interrupt",
            return_value=mock_event,
        ):
            with pytest.raises(
                ValueError, match="Unknown event type: UNKNOWN_EVENT_TYPE"
            ):
                await fetch_node.run(sample_state)
