from unittest.mock import patch

import pytest

from duo_workflow_service.agent_platform.v1.components.human_input.nodes.request_node import (
    RequestNode,
)
from duo_workflow_service.agent_platform.v1.components.human_input.ui_log import (
    AgentLogWriter,
    UILogEventsHumanInput,
)
from duo_workflow_service.agent_platform.v1.state import FlowStateKeys, IOKey
from duo_workflow_service.agent_platform.v1.ui_log import UIHistory
from duo_workflow_service.entities import MessageTypeEnum
from duo_workflow_service.entities.state import WorkflowStatusEnum


class TestRequestNode:
    """Test suite for RequestNode."""

    @pytest.fixture
    def request_node(self):
        """Create RequestNode instance for testing."""
        return RequestNode(
            name="test_component#request",
            component_name="test_component",
            message_template="Formatted prompt content with {{test_key}}",
            inputs=[IOKey(target="context", subkeys=["test_key"])],
            ui_history=UIHistory(
                events=[UILogEventsHumanInput.ON_USER_INPUT_PROMPT],
                writer_class=AgentLogWriter,
            ),
            status_key=IOKey(target="status"),
        )

    @pytest.fixture
    def sample_state(self):
        """Create sample FlowState for testing."""
        return {
            "status": WorkflowStatusEnum.EXECUTION,
            "conversation_history": {},
            "ui_chat_log": [],
            "context": {"test_key": "test_value"},
        }

    @pytest.mark.asyncio
    async def test_status_transition_to_input_required(
        self, request_node, sample_state
    ):
        """Test that request node transitions status to INPUT_REQUIRED."""
        result = await request_node.run(sample_state)

        assert FlowStateKeys.STATUS in result
        assert result[FlowStateKeys.STATUS] == WorkflowStatusEnum.INPUT_REQUIRED.value

    @pytest.mark.asyncio
    async def test_message_template_formatting(self, request_node, sample_state):
        """Test that message template is formatted with input variables."""
        with patch(
            "duo_workflow_service.agent_platform.v1.components.human_input.nodes.request_node.get_vars_from_state"
        ) as mock_get_vars:
            mock_get_vars.return_value = {"test_key": "test_value"}

            result = await request_node.run(sample_state)

            # Verify get_vars_from_state was called
            mock_get_vars.assert_called_once_with(request_node._inputs, sample_state)

            # Verify UI log entry was created with formatted content
            assert FlowStateKeys.UI_CHAT_LOG in result
            ui_log_entry = result[FlowStateKeys.UI_CHAT_LOG][0]
            assert ui_log_entry["content"] == "Formatted prompt content with test_value"

    @pytest.mark.asyncio
    async def test_ui_log_event_emission(self, request_node, sample_state):
        """Test that user_input_prompt UI log event is emitted."""
        with patch(
            "duo_workflow_service.agent_platform.v1.state.base.get_vars_from_state"
        ) as mock_get_vars:
            mock_get_vars.return_value = {"test_key": "test_value"}

            result = await request_node.run(sample_state)

            # Verify UI log entry was created
            assert FlowStateKeys.UI_CHAT_LOG in result
            assert len(result[FlowStateKeys.UI_CHAT_LOG]) == 1

            # Verify the log entry content
            ui_log_entry = result[FlowStateKeys.UI_CHAT_LOG][0]
            assert ui_log_entry["content"] == "Formatted prompt content with test_value"
            assert ui_log_entry["message_type"] == MessageTypeEnum.AGENT

    @pytest.mark.asyncio
    async def test_message_template_processed_when_present(self, sample_state):
        """Test that message template is processed and formatted correctly."""
        # Create RequestNode with ui_history and message_template
        request_node = RequestNode(
            name="test_component#request",
            component_name="test_component",
            message_template="Simple template without variables",
            inputs=[],
            ui_history=UIHistory(
                events=[
                    UILogEventsHumanInput.ON_USER_INPUT_PROMPT
                ],  # Include event so log is created
                writer_class=AgentLogWriter,
            ),
            status_key=IOKey(target="status"),
        )

        with patch(
            "duo_workflow_service.agent_platform.v1.components.human_input.nodes.request_node.get_vars_from_state"
        ) as mock_get_vars:
            mock_get_vars.return_value = {}

            result = await request_node.run(sample_state)

            # Should have status
            assert FlowStateKeys.STATUS in result
            assert (
                result[FlowStateKeys.STATUS] == WorkflowStatusEnum.INPUT_REQUIRED.value
            )

            # Should have UI log with the event in events list
            assert FlowStateKeys.UI_CHAT_LOG in result

            # Verify message template was processed
            ui_log_entry = result[FlowStateKeys.UI_CHAT_LOG][0]
            assert ui_log_entry["content"] == "Simple template without variables"

    @pytest.mark.asyncio
    async def test_input_variables_extraction(self, request_node, sample_state):
        """Test that input variables are correctly extracted from state."""
        with patch(
            "duo_workflow_service.agent_platform.v1.components.human_input.nodes.request_node.get_vars_from_state"
        ) as mock_get_vars:
            mock_get_vars.return_value = {"test_key": "test_value"}

            await request_node.run(sample_state)

            # Verify get_vars_from_state was called with correct inputs and state
            mock_get_vars.assert_called_once_with(request_node._inputs, sample_state)
