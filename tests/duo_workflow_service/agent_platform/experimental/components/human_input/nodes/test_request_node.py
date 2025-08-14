from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from ai_gateway.prompts.base import Prompt
from duo_workflow_service.agent_platform.experimental.components.human_input.nodes.request_node import (
    RequestNode,
)
from duo_workflow_service.agent_platform.experimental.components.human_input.ui_log import (
    AgentLogWriter,
    UILogEventsHumanInput,
)
from duo_workflow_service.agent_platform.experimental.state import FlowStateKeys, IOKey
from duo_workflow_service.agent_platform.experimental.ui_log import UIHistory
from duo_workflow_service.entities import MessageTypeEnum
from duo_workflow_service.entities.state import WorkflowStatusEnum


class TestRequestNode:
    """Test suite for RequestNode."""

    @pytest.fixture
    def mock_prompt(self):
        """Mock prompt."""
        prompt = Mock(spec=Prompt)
        prompt_tpl = Mock(spec=ChatPromptTemplate)

        # Mock the invoke method to return messages
        mock_messages = [HumanMessage(content="Formatted prompt content")]
        prompt_tpl.invoke.return_value.messages = mock_messages
        prompt.prompt_tpl = prompt_tpl

        return prompt

    @pytest.fixture
    def request_node(self, mock_prompt):
        """Create RequestNode instance for testing."""
        return RequestNode(
            name="test_component#request",
            component_name="test_component",
            prompt=mock_prompt,
            inputs=[IOKey(target="context", subkeys=["test_key"])],
            ui_history=UIHistory(
                events=[UILogEventsHumanInput.ON_USER_INPUT_PROMPT],
                writer_class=AgentLogWriter,
            ),
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
    async def test_prompt_registry_sourcing(
        self, request_node, sample_state, mock_prompt
    ):
        """Test that prompt is sourced from prompt registry and formatted."""
        with patch(
            "duo_workflow_service.agent_platform.experimental.state.base.get_vars_from_state"
        ) as mock_get_vars:
            mock_get_vars.return_value = {"test_key": "test_value"}

            await request_node.run(sample_state)

            # Verify prompt template was invoked with input variables
            mock_prompt.prompt_tpl.invoke.assert_called_once_with(
                {"test_key": "test_value"}
            )

    @pytest.mark.asyncio
    async def test_ui_log_event_emission(self, request_node, sample_state):
        """Test that user_input_prompt UI log event is emitted."""
        with patch(
            "duo_workflow_service.agent_platform.experimental.state.base.get_vars_from_state"
        ) as mock_get_vars:
            mock_get_vars.return_value = {"test_key": "test_value"}

            result = await request_node.run(sample_state)

            # Verify UI log entry was created
            assert FlowStateKeys.UI_CHAT_LOG in result
            assert len(result[FlowStateKeys.UI_CHAT_LOG]) == 1

            # Verify the log entry content
            ui_log_entry = result[FlowStateKeys.UI_CHAT_LOG][0]
            assert ui_log_entry["content"] == "Formatted prompt content"
            assert ui_log_entry["message_type"] == MessageTypeEnum.AGENT

    @pytest.mark.asyncio
    async def test_no_ui_log_when_both_prompt_and_ui_history_missing(
        self, sample_state
    ):
        """Test that no UI log is emitted when both prompt and ui_history are missing."""
        request_node = RequestNode(
            name="test_component#request",
            component_name="test_component",
            prompt=None,
            inputs=[],
            ui_history=None,
        )

        result = await request_node.run(sample_state)

        # Should only have status, no UI log
        assert FlowStateKeys.STATUS in result
        assert result[FlowStateKeys.STATUS] == WorkflowStatusEnum.INPUT_REQUIRED.value
        assert FlowStateKeys.UI_CHAT_LOG not in result

    @pytest.mark.asyncio
    async def test_ui_log_always_emitted_when_ui_history_present(
        self, mock_prompt, sample_state
    ):
        """Test that UI log is always emitted when ui_history is present, regardless of events."""
        # Create RequestNode with ui_history but without the specific event
        request_node = RequestNode(
            name="test_component#request",
            component_name="test_component",
            prompt=mock_prompt,
            inputs=[],
            ui_history=UIHistory(
                events=[],  # No ON_USER_INPUT_PROMPT event - but should still log
                writer_class=AgentLogWriter,
            ),
        )

        with patch(
            "duo_workflow_service.agent_platform.experimental.components.human_input.nodes.request_node.get_vars_from_state"  # pylint: disable=line-too-long
        ) as mock_get_vars:
            mock_get_vars.return_value = {}

            result = await request_node.run(sample_state)

            # Should have status
            assert FlowStateKeys.STATUS in result
            assert (
                result[FlowStateKeys.STATUS] == WorkflowStatusEnum.INPUT_REQUIRED.value
            )

            # Should have UI log even though event is not in events list (underlying framework handles this)
            assert FlowStateKeys.UI_CHAT_LOG in result
            # The underlying logger framework will decide whether to actually log based on events

            # Verify prompt was processed
            mock_prompt.prompt_tpl.invoke.assert_called_once_with({})

    @pytest.mark.asyncio
    async def test_validation_prompt_and_ui_history_both_or_neither(self, mock_prompt):
        """Test that RequestNode validation ensures prompt and ui_history are both present or both missing."""
        # Test case 1: ui_history present but prompt missing should fail
        with pytest.raises(
            ValueError,
            match="prompt and ui_history must be either both present or both missing",
        ):
            RequestNode(
                name="test_component#request",
                component_name="test_component",
                prompt=None,  # No prompt
                inputs=[],
                ui_history=UIHistory(
                    events=[UILogEventsHumanInput.ON_USER_INPUT_PROMPT],
                    writer_class=AgentLogWriter,
                ),
            )

        # Test case 2: prompt present but ui_history missing should fail
        with pytest.raises(
            ValueError,
            match="prompt and ui_history must be either both present or both missing",
        ):
            RequestNode(
                name="test_component#request",
                component_name="test_component",
                prompt=mock_prompt,  # Prompt present
                inputs=[],
                ui_history=None,  # No ui_history
            )

        # Test case 3: both present should succeed
        try:
            RequestNode(
                name="test_component#request",
                component_name="test_component",
                prompt=mock_prompt,
                inputs=[],
                ui_history=UIHistory(
                    events=[UILogEventsHumanInput.ON_USER_INPUT_PROMPT],
                    writer_class=AgentLogWriter,
                ),
            )
        except ValueError:
            pytest.fail("RequestNode should accept both prompt and ui_history present")

        # Test case 4: both missing should succeed
        try:
            RequestNode(
                name="test_component#request",
                component_name="test_component",
                prompt=None,
                inputs=[],
                ui_history=None,
            )
        except ValueError:
            pytest.fail("RequestNode should accept both prompt and ui_history missing")

    @pytest.mark.asyncio
    async def test_input_variables_extraction(self, request_node, sample_state):
        """Test that input variables are correctly extracted from state."""
        with patch(
            "duo_workflow_service.agent_platform.experimental.components.human_input.nodes.request_node.get_vars_from_state"  # pylint: disable=line-too-long
        ) as mock_get_vars:
            mock_get_vars.return_value = {"test_key": "test_value"}

            await request_node.run(sample_state)

            # Verify get_vars_from_state was called with correct inputs and state
            mock_get_vars.assert_called_once_with(request_node.inputs, sample_state)
