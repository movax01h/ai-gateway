import copy
from unittest.mock import AsyncMock, Mock, call, patch

import pytest
from anthropic import APIStatusError
from langchain_core.messages import AIMessage, ToolMessage

from ai_gateway.prompts import Prompt
from duo_workflow_service.agent_platform.experimental.components.agent.nodes.agent_node import (
    AgentFinalOutput,
    AgentNode,
)
from duo_workflow_service.agent_platform.experimental.state import FlowStateKeys
from duo_workflow_service.errors.error_handler import ModelError, ModelErrorType
from lib.internal_events.event_enum import CategoryEnum


@pytest.fixture(name="mock_prompt")
def mock_prompt_fixture(mock_ai_message):
    """Fixture for mock prompt."""
    mock_prompt = Mock(spec=Prompt)
    mock_prompt.model = Mock()
    mock_prompt.model.model_name = "claude-3-sonnet"
    mock_prompt.model_provider = "anthropic"
    mock_prompt.ainvoke = AsyncMock(return_value=mock_ai_message)

    return mock_prompt


@pytest.fixture(name="agent_node")
def agent_node_fixture(
    flow_id,
    mock_prompt,
    inputs,
    component_name,
    mock_internal_event_client,
):
    """Fixture for AgentNode instance."""
    return AgentNode(
        flow_id=flow_id,
        flow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        name="test_agent_node",
        prompt=mock_prompt,
        inputs=inputs,
        component_name=component_name,
        internal_event_client=mock_internal_event_client,
    )


@pytest.fixture(name="mock_get_vars_from_state")
def mock_get_vars_from_state_fixture(prompt_variables):
    with patch(
        "duo_workflow_service.agent_platform.experimental.components.agent.nodes.agent_node.get_vars_from_state"
    ) as mock_get_vars_from_state:
        mock_get_vars_from_state.return_value = prompt_variables
        yield mock_get_vars_from_state


class TestAgentNode:
    """Test suite for AgentNode class focusing on the run method."""

    @pytest.mark.asyncio
    async def test_run_success_with_empty_history(
        self,
        mock_prompt,
        inputs,
        agent_node,
        base_flow_state,
        mock_ai_message,
        component_name,
        prompt_variables,
        mock_get_vars_from_state,
    ):
        """Test successful run with empty conversation history."""

        result = await agent_node.run(base_flow_state)

        # Verify result structure
        assert FlowStateKeys.CONVERSATION_HISTORY in result
        assert component_name in result[FlowStateKeys.CONVERSATION_HISTORY]
        assert result[FlowStateKeys.CONVERSATION_HISTORY][component_name] == [
            mock_ai_message
        ]

        mock_get_vars_from_state.assert_called_once_with(
            inputs,
            base_flow_state,
        )
        mock_prompt.ainvoke.assert_called_once_with(
            input={
                **prompt_variables,
                "history": [],
            }
        )

    @pytest.mark.asyncio
    async def test_run_success_with_existing_history(
        self,
        mock_prompt,
        inputs,
        agent_node,
        flow_state_with_history,
        mock_ai_message,
        component_name,
        prompt_variables,
        mock_get_vars_from_state,
    ):
        """Test successful run with existing conversation history."""

        # Get the existing history from the state
        existing_history = flow_state_with_history[FlowStateKeys.CONVERSATION_HISTORY][
            component_name
        ]

        result = await agent_node.run(flow_state_with_history)

        # Verify result structure
        assert FlowStateKeys.CONVERSATION_HISTORY in result
        assert component_name in result[FlowStateKeys.CONVERSATION_HISTORY]

        # Verify component appends to existing conversation history
        result_history = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert (
            len(result_history) == len(existing_history) + 1
        ), "Expected existing messages plus new completion"
        assert (
            result_history[:-1] == existing_history
        ), "Existing history must be preserved"
        assert result_history[-1] == mock_ai_message, "New completion must be appended"

        mock_get_vars_from_state.assert_called_once_with(
            inputs,
            flow_state_with_history,
        )
        mock_prompt.ainvoke.assert_called_once_with(
            input={
                **prompt_variables,
                "history": flow_state_with_history[FlowStateKeys.CONVERSATION_HISTORY][
                    component_name
                ],
            }
        )

    @pytest.mark.asyncio
    async def test_run_with_missing_component_in_history(
        self,
        mock_prompt,
        inputs,
        agent_node,
        base_flow_state,
        mock_ai_message,
        component_name,
        prompt_variables,
        mock_get_vars_from_state,
    ):
        """Test run method with conversation_history missing the component key."""
        base_flow_state[FlowStateKeys.CONVERSATION_HISTORY] = {}

        result = await agent_node.run(base_flow_state)

        # Verify result structure
        assert FlowStateKeys.CONVERSATION_HISTORY in result
        assert component_name in result[FlowStateKeys.CONVERSATION_HISTORY]
        assert result[FlowStateKeys.CONVERSATION_HISTORY][component_name] == [
            mock_ai_message
        ]

        mock_get_vars_from_state.assert_called_once_with(
            inputs,
            base_flow_state,
        )
        mock_prompt.ainvoke.assert_called_once_with(
            input={
                **prompt_variables,
                "history": [],
            }
        )

    @pytest.mark.asyncio
    async def test_run_api_error(
        self,
        flow_id,
        inputs,
        component_name,
        mock_internal_event_client,
        base_flow_state,
        mock_ai_message,
        mock_prompt,
    ):
        """Test run method handles retryable API status errors."""
        # Create mock API error (429 - rate limit)
        mock_response = Mock()
        mock_response.status_code = 429
        api_error = APIStatusError(
            "Rate limit exceeded", response=mock_response, body=None
        )

        mock_prompt.ainvoke = AsyncMock(side_effect=[api_error, mock_ai_message])

        with patch(
            "duo_workflow_service.agent_platform.experimental.components.agent.nodes.agent_node.ModelErrorHandler",
        ) as mock_error_handler_cls:
            mock_error_handler = Mock()
            mock_error_handler_cls.return_value = mock_error_handler
            mock_error_handler.get_error_type.return_value = (
                ModelErrorType.RATE_LIMIT_ERROR
            )
            mock_error_handler.handle_error = AsyncMock()

            agent_node = AgentNode(
                flow_id=flow_id,
                flow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
                name="test_agent_node",
                prompt=mock_prompt,
                inputs=inputs,
                component_name=component_name,
                internal_event_client=mock_internal_event_client,
            )
            result = await agent_node.run(base_flow_state)

            # Verify error handler was called
            mock_error_handler.handle_error.assert_called_once()
            error = mock_error_handler.handle_error.call_args[0][0]
            assert isinstance(error, ModelError)
            assert error.error_type == ModelErrorType.RATE_LIMIT_ERROR
            assert error.status_code == 429
            assert error.message == "Rate limit exceeded"

            # Verify successful result after retry
            assert FlowStateKeys.CONVERSATION_HISTORY in result
            assert component_name in result[FlowStateKeys.CONVERSATION_HISTORY]
            assert result[FlowStateKeys.CONVERSATION_HISTORY][component_name] == [
                mock_ai_message
            ]

            # Verify prompt was called twice (first failed, second succeeded)
            assert mock_prompt.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_run_final_answer_combined_with_other_tools(
        self,
        mock_ai_message,
        mock_prompt,
        agent_node,
        base_flow_state,
        component_name,
        prompt_variables,
    ):
        """Test run method when final_response_tool is combined with other tools."""
        # Create mock AI message with final_response_tool and another tool
        mock_ai_message_invalid = copy.copy(mock_ai_message)
        mock_ai_message_invalid.tool_calls = [
            {
                "name": "final_response_tool",
                "id": "call_1",
                "args": {"final_response": "Done"},
            },
            {"name": "other_tool", "id": "call_2", "args": {"param": "value"}},
        ]

        # Configure prompt to return invalid message first, then valid
        mock_prompt.ainvoke = AsyncMock(
            side_effect=[mock_ai_message_invalid, mock_ai_message]
        )

        result = await agent_node.run(base_flow_state)

        # Verify prompt was called twice (first failed validation, second succeeded)
        assert mock_prompt.ainvoke.call_count == 2
        retry_history = [
            mock_ai_message_invalid,
            ToolMessage(
                tool_call_id="call_1",
                content=f"{AgentFinalOutput.tool_title} mustn't be combined with other tool calls",
            ),
            ToolMessage(
                tool_call_id="call_2",
                content=f"{AgentFinalOutput.tool_title} mustn't be combined with other tool calls",
            ),
        ]
        mock_prompt.ainvoke.assert_has_calls(
            [
                call(input={**prompt_variables, "history": []}),
                call(input={**prompt_variables, "history": retry_history}),
            ]
        )

        # Verify retry mechanism preserves full conversation history
        assert FlowStateKeys.CONVERSATION_HISTORY in result
        assert component_name in result[FlowStateKeys.CONVERSATION_HISTORY]

        # Expected: invalid attempt + 2 validation error responses + successful retry
        result_history = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert len(result_history) == 4, "Retry history must be preserved for debugging"
        assert result_history[0] == mock_ai_message_invalid
        assert isinstance(result_history[1], ToolMessage)
        assert result_history[1].tool_call_id == "call_1"
        assert isinstance(result_history[2], ToolMessage)
        assert result_history[2].tool_call_id == "call_2"
        assert result_history[3] == mock_ai_message

    @pytest.mark.asyncio
    async def test_run_valid_final_answer_tool(
        self,
        mock_ai_message,
        mock_prompt,
        agent_node,
        base_flow_state,
        component_name,
    ):
        """Test run method with valid final_response_tool."""
        # Create mock AI message with valid final_response_tool
        mock_ai_message.tool_calls = [
            {
                "name": "final_response_tool",
                "id": "call_1",
                "args": {"final_response": "Task completed successfully"},
            },
        ]

        mock_prompt.ainvoke = AsyncMock(return_value=mock_ai_message)

        result = await agent_node.run(base_flow_state)

        # Verify successful result
        assert FlowStateKeys.CONVERSATION_HISTORY in result
        assert component_name in result[FlowStateKeys.CONVERSATION_HISTORY]
        assert result[FlowStateKeys.CONVERSATION_HISTORY][component_name] == [
            mock_ai_message
        ]

        # Verify prompt was called only once (validation passed)
        assert agent_node._prompt.ainvoke.call_count == 1

    @pytest.mark.asyncio
    async def test_run_invalid_final_answer_tool_validation_error(
        self,
        prompt_variables,
        mock_ai_message,
        mock_prompt,
        agent_node,
        base_flow_state,
        component_name,
    ):
        """Test run method when final_response_tool has validation error."""
        # Create mock AI message with invalid final_response_tool args
        mock_ai_message_invalid = copy.copy(mock_ai_message)
        mock_ai_message_invalid.tool_calls = [
            {
                "name": "final_response_tool",
                "id": "call_1",
                "args": {"wrong_field": "value"},
            },  # Missing required field
        ]

        mock_prompt.ainvoke = AsyncMock(
            side_effect=[mock_ai_message_invalid, mock_ai_message]
        )

        result = await agent_node.run(base_flow_state)

        # Verify retry mechanism preserves full conversation history
        assert FlowStateKeys.CONVERSATION_HISTORY in result
        assert component_name in result[FlowStateKeys.CONVERSATION_HISTORY]

        # Expected: invalid attempt + validation error response + successful retry
        result_history = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert len(result_history) == 3, "Retry history must be preserved for debugging"
        assert result_history[0] == mock_ai_message_invalid
        assert isinstance(result_history[1], ToolMessage)
        assert result_history[2] == mock_ai_message

        # Verify prompt was called twice (first failed validation, second succeeded)
        assert mock_prompt.ainvoke.call_count == 2

        prompt_calls = mock_prompt.ainvoke.call_args_list
        assert prompt_calls[0] == call(input={**prompt_variables, "history": []})

        retry_messages_history = prompt_calls[1][1]["input"]["history"]
        assert len(retry_messages_history) == 2
        assert isinstance(retry_messages_history[0], AIMessage)
        assert isinstance(retry_messages_history[1], ToolMessage)
        assert retry_messages_history[1].tool_call_id == "call_1"
        assert (
            f"{AgentFinalOutput.tool_title} raised validation error:"
            in retry_messages_history[1].content
        )
