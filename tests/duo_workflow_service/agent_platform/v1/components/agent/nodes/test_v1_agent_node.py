# pylint: disable=file-naming-for-tests, too-many-lines
import copy
import importlib
from datetime import datetime
from unittest.mock import AsyncMock, Mock, call, patch

import pytest
from anthropic import APIStatusError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_gateway.prompts import Prompt
from duo_workflow_service.agent_platform.v1.components.agent.nodes.agent_node import (
    _LITELLM_EMPTY_CONTENT_PLACEHOLDER,
    AgentFinalOutput,
    AgentNode,
    AgentStuckError,
)
from duo_workflow_service.agent_platform.v1.components.agent.ui_log import (
    UILogEventsAgent,
    agent_tools_ui_log_writer_class,
)
from duo_workflow_service.agent_platform.v1.state import (
    FlowStateKeys,
    IOKey,
    RuntimeIOKey,
)
from duo_workflow_service.agent_platform.v1.ui_log import UIHistory
from duo_workflow_service.conversation.compaction import ConversationCompactor
from duo_workflow_service.entities import MessageTypeEnum
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


@pytest.fixture(name="conversation_history_key")
def conversation_history_key_fixture(component_name):
    """Fixture for conversation history IOKey."""
    return IOKey(
        target="conversation_history",
        subkeys=[component_name],
        optional=True,
    )


@pytest.fixture(name="agent_node")
def agent_node_fixture(
    flow_id,
    mock_prompt,
    inputs,
    conversation_history_key,
    mock_internal_event_client,
):
    """Fixture for AgentNode instance (default, no response schema)."""
    return AgentNode(
        flow_id=flow_id,
        flow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        name="test_agent_node",
        prompt=mock_prompt,
        inputs=inputs,
        conversation_history_key=RuntimeIOKey(
            alias="conversation_history", factory=lambda _: conversation_history_key
        ),
        internal_event_client=mock_internal_event_client,
        invoke_config={},
    )


@pytest.fixture(name="agent_node_with_schema")
def agent_node_with_schema_fixture(
    flow_id,
    mock_prompt,
    inputs,
    conversation_history_key,
    mock_internal_event_client,
):
    """Fixture for AgentNode instance with AgentFinalOutput response schema."""
    return AgentNode(
        flow_id=flow_id,
        flow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        name="test_agent_node",
        prompt=mock_prompt,
        inputs=inputs,
        conversation_history_key=RuntimeIOKey(
            alias="conversation_history", factory=lambda _: conversation_history_key
        ),
        internal_event_client=mock_internal_event_client,
        invoke_config={},
        response_schema=AgentFinalOutput,
    )


@pytest.fixture(name="mock_get_vars_from_state")
def mock_get_vars_from_state_fixture(prompt_variables):
    with patch(
        "duo_workflow_service.agent_platform.v1.components.agent.nodes.agent_node.get_vars_from_state"
    ) as mock_get_vars_from_state:
        mock_get_vars_from_state.return_value = prompt_variables
        yield mock_get_vars_from_state


@pytest.fixture(name="_mock_get_vars_from_state")
def _mock_get_vars_from_state_fixture(prompt_variables):
    """Side-effect fixture: mocks get_vars_from_state (use when return value is not needed in test)."""
    with patch(
        "duo_workflow_service.agent_platform.v1.components.agent.nodes.agent_node.get_vars_from_state"
    ) as mock_get_vars_from_state:
        mock_get_vars_from_state.return_value = prompt_variables
        yield mock_get_vars_from_state


@pytest.fixture(name="_mock_maybe_compact_history")
def _mock_maybe_compact_history_fixture():
    """Fixture for mocking maybe_compact_history to return input unchanged."""
    with patch(
        "duo_workflow_service.agent_platform.v1.components.agent.nodes.agent_node.maybe_compact_history",
        new_callable=AsyncMock,
    ) as mock_compact:

        async def return_history(*, compactor, history, agent_name):
            _ = compactor, agent_name
            return history, None

        mock_compact.side_effect = return_history
        yield mock_compact


FAKE_RUNTIME_VARS = {
    "current_date": "2026-01-01",
    "current_time": "12:00:00",
    "current_timezone": "UTC",
}


@pytest.fixture(name="_mock_predefined_runtime_variables")
def _mock_predefined_runtime_variables_fixture():
    """Fixture for mocking _predefined_runtime_variables to return deterministic values."""
    with patch.object(
        AgentNode,
        "_predefined_runtime_variables",
        return_value=FAKE_RUNTIME_VARS,
    ):
        yield


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
        _mock_maybe_compact_history,
        _mock_predefined_runtime_variables,
    ):
        """Test successful run with empty conversation history."""

        result = await agent_node.run(base_flow_state)

        # Verify result structure (with replace mode: full history returned)
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
                **FAKE_RUNTIME_VARS,
            },
            config={},
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
        _mock_maybe_compact_history,
        _mock_predefined_runtime_variables,
    ):
        """Test successful run with existing conversation history."""
        existing_history = flow_state_with_history[FlowStateKeys.CONVERSATION_HISTORY][
            component_name
        ]

        result = await agent_node.run(flow_state_with_history)

        # Verify result structure (with replace mode: full history returned)
        assert FlowStateKeys.CONVERSATION_HISTORY in result
        assert component_name in result[FlowStateKeys.CONVERSATION_HISTORY]
        # Replace mode: existing history + new completion
        assert result[FlowStateKeys.CONVERSATION_HISTORY][component_name] == [
            *existing_history,
            mock_ai_message,
        ]

        mock_get_vars_from_state.assert_called_once_with(
            inputs,
            flow_state_with_history,
        )
        mock_prompt.ainvoke.assert_called_once_with(
            input={
                **prompt_variables,
                "history": existing_history,
                **FAKE_RUNTIME_VARS,
            },
            config={},
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
        _mock_maybe_compact_history,
        _mock_predefined_runtime_variables,
    ):
        """Test run method with conversation_history missing the component key."""
        base_flow_state[FlowStateKeys.CONVERSATION_HISTORY] = {}

        result = await agent_node.run(base_flow_state)

        # Verify result structure (with replace mode: full history returned)
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
                **FAKE_RUNTIME_VARS,
            },
            config={},
        )

    @pytest.mark.asyncio
    async def test_run_api_error(
        self,
        flow_id,
        inputs,
        conversation_history_key,
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
            "duo_workflow_service.agent_platform.v1.components.agent.nodes.agent_node.ModelErrorHandler",
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
                conversation_history_key=RuntimeIOKey(
                    alias="conversation_history",
                    factory=lambda _: conversation_history_key,
                ),
                internal_event_client=mock_internal_event_client,
                invoke_config={},
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
        agent_node_with_schema,
        base_flow_state,
        component_name,
        prompt_variables,
        _mock_maybe_compact_history,
        _mock_predefined_runtime_variables,
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

        result = await agent_node_with_schema.run(base_flow_state)

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
                call(
                    input={**prompt_variables, "history": [], **FAKE_RUNTIME_VARS},
                    config={},
                ),
                call(
                    input={
                        **prompt_variables,
                        "history": retry_history,
                        **FAKE_RUNTIME_VARS,
                    },
                    config={},
                ),
            ]
        )

        # Verify the method retried and returned successful result (with replace mode: full history)
        assert FlowStateKeys.CONVERSATION_HISTORY in result
        assert component_name in result[FlowStateKeys.CONVERSATION_HISTORY]
        # Replace mode returns full history: retry_history + final completion
        assert result[FlowStateKeys.CONVERSATION_HISTORY][component_name] == [
            *retry_history,
            mock_ai_message,
        ]

    @pytest.mark.asyncio
    async def test_run_valid_final_answer_tool(
        self,
        mock_ai_message,
        mock_prompt,
        agent_node_with_schema,
        base_flow_state,
        component_name,
        _mock_maybe_compact_history,
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

        result = await agent_node_with_schema.run(base_flow_state)

        # Verify successful result (with replace mode: full history)
        assert FlowStateKeys.CONVERSATION_HISTORY in result
        assert component_name in result[FlowStateKeys.CONVERSATION_HISTORY]
        assert result[FlowStateKeys.CONVERSATION_HISTORY][component_name] == [
            mock_ai_message
        ]

        # Verify prompt was called only once (validation passed)
        assert agent_node_with_schema._prompt.ainvoke.call_count == 1

    @pytest.mark.asyncio
    async def test_run_invalid_final_answer_tool_validation_error(
        self,
        prompt_variables,
        mock_ai_message,
        mock_prompt,
        agent_node_with_schema,
        base_flow_state,
        component_name,
        _mock_maybe_compact_history,
        _mock_predefined_runtime_variables,
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

        result = await agent_node_with_schema.run(base_flow_state)

        # Verify the method retried and returned successful result (with replace mode)
        assert FlowStateKeys.CONVERSATION_HISTORY in result
        assert component_name in result[FlowStateKeys.CONVERSATION_HISTORY]
        # Replace mode returns full history: retry messages + final completion
        history = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert len(history) == 3  # invalid AI message + tool message + final AI message
        assert history[-1] == mock_ai_message

        # Verify prompt was called twice (first failed validation, second succeeded)
        assert mock_prompt.ainvoke.call_count == 2

        prompt_calls = mock_prompt.ainvoke.call_args_list
        assert prompt_calls[0] == call(
            input={**prompt_variables, "history": [], **FAKE_RUNTIME_VARS},
            config={},
        )

        retry_messages_history = prompt_calls[1][1]["input"]["history"]
        assert len(retry_messages_history) == 2
        assert isinstance(retry_messages_history[0], AIMessage)
        assert isinstance(retry_messages_history[1], ToolMessage)
        assert retry_messages_history[1].tool_call_id == "call_1"
        assert (
            f"{AgentFinalOutput.tool_title} raised validation error:"
            in retry_messages_history[1].content
        )


class TestAgentNodeContextLimits:
    """Test suite for AgentNode stamping per-agent context-window limits."""

    @pytest.fixture(name="agent_node_with_limit")
    def agent_node_with_limit_fixture(
        self,
        flow_id,
        mock_prompt,
        inputs,
        conversation_history_key,
        mock_internal_event_client,
    ):
        """AgentNode constructed with an explicit per-agent max_context_tokens."""
        return AgentNode(
            flow_id=flow_id,
            flow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
            name="test_agent_node",
            prompt=mock_prompt,
            inputs=inputs,
            conversation_history_key=RuntimeIOKey(
                alias="conversation_history",
                factory=lambda _: conversation_history_key,
            ),
            internal_event_client=mock_internal_event_client,
            invoke_config={},
            max_context_tokens=64000,
        )

    @pytest.mark.asyncio
    async def test_run_stamps_agent_context_limits_keyed_by_history_slot(
        self,
        agent_node_with_limit,
        base_flow_state,
        component_name,
        _mock_get_vars_from_state,
        _mock_maybe_compact_history,
        _mock_predefined_runtime_variables,
    ):
        """The resolved max is stamped into agent_context_limits under the conversation_history key."""
        result = await agent_node_with_limit.run(base_flow_state)

        assert FlowStateKeys.AGENT_CONTEXT_LIMITS in result
        assert result[FlowStateKeys.AGENT_CONTEXT_LIMITS] == {component_name: 64000}


class TestAgentNodeCompaction:
    """Test suite for AgentNode compaction support."""

    @pytest.fixture(name="mock_compactor")
    def mock_compactor_fixture(self):
        """Fixture for mock ConversationCompactor."""
        return Mock(spec=ConversationCompactor)

    @pytest.fixture(name="agent_node_with_compactor")
    def agent_node_with_compactor_fixture(
        self,
        flow_id,
        mock_prompt,
        inputs,
        conversation_history_key,
        mock_internal_event_client,
        mock_compactor,
    ):
        """Fixture for AgentNode instance with compactor."""
        return AgentNode(
            flow_id=flow_id,
            flow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
            name="test_agent_node",
            prompt=mock_prompt,
            inputs=inputs,
            conversation_history_key=RuntimeIOKey(
                alias="conversation_history", factory=lambda _: conversation_history_key
            ),
            internal_event_client=mock_internal_event_client,
            invoke_config={},
            compactor=mock_compactor,
        )

    def test_agent_node_stores_compactor(
        self,
        agent_node_with_compactor,
        mock_compactor,
    ):
        """Test that AgentNode stores the compactor when provided."""
        assert agent_node_with_compactor._compactor == mock_compactor

    def test_agent_node_without_compactor_has_none(
        self,
        agent_node,
    ):
        """Test that AgentNode has None compactor when not provided."""
        assert agent_node._compactor is None

    @pytest.mark.asyncio
    async def test_run_calls_maybe_compact_history_with_compactor(
        self,
        agent_node_with_compactor,
        base_flow_state,
        mock_compactor,
        _mock_get_vars_from_state,
    ):
        """Test that run() passes the compactor to maybe_compact_history."""
        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent.nodes.agent_node.maybe_compact_history",
            new_callable=AsyncMock,
        ) as mock_compact:
            mock_compact.return_value = ([], None)

            await agent_node_with_compactor.run(base_flow_state)

            mock_compact.assert_called_once_with(
                compactor=mock_compactor,
                history=[],
                agent_name="test_agent_node",
            )

    @pytest.mark.asyncio
    async def test_run_calls_maybe_compact_history_without_compactor(
        self,
        agent_node,
        base_flow_state,
        _mock_get_vars_from_state,
    ):
        """Test that run() passes None to maybe_compact_history when no compactor."""
        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent.nodes.agent_node.maybe_compact_history",
            new_callable=AsyncMock,
        ) as mock_compact:
            mock_compact.return_value = ([], None)

            await agent_node.run(base_flow_state)

            mock_compact.assert_called_once_with(
                compactor=None,
                history=[],
                agent_name="test_agent_node",
            )


class TestAgentNodeTruncation:
    """Test suite for AgentNode truncation recovery support."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "truncation_finish_reason,metadata_key",
        [
            ("length", "finish_reason"),
            ("max_tokens", "stop_reason"),
        ],
    )
    async def test_run_truncated_response_retries_internally(
        self,
        truncation_finish_reason,
        metadata_key,
        mock_ai_message,
        mock_prompt,
        agent_node,
        base_flow_state,
        component_name,
        prompt_variables,
        _mock_maybe_compact_history,
        _mock_predefined_runtime_variables,
    ):
        """Test that a truncated LLM response retries internally within AgentNode."""
        truncated_message = copy.copy(mock_ai_message)
        truncated_message.content = "I was writing a long response but got cut off..."
        truncated_message.response_metadata = {metadata_key: truncation_finish_reason}
        truncated_message.tool_calls = []

        # First call returns truncated, second call returns a normal completion
        normal_message = copy.copy(mock_ai_message)
        normal_message.response_metadata = {}
        normal_message.tool_calls = []
        mock_prompt.ainvoke = AsyncMock(side_effect=[truncated_message, normal_message])

        result = await agent_node.run(base_flow_state)

        # Called twice — internal retry after truncation
        assert mock_prompt.ainvoke.call_count == 2

        # Second call should include the truncated message + recovery HumanMessage in history
        expected_retry_history = [
            truncated_message,
            HumanMessage(content=AgentNode._TRUNCATION_RECOVERY_MESSAGE),
        ]
        mock_prompt.ainvoke.assert_has_calls(
            [
                call(
                    input={
                        **prompt_variables,
                        "history": [],
                        **FAKE_RUNTIME_VARS,
                    },
                    config={},
                ),
                call(
                    input={
                        **prompt_variables,
                        "history": expected_retry_history,
                        **FAKE_RUNTIME_VARS,
                    },
                    config={},
                ),
            ]
        )

        # Final history includes truncated message + recovery message + normal completion
        result_history = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert len(result_history) == 3
        assert result_history[0] == truncated_message
        assert isinstance(result_history[1], HumanMessage)
        assert "cut off" in result_history[1].content
        assert "concise" in result_history[1].content
        assert result_history[2] == normal_message

    @pytest.mark.asyncio
    async def test_run_non_truncation_abnormal_finish_reason_does_not_retry(
        self,
        mock_ai_message,
        mock_prompt,
        agent_node,
        base_flow_state,
        component_name,
    ):
        """Test that non-truncation abnormal finish reasons (e.g. content_filter) do not trigger recovery retry."""
        content_filter_message = copy.copy(mock_ai_message)
        content_filter_message.response_metadata = {"finish_reason": "content_filter"}
        content_filter_message.tool_calls = []

        mock_prompt.ainvoke = AsyncMock(return_value=content_filter_message)

        result = await agent_node.run(base_flow_state)

        # Should only be called once — no retry for content_filter
        assert mock_prompt.ainvoke.call_count == 1

        result_history = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert result_history == [content_filter_message]

    @pytest.mark.asyncio
    async def test_run_raises_agent_stuck_error_after_max_truncation_retries(
        self,
        mock_ai_message,
        mock_prompt,
        agent_node,
        base_flow_state,
        _mock_maybe_compact_history,
    ):
        """Test that AgentStuckError is raised after exceeding max truncation retries."""
        truncated_message = copy.copy(mock_ai_message)
        truncated_message.content = "I was writing a long response but got cut off..."
        truncated_message.response_metadata = {"finish_reason": "length"}
        truncated_message.tool_calls = []

        # Always return truncated response
        mock_prompt.ainvoke = AsyncMock(return_value=truncated_message)

        with pytest.raises(AgentStuckError) as exc_info:
            await agent_node.run(base_flow_state)

        assert agent_node.name in str(exc_info.value)
        assert str(AgentNode._MAX_TRUNCATION_RETRIES) in str(exc_info.value)
        # Should have been called exactly MAX_TRUNCATION_RETRIES times
        assert mock_prompt.ainvoke.call_count == AgentNode._MAX_TRUNCATION_RETRIES

    @pytest.mark.asyncio
    async def test_run_truncation_retry_counter_resets_between_runs(
        self,
        mock_ai_message,
        mock_prompt,
        agent_node,
        base_flow_state,
        component_name,
        _mock_maybe_compact_history,
    ):
        """Test that the truncation retry counter is local to each run() call."""
        truncated_message = copy.copy(mock_ai_message)
        truncated_message.content = "I was writing a long response but got cut off..."
        truncated_message.response_metadata = {"finish_reason": "length"}
        truncated_message.tool_calls = []

        normal_message = copy.copy(mock_ai_message)
        normal_message.response_metadata = {}
        normal_message.tool_calls = []

        # First run: truncated once, then normal
        mock_prompt.ainvoke = AsyncMock(side_effect=[truncated_message, normal_message])
        await agent_node.run(base_flow_state)

        # Second run: truncated once, then normal — should not accumulate retries from first run
        mock_prompt.ainvoke = AsyncMock(side_effect=[truncated_message, normal_message])
        result = await agent_node.run(base_flow_state)

        result_history = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert result_history[-1] == normal_message


class TestAgentNodeReasoning:
    """Test suite for AgentNode reasoning log emission."""

    @pytest.fixture(name="ui_history_with_reasoning")
    def ui_history_with_reasoning_fixture(self):
        """Fixture for UIHistory with ON_AGENT_REASONING enabled."""
        return UIHistory(
            events=[UILogEventsAgent.ON_AGENT_REASONING],
            writer_class=agent_tools_ui_log_writer_class(component_name="test_agent"),
        )

    @pytest.fixture(name="agent_node_with_ui_history")
    def agent_node_with_ui_history_fixture(
        self,
        flow_id,
        mock_prompt,
        inputs,
        conversation_history_key,
        mock_internal_event_client,
        ui_history_with_reasoning,
    ):
        """Fixture for AgentNode with ui_history enabled."""
        return AgentNode(
            flow_id=flow_id,
            flow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
            name="test_agent_node",
            prompt=mock_prompt,
            inputs=inputs,
            conversation_history_key=RuntimeIOKey(
                alias="conversation_history", factory=lambda _: conversation_history_key
            ),
            internal_event_client=mock_internal_event_client,
            invoke_config={},
            ui_history=ui_history_with_reasoning,
        )

    @pytest.mark.asyncio
    async def test_no_reasoning_for_text_only_message(
        self,
        mock_prompt,
        agent_node_with_ui_history,
        base_flow_state,
        _mock_get_vars_from_state,
        _mock_maybe_compact_history,
    ):
        """Test that a text-only AIMessage (no tool calls) produces no ON_AGENT_REASONING entry.

        Text-only completions are routed to FinalResponseNode which emits ON_AGENT_FINAL_ANSWER. Emitting
        ON_AGENT_REASONING here too would duplicate the text in the session view.
        """
        text_message = AIMessage(content="I will now look at the codebase.")
        mock_prompt.ainvoke = AsyncMock(return_value=text_message)

        result = await agent_node_with_ui_history.run(base_flow_state)

        reasoning_logs = [
            log
            for log in result.get(FlowStateKeys.UI_CHAT_LOG, [])
            if log["message_type"] == MessageTypeEnum.AGENT
        ]
        assert len(reasoning_logs) == 0

    @pytest.mark.asyncio
    async def test_reasoning_emitted_for_mixed_content_and_tool_calls(
        self,
        mock_prompt,
        agent_node_with_ui_history,
        base_flow_state,
        _mock_get_vars_from_state,
        _mock_maybe_compact_history,
    ):
        """Test that an AIMessage with text AND tool calls produces a reasoning log entry."""
        mixed_message = AIMessage(
            content="Now let me look at the existing schema to understand the structure:",
            tool_calls=[
                {
                    "id": "toolu_01",
                    "name": "gitlab_api_get",
                    "args": {"endpoint": "/api/v4/projects/123/repository/tree"},
                }
            ],
        )
        mock_prompt.ainvoke = AsyncMock(return_value=mixed_message)

        result = await agent_node_with_ui_history.run(base_flow_state)

        assert FlowStateKeys.UI_CHAT_LOG in result
        reasoning_logs = [
            log
            for log in result[FlowStateKeys.UI_CHAT_LOG]
            if log["message_type"] == MessageTypeEnum.AGENT
        ]
        assert len(reasoning_logs) == 1
        assert (
            reasoning_logs[0]["content"]
            == "Now let me look at the existing schema to understand the structure:"
        )

    @pytest.mark.asyncio
    async def test_no_reasoning_emitted_for_tool_calls_only(
        self,
        mock_prompt,
        agent_node_with_ui_history,
        base_flow_state,
        _mock_get_vars_from_state,
        _mock_maybe_compact_history,
    ):
        """Test that an AIMessage with tool calls but no text produces no reasoning log."""
        tool_only_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "toolu_01",
                    "name": "read_file",
                    "args": {"file_path": "README.md"},
                }
            ],
        )
        mock_prompt.ainvoke = AsyncMock(return_value=tool_only_message)

        result = await agent_node_with_ui_history.run(base_flow_state)

        reasoning_logs = [
            log
            for log in result.get(FlowStateKeys.UI_CHAT_LOG, [])
            if log["message_type"] == MessageTypeEnum.AGENT
        ]
        assert len(reasoning_logs) == 0

    @pytest.mark.asyncio
    async def test_no_reasoning_emitted_for_whitespace_only_content(
        self,
        mock_prompt,
        agent_node_with_ui_history,
        base_flow_state,
        _mock_get_vars_from_state,
        _mock_maybe_compact_history,
    ):
        """Test that whitespace-only content produces no reasoning log."""
        whitespace_message = AIMessage(content="   \n\t  ")
        mock_prompt.ainvoke = AsyncMock(return_value=whitespace_message)

        result = await agent_node_with_ui_history.run(base_flow_state)

        reasoning_logs = [
            log
            for log in result.get(FlowStateKeys.UI_CHAT_LOG, [])
            if log["message_type"] == MessageTypeEnum.AGENT
        ]
        assert len(reasoning_logs) == 0

    @pytest.mark.asyncio
    async def test_reasoning_emitted_for_list_content_with_text_block(
        self,
        mock_prompt,
        agent_node_with_ui_history,
        base_flow_state,
        _mock_get_vars_from_state,
        _mock_maybe_compact_history,
    ):
        """Test that list-of-blocks content with a text block produces a reasoning log."""
        list_content_message = AIMessage(
            content=[
                {"type": "text", "text": "Let me analyze the code structure."},
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "read_file",
                    "input": {},
                },
            ],
            tool_calls=[
                {
                    "id": "toolu_01",
                    "name": "read_file",
                    "args": {"file_path": "README.md"},
                }
            ],
        )
        mock_prompt.ainvoke = AsyncMock(return_value=list_content_message)

        result = await agent_node_with_ui_history.run(base_flow_state)

        assert FlowStateKeys.UI_CHAT_LOG in result
        reasoning_logs = [
            log
            for log in result[FlowStateKeys.UI_CHAT_LOG]
            if log["message_type"] == MessageTypeEnum.AGENT
        ]
        assert len(reasoning_logs) == 1
        assert reasoning_logs[0]["content"] == "Let me analyze the code structure."

    @pytest.mark.asyncio
    async def test_no_reasoning_without_ui_history(
        self,
        mock_prompt,
        agent_node,
        base_flow_state,
        _mock_get_vars_from_state,
        _mock_maybe_compact_history,
    ):
        """Test that no reasoning log is emitted when ui_history is not provided."""
        text_message = AIMessage(content="I will now look at the codebase.")
        mock_prompt.ainvoke = AsyncMock(return_value=text_message)

        result = await agent_node.run(base_flow_state)

        # No UI_CHAT_LOG key should be present when ui_history is None
        assert FlowStateKeys.UI_CHAT_LOG not in result

    @pytest.mark.asyncio
    async def test_no_reasoning_when_event_not_in_ui_history(
        self,
        flow_id,
        mock_prompt,
        inputs,
        conversation_history_key,
        mock_internal_event_client,
        base_flow_state,
        _mock_get_vars_from_state,
        _mock_maybe_compact_history,
    ):
        """Test that no reasoning log is emitted when ON_AGENT_REASONING is not in ui_history events."""
        ui_history_no_reasoning = UIHistory(
            events=[UILogEventsAgent.ON_TOOL_EXECUTION_SUCCESS],
            writer_class=agent_tools_ui_log_writer_class(component_name="test_agent"),
        )
        node = AgentNode(
            flow_id=flow_id,
            flow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
            name="test_agent_node",
            prompt=mock_prompt,
            inputs=inputs,
            conversation_history_key=RuntimeIOKey(
                alias="conversation_history", factory=lambda _: conversation_history_key
            ),
            internal_event_client=mock_internal_event_client,
            invoke_config={},
            ui_history=ui_history_no_reasoning,
        )

        # Message must have tool_calls so _emit_reasoning is entered; the early
        # return at the "event not in events" check is then what we're covering.
        text_message = AIMessage(
            content="I will now look at the codebase.",
            tool_calls=[
                {
                    "id": "toolu_01",
                    "name": "read_file",
                    "args": {"file_path": "README.md"},
                }
            ],
        )
        mock_prompt.ainvoke = AsyncMock(return_value=text_message)

        result = await node.run(base_flow_state)

        reasoning_logs = [
            log
            for log in result.get(FlowStateKeys.UI_CHAT_LOG, [])
            if log["message_type"] == MessageTypeEnum.AGENT
        ]
        assert len(reasoning_logs) == 0

    @pytest.mark.asyncio
    async def test_no_reasoning_emitted_for_litellm_placeholder(
        self,
        mock_prompt,
        agent_node_with_ui_history,
        base_flow_state,
        _mock_get_vars_from_state,
        _mock_maybe_compact_history,
    ):
        """LiteLLM placeholder injected into empty text content must not appear in the UI chat log."""
        placeholder_message = AIMessage(
            content=_LITELLM_EMPTY_CONTENT_PLACEHOLDER,
            tool_calls=[
                {
                    "id": "toolu_01",
                    "name": "read_file",
                    "args": {"file_path": "README.md"},
                }
            ],
        )
        mock_prompt.ainvoke = AsyncMock(return_value=placeholder_message)

        result = await agent_node_with_ui_history.run(base_flow_state)

        reasoning_logs = [
            log
            for log in result.get(FlowStateKeys.UI_CHAT_LOG, [])
            if log["message_type"] == MessageTypeEnum.AGENT
        ]
        assert len(reasoning_logs) == 0

    def test_non_str_non_list_content_returns_empty(self):
        """Content that is neither str nor list returns an empty string."""
        message = AIMessage(content="placeholder")
        # Bypass the normal str path by directly setting content to an unexpected type
        message.content = 42
        assert AgentNode._extract_text(message) == ""


class TestPredefinedRuntimeVariables:
    """Unit tests for AgentNode._predefined_runtime_variables."""

    def test_returns_expected_keys_with_valid_values(self):
        result = AgentNode._predefined_runtime_variables()
        assert set(result.keys()) == {
            "current_date",
            "current_time",
            "current_timezone",
        }
        datetime.strptime(result["current_date"], "%Y-%m-%d")
        datetime.strptime(result["current_time"], "%H:%M:%S")
        assert isinstance(result["current_timezone"], str)

    @pytest.mark.asyncio
    async def test_runtime_variables_included_in_prompt_invocation(
        self,
        mock_prompt,
        agent_node,
        base_flow_state,
        _mock_get_vars_from_state,
        _mock_maybe_compact_history,
    ):
        """_predefined_runtime_variables keys are passed to the prompt ainvoke call."""
        await agent_node.run(base_flow_state)

        invoked_input = mock_prompt.ainvoke.call_args[1]["input"]
        assert "current_date" in invoked_input
        assert "current_time" in invoked_input
        assert "current_timezone" in invoked_input

    @pytest.mark.parametrize(
        "component_cls",
        [
            "duo_workflow_service.agent_platform.v1.components.agent.component.AgentComponentBase",
            "duo_workflow_service.agent_platform.v1.components.one_off.component.OneOffComponent",
        ],
    )
    def test_runtime_injected_vars(self, component_cls):
        """_RUNTIME_INJECTED_VARS includes history, current_time, and current_timezone."""

        module_path, cls_name = component_cls.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_path), cls_name)
        assert cls._RUNTIME_INJECTED_VARS >= {
            "history",
            "current_date",
            "current_time",
            "current_timezone",
        }


class TestAgentNodeInvokeConfig:
    """Test suite for AgentNode invoke_config (TAG_NOSTREAM) support."""

    @pytest.mark.asyncio
    async def test_invoke_config_forwarded_to_ainvoke(
        self,
        flow_id,
        mock_prompt,
        inputs,
        conversation_history_key,
        mock_internal_event_client,
        base_flow_state,
        _mock_get_vars_from_state,
        _mock_maybe_compact_history,
        _mock_predefined_runtime_variables,
        prompt_variables,
    ):
        """invoke_config is passed as config= to every ainvoke call."""
        config = {"tags": ["some tag"]}
        node = AgentNode(
            flow_id=flow_id,
            flow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
            name="test_agent_node",
            prompt=mock_prompt,
            inputs=inputs,
            conversation_history_key=RuntimeIOKey(
                alias="conversation_history",
                factory=lambda _: conversation_history_key,
            ),
            internal_event_client=mock_internal_event_client,
            invoke_config=config,
        )

        await node.run(base_flow_state)

        mock_prompt.ainvoke.assert_called_once_with(
            input={
                **prompt_variables,
                "history": [],
                **FAKE_RUNTIME_VARS,
            },
            config=config,
        )
